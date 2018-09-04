import argparse
import os
import random
import sys
import time
import struct
from collections import Counter
from collections import deque
from operator import itemgetter
from tempfile import NamedTemporaryFile as NTF
import numpy as np
import SharedArray as sa
from numba import jit
from scipy import sparse as sp
from sklearn.linear_model import LinearRegression as LR
from text_embedding.documents import *


FLOAT = np.float32
INT = np.uint32
CHUNK = 1000000
STORE = 10*CHUNK
FMT = 'iif'
NBYTES = 12


def vocab_count(corpusfile, vocabfile, min_count=1, verbose=True, comm=None):
    '''counts word occurrences to determine vocabulary
    Args:
        corpusfile: corpus .txt file
        vocabfile: output .txt file
        min_count: minimum word count
        verbose: display progress
        comm: MPI Communicator
    Returns:
        None
    '''

    rank, size = ranksize(comm)
    if verbose:
        write('Counting Words with Minimum Count '+str(min_count)+'\n', comm)
        t = time.time()

    with open(corpusfile, 'r') as f:
        documents = (line for i, line in enumerate(f) if i%size == rank)

        counts = Counter(w for doc in documents for w in doc.split())
        if size > 1:
            counts = comm.reduce(counts, root=0)

    if not rank:
        vocab = sorted((item for item in counts.items() if item[1] >= min_count), key=itemgetter(1), reverse=True)
        if verbose:
            write('Counted '+str(len(vocab))+' Words, Time='+str(round(time.time()-t))+' sec\n')
        with open(vocabfile, 'w') as f:
            for word, count in vocab:
                f.write(word+' '+str(count)+'\n')
    checkpoint(comm)

@jit
def doc2cooc(indices, weights, window_size, V):

    row, col, val = [], [], []
    start = 0
    for i, index in enumerate(indices):
        if index != V:
            for w, other in zip(weights[start-i:], indices[start:i]):
                if other != V:
                    if index < other:
                        row.append(index)
                        col.append(other)
                    else:
                        row.append(other)
                        col.append(index)
                    val.append(w)
        start += i >= window_size
    return row, col, val

@jit
def doc2cooc_unweighted(indices, window_size, V):
    row, col = [], []
    start = 0
    for i, index in enumerate(indices):
        if index != V:
            for other in indices[start:i]:
                if other != V:
                    if index < other:
                        row.append(index)
                        col.append(other)
                    else:
                        row.append(other)
                        col.append(index)
        start += i >= window_size
    return row, col

def counts2bin(counts, f):

    for (i, j), v in counts.items():
        f.write(struct.pack(FMT, i, j, v))

def bin2counts(f, counts, subset):

    position = f.tell()
    ncooc = int((f.seek(0, 2)-position)/NBYTES)
    f.seek(position)

    for cooc in range(ncooc):
        i, j, v = struct.unpack(FMT, f.read(NBYTES))
        if i in subset:
            counts[(i, j)] += v

# NOTE: Result is highly non-random and contains only upper triangular entries
def cooc_count(corpusfile, vocabfile, coocfile, window_size=10, unweighted=False, verbose=True, comm=None):
    '''counts word cooccurrence in a corpus
    Args:
        corpusfile: corpus .txt file
        vocabfile: vocab .txt file
        coocfile: cooccurrence .bin file
        window_size: length of cooccurrence window
        unweighted: do not weight cooccurrence by distance
        verbose: display progress
        comm: MPI Communicator
    Returns:
        None
    '''

    rank, size = ranksize(comm)
    with open(vocabfile, 'r') as f:
        word2index = {line.split()[0]: INT(i) for i, line in enumerate(f)}
    if unweighted:
        one = FLOAT(1)
    else:
        weights = np.fromiter((1.0/d for d in reversed(range(1, window_size+1))), FLOAT, window_size)
    V = INT(len(word2index))
    counts = Counter()
    if verbose:
        write('\rCounting Cooccurrences with Window Size '+str(window_size)+'\n', comm)
        lines = 0
        t = time.time()

    if size > 1:
        random.seed(0)
        idx = list(range(V))
        random.shuffle(idx)
        start, stop = int(rank/size*V), int((rank+1)/size*V)
        subset = set(idx[start:stop])
        positions = [0]*size
        with open(corpusfile, 'r') as f:
            n = 0
            while True:
                v = None
                with NTF() as tmp:
                    dump = Counter()
                    files = comm.allgather(tmp.name)
                    for k, line in enumerate(f):
                        if k%size == rank:
                            doc = line.split()
                            if unweighted:
                                for i, j in zip(*doc2cooc_unweighted(np.fromiter((word2index.get(word, V) for word in doc), INT, len(doc)), window_size, V)):
                                    if i in subset:
                                        counts[(i, j)] += one
                                    else:
                                        dump[(i, j)] += one
                            else:
                                for i, j, v in zip(*doc2cooc(np.fromiter((word2index.get(word, V) for word in doc), INT, len(doc)), weights, window_size, V)):
                                    if i in subset:
                                        counts[(i, j)] += v
                                    else:
                                        dump[(i, j)] += v
                        if not (k+1)%CHUNK:
                            counts2bin(dump, tmp)
                            dump = Counter()
                            if verbose:
                                write('\rProcessed '+str(n+k+1)+' Lines, Time='+str(round(time.time()-t))+' sec', comm)
                        if not (k+1)%STORE:
                            n += k+1
                            break
                    counts2bin(dump, tmp)
                    tmp.flush()
                    for k in range(2):
                        for i, name in enumerate(files):
                            if i != rank:
                                with open(name, 'rb') as g:
                                    g.seek(positions[i])
                                    bin2counts(g, counts, subset)
                                    positions[i] = g.tell() * (k == 0)
                        checkpoint(comm)
                        if verbose:
                            write('\rProcessed '+str(n)+' Lines, Time='+str(round(time.time()-t))+' sec', comm)
                if not comm.allreduce(int(not v is None)):
                    break
        if verbose:
            write('\rCounted '+str(comm.allreduce(len(counts.items())))+' Cooccurrences, Time='+str(round(time.time()-t))+' sec\n', comm)
        for k in range(size):
            if k == rank:
                mode = 'ab' if rank else 'wb'
                with open(coocfile, mode) as f:
                    counts2bin(counts, f)
            checkpoint(comm)

    else:
        with open(corpusfile, 'r') as f:
            for k, line in enumerate(f):
                doc = line.split()
                if unweighted:
                    for i, j in zip(*doc2cooc_unweighted(np.fromiter((word2index.get(word, V) for word in doc), INT, len(doc)), window_size, V)):
                        counts[(i, j)] += one
                else:
                    for i, j, v in zip(*doc2cooc(np.fromiter((word2index.get(word, V) for word in doc), INT, len(doc)), weights, window_size, V)):
                        counts[(i, j)] += v
                if verbose and not (k+1)%CHUNK:
                    write('\rProcessed '+str(k+1)+' Lines, Time='+str(round(time.time()-t))+' sec')
        if verbose:
            write('\rCounted '+str(len(counts.items()))+' Cooccurrences, Time='+str(round(time.time()-t))+' sec\n')
        with open(coocfile, 'wb') as f:
            counts2bin(counts, f)


# NOTE: Open using 'with ... as' to prevent too many open POSIX files
class SharedArrayManager:

    _shared = []

    def __init__(self, comm=None):

        self._comm = comm
        self._rank, self._size = ranksize(comm)

    def __enter__(self):

        return self

    def __exit__(self, *args):

        for array in self._shared:
            try:
                sa.delete(array)
            except FileNotFoundError:
                pass

    def create(self, array=None, dtype=None):

        comm, rank = self._comm, self._rank

        if rank:
            shared = sa.attach(comm.bcast(None, root=0))
        else:
            dtype = array.dtype if dtype is None else dtype
            if self._size == 1:
                return array.astype(dtype)
            filename = str(time.time())
            shared = sa.create(filename, array.shape, dtype=dtype)
            shared += array.astype(dtype)
            self._shared.append(comm.bcast(filename, root=0))

        checkpoint(comm)
        return shared


def splitcooc(f, ncooc=None):

    row = deque()
    col = deque()

    if ncooc is None:
        position = f.tell()
        ncooc = int((f.seek(0, 2)-position)/NBYTES)
        f.seek(position)
    
    for cooc in range(ncooc):
        i, j, xij = struct.unpack(FMT, f.read(NBYTES))
        row.append(INT(i))
        col.append(INT(j))
        yield FLOAT(xij)

    for idx in [row, col]:
        for cooc in range(ncooc):
            yield idx.popleft()

def symcooc(coocfile, comm=None):

    rank, size = ranksize(comm)

    with open(coocfile, 'rb') as f:
        flength = f.seek(0, 2)
        offset = int(flength*rank/size / NBYTES)
        ncooc = int(flength*(rank+1)/size / NBYTES) - offset
        f.seek(NBYTES*offset)
        coocs = splitcooc(f, ncooc)
        val = np.fromiter(coocs, FLOAT, ncooc)
        row = np.fromiter(coocs, INT, ncooc)
        col = np.fromiter(coocs, INT, ncooc)

    sym = row < col
    symcooc = ncooc + sum(sym)
    values, rowdata, coldata = [np.empty(symcooc, dtype=dtype) for dtype in [FLOAT, INT, INT]]
    values[:ncooc], rowdata[:ncooc], coldata[:ncooc] = val, row, col
    values[ncooc:], rowdata[ncooc:], coldata[ncooc:] = val[sym], col[sym], row[sym]
    return values, rowdata, coldata


# NOTE: Open using 'with ... as' to prevent too many open POSIX files
class GloVe(SharedArrayManager):

    def _load_cooc_data(self, coocfile, alpha, xmax):

        data, self.row, self.col = symcooc(coocfile, self._comm)
        self.logcooc = np.log(data)
        data /= FLOAT(xmax)
        mask = data<1.0
        data[mask] **= FLOAT(alpha)
        data[~mask] = FLOAT(1.0)
        self.weights = data
        self.ncooc = data.shape[0]

    def _shuffle_cooc_data(self, seed):

        for data in [self.row, self.col, self.weights, self.logcooc]:
            np.random.seed(seed)
            np.random.shuffle(data)

    @staticmethod
    def _shapes(V, d):

        return [(V, d)]*2 + [(V,)]*2

    def _init_vecs(self, shapes, d, seed, init):

        create = self.create
        if self._rank:
            self._params = [create() for shape in shapes]
        elif init is None:
            np.random.seed(seed)
            self._params = [create((np.random.rand(*shape)-0.5)/d, dtype=FLOAT) for shape in shapes]
        else:
            self._params = [create(param, dtype=FLOAT) for param in init]

    def __init__(self, coocfile, V=None, d=None, seed=None, init=None, alpha=0.75, xmax=100.0, comm=None):
        '''
        Args:
          coocfile: binary cooccurrence file (assumed to have only upper triangular entries)
          V: vocab size
          d: vector dimension
          seed: random seed for initializing vectors
          init: tuple of numpy arrays to initialize parameters
          alpha: GloVe weighting parameter
          xmax: GloVe max cooccurrence parameter
          comm: MPI Communicator
        '''

        super().__init__(comm=comm)
        self._load_cooc_data(coocfile, alpha, xmax)
        assert not (init is None and (V is None or d is None)), "'V' and 'd' must be defined if 'init' not given"
        self._init_vecs(self._shapes(V, d), d, seed, init)

    def embeddings(self):
        '''returns GloVe embeddings using current parameters
        Returns:
            numpy array of size V x d
        '''

        return sum(self._params[:2]) / FLOAT(2.0)

    def dump(self, f):
        '''dumps GloVe embeddings to binary file
        Args:
            f: open file object or filename string
        Returns:
            None
        '''

        if not self._rank:
            self.embeddings().tofile(f)

    @staticmethod
    @jit
    def predict(i, j, wv, cv, wb, cb):
        
      return np.dot(wv[i].T, cv[j])+wb[i]+cb[j]

    _numpar = 4

    def loss(self):

        row, col = self.row, self.col
        ncooc = self.ncooc
        checkpoint(self._comm)
        params = self._params[:self._numpar]
        predict = self.predict
        errors = np.fromiter((predict(i, j, *params) for i, j in zip(row, col)), FLOAT, ncooc) - self.logcooc
        loss = np.inner(self.weights*errors, errors)
        if self._size > 1:
            ncooc = self._comm.allreduce(ncooc)
            return self._comm.allreduce(loss/ncooc)
        return loss/ncooc

    @staticmethod
    @jit
    def sgd_epoch(row, col, weights, logcoocs, wv, cv, wb, cb, ncooc, eta):

        etax2 = FLOAT(2.0*eta)
        loss = FLOAT(0.0)
        for i, j, weight, logcooc in zip(row, col, weights, logcoocs):
            wvi, cvj, wbi, cbj = wv[i], cv[j], wb[i], cb[j]
            error = np.dot(wvi.T, cvj) + wbi + cbj - logcooc
            werror = weight*error
            coef = werror*etax2
            upd = coef*cvj
            cvj -= coef*wvi
            wvi -= upd
            wbi -= coef
            cbj -= coef
            loss += werror*error
        return loss / ncooc

    def sgd(self, epochs=25, eta=0.01, seed=None, verbose=True, cumulative=True):
        '''runs SGD on GloVe objective
        Args:
          epochs: number of epochs
          eta: learning rate
          seed: random seed for cooccurrence shuffling
          verbose: write loss and time information
          cumulative: compute cumulative loss instead of true loss; ignored if not verbose
        Returns:
          None
        '''

        comm, rank, size = self._comm, self._rank, self._size
        random.seed(seed)

        if verbose:
            write('\rRunning '+str(epochs)+' Epochs of SGD with Learning Rate '+str(eta)+'\n', comm)
            if not cumulative:
                write('\rInitial Loss='+str(self.loss())+'\n', comm)
        ncooc = comm.allreduce(self.ncooc)

        t = time.time()
        for ep in range(epochs):

            if verbose:
                write('Epoch '+str(ep+1), comm)

            self._shuffle_cooc_data(random.randint(0, 2**32-1))
            loss = self.sgd_epoch(self.row, self.col, self.weights, self.logcooc, *self._params, ncooc, eta)

            if verbose:
                loss = comm.allreduce(loss) if cumulative else self.loss()
            checkpoint(comm)
            if verbose:
                write(': Loss='+str(loss)+', Time='+str(round(time.time()-t))+' sec\n', comm)
                t = time.time()

    @staticmethod
    @jit
    def adagrad_epoch(row, col, weights, logcoocs, wv, cv, wb, cb, ssg_wv, ssg_cv, ssg_wb, ssg_cb, ncooc, eta):

        eta = FLOAT(eta)
        two = FLOAT(2.0)
        loss = FLOAT(0.0)
        for i, j, weight, logcooc in zip(row, col, weights, logcoocs):
            wvi, cvj, wbi, cbj = wv[i], cv[j], wb[i], cb[j]
            ssg_wvi, ssg_cvj, ssg_wbi, ssg_cbj = ssg_wv[i], ssg_cv[j], ssg_wb[i], ssg_cb[j]
            error = np.dot(wvi.T, cvj) + wbi + cbj - logcooc
            werror = weight*error
            coef = two*werror
            updi = coef*cvj
            updj = coef*wvi
            reg_wvi = np.sqrt(ssg_wvi)
            reg_cvj = np.sqrt(ssg_cvj)
            ssg_wvi += updi ** 2
            ssg_cvj += updj ** 2
            wvi -= eta * updi / reg_wvi
            cvj -= eta * updj / reg_cvj
            reg_wbi = np.sqrt(ssg_wbi)
            reg_cbj = np.sqrt(ssg_cbj)
            coefsq = coef ** 2
            ssg_wbi += coefsq
            ssg_cbj += coefsq
            coef *= eta
            wbi -= coef / reg_wbi
            cbj -= coef / reg_cbj
            loss += werror*error
        return loss / ncooc

    def adagrad(self, epochs=25, eta=0.05, seed=None, verbose=True, cumulative=True):
        '''runs AdaGrad on GloVe objective
        Args:
          epochs: number of epochs
          eta: learning rate
          seed: random seed for cooccurrence shuffling
          verbose: write loss and time information
          cumulative: compute cumulative loss instead of true loss; ignored if not verbose
        Returns:
          None
        '''

        comm, rank, size = self._comm, self._rank, self._size
        random.seed(seed)

        if not hasattr(self, '_ssg'):
            self._ssg = [self.create(np.ones(param.shape, dtype=FLOAT)) for param in self._params[:self._numpar]]

        if verbose:
            write('\rRunning '+str(epochs)+' Epochs of AdaGrad with Learning Rate '+str(eta)+'\n', comm)
            if not cumulative:
                write('\rInitial Loss='+str(self.loss())+'\n', comm)
        ncooc = comm.allreduce(self.ncooc)

        t = time.time()
        for ep in range(epochs):

            if verbose:
                write('Epoch '+str(ep+1), comm)

            self._shuffle_cooc_data(random.randint(0, 2**32-1))
            loss = self.adagrad_epoch(self.row, self.col, self.weights, self.logcooc, *self._params, *self._ssg, ncooc, eta)

            if verbose:
                loss = comm.allreduce(loss) if cumulative else self.loss()
            checkpoint(comm)
            if verbose:
                write(': Loss='+str(loss)+', Time='+str(round(time.time()-t))+' sec\n', comm)
                t = time.time()


# NOTE: Open using 'with ... as' to prevent too many open POSIX files
class SN(GloVe):

    @staticmethod
    def _shapes(V, d): 
        
        return [(V, d), (1,)]

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

    def embeddings(self):
        
        return self._params[0]

    @staticmethod
    @jit
    def predict(i, j, wv, b):
        
        sumij = wv[i] + wv[j]
        return np.dot(sumij.T, sumij) + b[0]

    _numpar = 2

    @staticmethod
    @jit
    def sgd_epoch(row, col, weights, logcoocs, wv, b, ncooc, eta):

        etax2 = FLOAT(2.0*eta)
        two = FLOAT(2.0)
        loss = FLOAT(0.0)
        for i, j, weight, logcooc in zip(row, col, weights, logcoocs):
            wvi, wvj = wv[i], wv[j]
            sumij = wvi + wvj
            error = np.dot(sumij.T, sumij) + b[0] - logcooc
            werror = weight*error
            coef = werror*etax2
            b -= coef
            upd = (two*coef)*sumij
            wvi -= upd
            wvj -= upd
            loss += werror * error
        return loss / ncooc

    @staticmethod
    @jit
    def adagrad_epoch(row, col, weights, logcoocs, wv, b, ssg_wv, ssg_b, ncooc, eta):

        eta = FLOAT(eta)
        two = FLOAT(2.0)
        loss = FLOAT(0.0)
        for i, j, weight, logcooc in zip(row, col, weights, logcoocs):
            wvi, wvj = wv[i], wv[j]
            ssg_wvi, ssg_wvj = ssg_wv[i], ssg_wv[j]
            sumij = wvi + wvj
            error = np.dot(sumij.T, sumij) + b[0] - logcooc
            werror = weight*error
            coef = two*werror
            reg_b = np.sqrt(ssg_b)
            ssg_b += coef ** 2
            b -= eta*coef
            upd = (two*coef)*sumij
            updsq = upd ** 2
            reg_wvi = np.sqrt(ssg_wvi)
            ssg_wvi += updsq
            reg_wvj = np.sqrt(ssg_wvj)
            ssg_wvj += updsq
            upd *= eta
            wvi -= upd / reg_wvi
            wvj -= upd / reg_wvj
            loss += werror * error
        return loss / ncooc


# NOTE: Open using 'with ... as' to prevent too many open POSIX files
class RegularizedGloVe(GloVe):

    def _word_cooc_counts(self, V):
        
        counts = Counter(self.row)+Counter(self.col)
        array = np.fromiter((counts[i] for i in range(V)), INT, V)
        if self._size > 1:
            output = None if self._rank else np.empty(V, dtype=INT)
            self._comm.Reduce(array, output, root=0)
            return output
        return array

    def __init__(self, src, *args, reg=1.0, **kwargs):

        super().__init__(*args, **kwargs)
        create = self.create
        params = self._params
        params.append(create(src, dtype=FLOAT))
        params.append(FLOAT(reg))
        params.append(create(self._word_cooc_counts(src.shape[0]), dtype=FLOAT))

        oloss = self.loss
        if self._rank:
            self.loss = lambda: oloss() + self._comm.bcast(None, root=0)
        else:
            rloss = lambda: reg/src.shape[0]*norm(self.embeddings()-src)**2
            if self._size > 1:
                self.loss = lambda: oloss() + self._comm.bcast(rloss(), root=0)
            else:
                self.loss = lambda: oloss() + rloss()

    @staticmethod
    @jit
    def sgd_epoch(row, col, weights, logcoocs, wv, cv, wb, cb, src, reg, wcc, ncooc, eta):

        etax2 = FLOAT(2.0*eta)
        two = FLOAT(2.0)
        regoV = FLOAT(reg / wcc.shape[0])
        regcoef = FLOAT(eta * ncooc * regoV)
        oloss = FLOAT(0.0)
        rloss = FLOAT(0.0)
        for i, j, weight, logcooc in zip(row, col, weights, logcoocs):
            wvi, cvj, wbi, cbj, wcci, wccj = wv[i], cv[j], wb[i], cb[j], wcc[i], wcc[j]
            error = np.dot(wvi.T, cvj) + wbi + cbj - logcooc
            werror = weight*error
            coef = werror*etax2
            diffi = (wvi+cv[i])/two - src[i]
            diffj = (wv[j]+cvj)/two - src[j]
            upd = coef*cvj + (regcoef/wcci)*diffi
            cvj -= coef*wvi + (regcoef/wccj)*diffj
            wvi -= upd
            wbi -= coef
            cbj -= coef
            oloss += werror*error
            rloss += np.dot(diffi.T, diffi)/wcci + np.dot(diffj.T, diffj)/wccj
        return (oloss + regoV*rloss) / ncooc

    @staticmethod
    @jit
    def adagrad_epoch(row, col, weights, logcoocs, wv, cv, wb, cb, src, reg, wcc, ssg_wv, ssg_cv, ssg_wb, ssg_cb, ncooc, eta):

        eta = FLOAT(eta)
        two = FLOAT(2.0)
        regoV = FLOAT(reg / wcc.shape[0])
        regcoef = FLOAT(ncooc * regoV)
        oloss = FLOAT(0.0)
        rloss = FLOAT(0.0)
        for i, j, weight, logcooc in zip(row, col, weights, logcoocs):
            wvi, cvj, wbi, cbj, wcci, wccj = wv[i], cv[j], wb[i], cb[j], wcc[i], wcc[j]
            ssg_wvi, ssg_cvj, ssg_wbi, ssg_cbj = ssg_wv[i], ssg_cv[j], ssg_wb[i], ssg_cb[j]
            error = np.dot(wvi.T, cvj) + wbi + cbj - logcooc
            werror = weight*error
            coef = two*werror
            diffi = (wvi+cv[i])/two - src[i]
            diffj = (wv[j]+cvj)/two - src[j]
            updi = coef*cvj + (regcoef/wcci)*diffi
            updj = coef*wvi + (regcoef/wccj)*diffj
            reg_wvi = np.sqrt(ssg_wvi)
            reg_cvj = np.sqrt(ssg_cvj)
            ssg_wvi += updi ** 2
            ssg_cvj += updj ** 2
            wvi -= eta * updi / reg_wvi
            cvj -= eta * updj / reg_cvj
            reg_wbi = np.sqrt(ssg_wbi)
            reg_cbj = np.sqrt(ssg_cbj)
            coefsq = coef ** 2
            ssg_wbi += coefsq
            ssg_cbj += coefsq
            coef *= eta
            wbi -= coef / reg_wbi
            cbj -= coef / reg_cbj
            oloss += werror*error
            rloss += np.dot(diffi.T, diffi)/wcci + np.dot(diffj.T, diffj)/wccj
        return (oloss + regoV*rloss) / ncooc


# NOTE: Open using 'with ... as' to prevent too many open POSIX files
class RegularizedSN(SN, RegularizedGloVe):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @staticmethod
    @jit
    def sgd_epoch(row, col, weights, logcoocs, wv, b, src, reg, wcc, ncooc, eta):

        etax2 = FLOAT(2.0*eta)
        two = FLOAT(2.0)
        regoV = FLOAT(reg / wcc.shape[0])
        regcoef = FLOAT(etax2 * ncooc * regoV)
        oloss = FLOAT(0.0)
        rloss = FLOAT(0.0)
        for i, j, weight, logcooc in zip(row, col, weights, logcoocs):
            wvi, wvj, wcci, wccj = wv[i], wv[j], wcc[i], wcc[j]
            sumij = wvi + wvj
            error = np.dot(sumij.T, sumij) + b[0] - logcooc
            werror = weight*error
            coef = werror*etax2
            b -= coef
            diffi = wvi - src[i]
            diffj = wvj - src[j]
            upd = (two*coef)*sumij
            wvi -= upd + (regcoef/wcci)*diffi
            wvj -= upd + (regcoef/wccj)*diffj
            oloss += werror*error
            rloss += np.dot(diffi.T, diffi)/wcci + np.dot(diffj.T, diffj)/wccj
        return (oloss + regoV*rloss) / ncooc

    @staticmethod
    @jit
    def adagrad_epoch(row, col, weights, logcoocs, wv, b, src, reg, wcc, ssg_wv, ssg_b, ncooc, eta):

        eta = FLOAT(eta)
        two = FLOAT(2.0)
        regoV = FLOAT(reg / wcc.shape[0])
        regcoef = FLOAT(ncooc * regoV)
        oloss = FLOAT(0.0)
        rloss = FLOAT(0.0)
        for i, j, weight, logcooc in zip(row, col, weights, logcoocs):
            wvi, wvj, wcci, wccj = wv[i], wv[j], wcc[i], wcc[j]
            ssg_wvi, ssg_wvj = ssg_wv[i], ssg_wv[j]
            sumij = wvi + wvj
            error = np.dot(sumij.T, sumij) + b[0] - logcooc
            werror = weight*error
            coef = two*werror
            reg_b = np.sqrt(ssg_b)
            ssg_b += coef ** 2
            b -= eta*coef
            diffi = wvi - src[i]
            diffj = wvj - src[j]
            upd = (two*coef)*sumij
            updi = upd + (regcoef/wcci)*diffi
            updj = upd + (regcoef/wccj)*diffj
            regi = np.sqrt(ssg_wvi)
            regj = np.sqrt(ssg_wvj)
            ssg_wvi += updi ** 2
            ssg_wvj += updj ** 2
            wvi -= eta * updi
            wvj -= eta * updj
            oloss += werror*error
            rloss += np.dot(diffi.T, diffi)/wcci + np.dot(diffj.T, diffj)/wccj
        return (oloss + regoV*rloss) / ncooc


def align_params(params, srcvocab, tgtvocab, mean_fill=True):

    output = []
    for param in params:
        if len(param.shape) == 1:
            if param.shape[0] == 1:
                output.append(param)
                continue
            shape = (len(tgtvocab),)
            default = np.mean(param)
        else:
            shape = (len(tgtvocab), param.shape[1])
            default = np.mean(param, axis=0)
        array = np.empty(shape, dtype=FLOAT)
        if not mean_fill:
            default *= FLOAT(0.0)
        w2e = dict(zip(srcvocab, param))
        for i, w in enumerate(tgtvocab):
            array[i] = w2e.get(w, default)
        output.append(array)
    return output

def induce_embeddings(srcvocab, srccooc, srcvecs, tgtvocab, tgtcooc, comm=None):

    rank, size = ranksize(comm)
    Vsrc, d = srcvecs.shape
    Vtgt = len(tgtvocab)

    with SharedArrayManager(comm=comm) as sam:

        write('Loading Source Cooccurrences\n', comm)
        data, row, col = symcooc(srccooc, comm)
        srcvecs = sam.create(srcvecs, dtype=FLOAT)
        X = sp.csr_matrix((data, (row, col)), shape=(Vsrc, Vsrc), dtype=FLOAT)

        write('Computing Source Counts\n', comm)
        if size > 1:
            C = None if rank else np.empty(Vsrc, dtype=FLOAT)
            comm.Reduce(np.array(X.sum(1))[:,0], C, root=0)
            C = sam.create(C)
        else:
            C = np.array(X.sum(1))[:,0]

        write('Building Source Context Vectors\n', comm)
        if size > 1:
            U = None if rank else np.empty((Vsrc, d), dtype=FLOAT)
            comm.Reduce(X.dot(srcvecs), U, root=0)
            U = sam.create(U)
        else:
            U = X.dot(srcvecs)
        U = U[C>0]
        C = C[C>0]
        start, stop = int(rank/size*Vsrc), int((rank+1)/size*Vsrc)
        U[start:stop] /= C[start:stop, None]
        checkpoint(comm)
        
        write('Learning Induction Matrix\n', comm)
        M = sam.create(np.zeros((d, d), dtype=FLOAT))
        start, stop = int(rank/size*d), int((rank+1)/size*d)
        M[:,start:stop] = LR(fit_intercept=False).fit(X[:,start:stop], srcvecs).coef_
        checkpoint(comm)

        write('Loading Target Cooccurrences\n', comm)
        data, row, col = symcooc(tgtcooc, comm)
        tgt2idx = {w: i for i, w in enumerate(tgtvocab)}
        tgt2src = {tgt2idx.get(w): i for i, w in enumerate(srcvocab)}
        zero = FLOAT(0.0)
        for i, j in enumerate(col):
            try:
                col[i] = tgt2src[j]
            except KeyError:
                data[i] = zero
        X = sp.csr_matrix((data, (row, col)), shape=(Vtgt, Vsrc), dtype=FLOAT)
        X.eliminate_zeros()

        write('Computing Target Counts\n', comm)
        if size > 1:
            C = None if rank else np.empty(Vtgt, dtype=FLOAT)
            comm.Reduce(np.array(X.sum(1))[:,0], C, root=0)
            C = sam.create(C)
        else:
            C = np.array(X.sum(1))[:,0]

        write('Building Target Context Vectors\n', comm)
        rank, size = ranksize(comm)
        if size > 1:
            U = None if rank else np.empty((Vtgt, d), dtype=FLOAT)
            comm.Reduce(X.dot(srcvecs), U, root=0)
            U = sam.create(U)
        else:
            U = X.dot(srcvecs)
        nz = sum(C>0)
        start, stop = int(rank/size*nz), int((rank+1)/size*nz)
        U[C>0][start:stop] /= C[C>0][start:stop, None]

        write('Computing Induced Embeddings\n', comm)
        tgtvecs = sam.create(np.zeros((Vtgt, d), dtype=FLOAT))
        tgtvecs[start:stop] = U[start:stop].dot(M.T)
        checkpoint(comm)
        if not rank:
            return tgtvecs

def main(args, comm=None):

    if args.mode == 'vocab' or args.mode[:4] in 'thru':
        vocab_count(args.input, args.vocab, args.min_count, args.verbose, comm)

    if args.mode == 'cooc' or args.mode[:4] in 'thru':
        cooc_count(args.input, args.vocab, args.cooc, args.window_size, args.unweighted, args.verbose, comm)

    Embedding = GloVe if args.mode[-5:] == 'glove' else SN if args.mode[-2:] == 'sn' else None
    if Embedding is None:
        if not args.mode in {'vocab', 'cooc', 'thru-cooc'}:
            raise(NotImplementedError)
        return

    with open(args.vocab, 'r') as f:
        V = len(f.readlines())
    with Embedding(args.cooc, V, args.dimension, alpha=args.alpha, xmax=args.xmax, comm=comm) as E:
        if args.sgd:
            E.sgd(args.niter, args.eta, verbose=args.verbose)
        else:
            E.adagrad(args.niter, args.eta, verbose=args.verbose)
        E.dump(args.output)

def parse():

    parser = argparse.ArgumentParser(prog='python text_embeddings/solvers.py')
    parser.add_argument('mode', help="'vocab', 'cooc', 'glove', 'sn', 'thru-cooc', 'thru-glove', or 'thru-sn'")
    parser.add_argument('vocab', help='vocabulary .txt file')
    parser.add_argument('-i', '--input', help='corpus .txt file')
    parser.add_argument('-c', '--cooc', help='cooccurrence .bin file')
    parser.add_argument('-o', '--output', help='embedding .bin file')
    parser.add_argument('-m', '--min_count', default=1, help='minimum word count in corpus', type=int)
    parser.add_argument('-w', '--window_size', default=10, help='size of cooccurrence window', type=int)
    parser.add_argument('-u', '--unweighted', action='store_true', help='no distance weighting')
    parser.add_argument('-d', '--dimension', default=300, help='embedding dimension', type=int)
    parser.add_argument('-x', '--xmax', default=100.0, help='maximum cooccurrence', type=float)
    parser.add_argument('-a', '--alpha', default=0.75, help='weighting exponent', type=float)
    parser.add_argument('-s', '--sgd', action='store_true', help='use SGD')
    parser.add_argument('-n', '--niter', default=25, help='number of training epochs', type=int)
    parser.add_argument('-e', '--eta', default=0.05, help='learning rate', type=float)
    parser.add_argument('-v', '--verbose', action='store_true', help='display output')
    return parser.parse_args()

if __name__ == '__main__':

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None
    main(parse(), comm=comm)
