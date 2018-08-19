# mpirun -n $NUM_THREADS python text_embedding/solvers.py $VOCAB_FILE $COOC_FILE
import pdb
import random
import sys
import time
import struct
from collections import Counter
from collections import deque
import numpy as np
np.seterr(all='raise')
import SharedArray as sa
from scipy import sparse as sp
from text_embedding.documents import *
from text_embedding.testvecs import *


FLOAT = np.float32
INT = np.uint32
FMT = 'iif'
NBYTES = 12


def cooc2sparse(f, ncooc=None):

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


class GloVe:

    def _load_cooc_data(self, coocfile, alpha, xmax):

        rank, size = self._rank, self._size

        with open(coocfile, 'rb') as f:
            flength = f.seek(0, 2)
            offset = int(flength*rank/size / NBYTES)
            ncooc = int(flength*(rank+1)/size / NBYTES) - offset
            f.seek(NBYTES*offset)
            coocs = cooc2sparse(f, ncooc)
            data = np.fromiter(coocs, FLOAT, ncooc)
            row = np.fromiter(coocs, INT, ncooc)
            col = np.fromiter(coocs, INT, ncooc)

        sym = row < col
        self.ncooc = ncooc + sum(sym)
        self.row, self.col = np.empty(self.ncooc, dtype=INT), np.empty(self.ncooc, dtype=INT)
        self.row[:ncooc], self.col[:ncooc] = row, col
        self.row[ncooc:], self.col[ncooc:] = col[sym], row[sym]

        self.logcooc = np.empty(self.ncooc, dtype=FLOAT)
        self.logcooc[:ncooc] = np.log(data)
        self.logcooc[ncooc:] = self.logcooc[:ncooc][sym]

        data /= xmax
        mask = data<1
        data[mask] **= (alpha/2)
        data[~mask] = 1.0
        self.weights = np.empty(self.ncooc, dtype=FLOAT)
        self.weights[:ncooc] = data
        self.weights[ncooc:] = data[sym]

    def _shuffle_cooc_data(self, seed):

        for data in [self.row, self.col, self.weights, self.logcooc]:
            np.random.seed(seed)
            np.random.shuffle(data)

    def _create_shared_zeros(self, shape):

        comm, rank = self._comm, self._rank

        if rank:
            zeros = sa.attach(comm.bcast(None, root=0))
        elif self._size == 1:
            return np.zeros(shape, dtype=FLOAT)
        else:
            filename = str(time.time())
            zeros = sa.create(filename, shape, dtype=FLOAT)
            self._shared.append(comm.bcast(filename, root=0))

        checkpoint(comm)
        return zeros

    def __init__(self, V, d, coocfile, seed=None, alpha=0.75, xmax=100, comm=None):
        '''
        Args:
          V: vocab size
          d: vector dimension
          coocfile: binary cooccurrence file (assumed to have only upper triangular entries)
          seed: random seed for initializing vectors
          alpha: GloVe weighting parameter
          xmax: GloVe max cooccurrence parameter
          comm: MPI Communicator
        '''

        rank, size = ranksize(comm)
        self._comm, self._rank, self._size = comm, rank, size
        self.V, self.d = V, d
        self._load_cooc_data(coocfile, alpha, xmax)
        self.seed = seed

    def __enter__(self):

        V, d = self.V, self.d
        self._shared = []
        self.params = [self._create_shared_zeros(shape) for shape in [(V, d)]*2 + [V]*2]
        np.random.seed(self.seed)
        if not self._rank:
            for param in self.params:
                param += ((np.random.rand(*param.shape)-0.5)/d).astype(FLOAT)
        checkpoint(self._comm)
        return self

    def __exit__(self, *args):

        for array in self._shared:
            sa.delete(array)

    def embeddings(self):
        '''returns GloVe embeddings using current parameters
        Returns:
            numpy array of size V x d
        '''

        return sum(self.params[:2]) / 2.0

    def loss(self):

        row, col = self.row, self.col
        wv, cv, wb, cb = self.params
        ncooc = self.ncooc
        errors = np.fromiter((np.inner(wv[i], cv[j])+wb[i]+cb[j] for i, j in zip(row, col)), FLOAT, ncooc)
        errors -= self.logcooc
        errors *= self.weights
        loss = norm(errors) ** 2
        if self._size > 1:
            ncooc = self._comm.allreduce(ncooc)
            loss = self._comm.reduce(loss/ncooc, root=0)
        return loss

    def gradient(self):

        row, col = self.row, self.col
        wv, cv, wb, cb = self.params
        errors = np.fromiter((np.inner(wv[i], cv[j])+wb[i]+cb[j] for i, j in zip(row, col)), FLOAT, self.ncooc)
        errors -= self.logcooc
        errors *= self.weights

        V, d = self.V, self.d
        errors *= 2.0*self.weights
        E = sp.csr_matrix((errors, (row, col)), shape=(V, V), dtype=FLOAT)
        grad_wv, grad_cv, grad_b = E.dot(cv), E.dot(wv), E.dot(np.ones(V, dtype=FLOAT))

        comm, rank, size = self._comm, self._rank, self._size
        if size > 1:
            grads = []
            for grad in [grad_wv, grad_cv, grad_b]:
                if rank:
                    grads.append(None)
                else:
                    grads.append(np.empty(grad.shape, dtype=FLOAT))
                comm.Reduce(grad, grads[-1], root=0)
            grads.append(grads[-1])
        else:
            grads = [grad_wv, grad_cv, grad_b, grad_b]

        return grads

    def word_cooc_counts(self):
        
        counts = Counter(self.row)+Counter(self.col)
        array = np.fromiter((counts[i] for i in range(self.V)), INT, self.V)
        if self._size > 1:
            output = None if self._rank else np.empty(self.V, dtype=INT)
            self._comm.Reduce(array, output, root=0)
            return output
        return array

    def sgd_update(self, i, j, weight, logcooc, eta):
        
        wv, cv, wb, cb = self.params
        wvi, cvj, wbi, cbj = wv[i], cv[j], wb[i], cb[j]
        root = weight * (np.inner(wvi, cvj) + wbi + cbj - logcooc)
        coef = 2.0*weight*root*eta
        updi, updj = coef*cvj, coef*wvi
        wvi -= updi
        cvj -= updj
        wbi -= coef
        cbj -= coef
        return root * root

    def sgd(self, eta=0.01, epochs=25, seed=None, verbose=True, cumulative=True):
        '''runs SGD on GloVe objective
        Args:
          eta: learning rate
          epochs: number of epochs
          seed: random seed for cooccurrence shuffling
          verbose: write loss and time information
          cumulative: compute cumulative loss instead of true loss
        Returns:
          None
        '''

        comm, rank, size = self._comm, self._rank, self._size
        update = self.sgd_update
        shuffle = self._shuffle_cooc_data
        random.seed(seed)

        if verbose:
            write('\rRunning '+str(epochs)+' Epochs of SGD with LR='+str(eta)+'\n', comm)
        if cumulative or not verbose:
            t0 = 0
        else:
            t = time.time()
            write('\rInitial Loss='+str(self.loss())+'\n', comm)
            t0 = time.time()-t

        checkpoint(comm)
        t = time.time()

        for ep in range(epochs):

            epoch = '\rEpoch '+str(ep+1)+verbose*': '
            write(epoch, comm)
            shuffle(random.randint(0, 2**32-1))

            loss = 0.0
            for c, (i, j, wei, logx) in enumerate(zip(self.row, self.col, self.weights, self.logcooc)):
                loss += update(i, j, wei, logx, eta)
                if verbose and not (c+1)%100000:
                    write(epoch+'ETA '+str(round((self.ncooc-c-1)/(c+1)*(time.time()-t)+t0))+' sec'+int(np.log(self.ncooc))*' ', comm)

            if cumulative:
                if size > 1:
                    loss = comm.reduce(loss/(self.ncooc*size), root=0)
            elif verbose:
                write(epoch+'Loss, ET '+str(round(t0))+' sec', comm)
                loss = self.loss()
                checkpoint(comm)
            if verbose:
                write(epoch+'Loss='+str(loss)+', Time='+str(round(time.time()-t))+' sec\n', comm)
                t = time.time()
        write('\r', comm)


if __name__ == '__main__':

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    with open(sys.argv[1], 'r') as f:
        vocab = [line.split()[0] for line in f]

    V, d, epochs, interval = len(vocab), 300, 100, 10

    with GloVe(V, d, sys.argv[2], comm=comm) as g:
        for i in range(0, round(epochs/interval)):
            g.sgd(epochs=interval)
            if not g._rank:
                write('Epoch '+str((i+1)*interval)+' Evaluation\n')
                w2v = dict(zip(vocab, g.embeddings()))
                for task in ['IMDB', 'SST']:
                    tr, te = unigram_baseline(w2v, task.lower())
                    write('\t'+task+'\tTrain='+str(round(tr, 2))+'\tTest='+str(round(te, 2))+'\n')
