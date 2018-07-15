import csv
import os
import sys
import unicodedata
import nltk
import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegressionCV as LogitCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize


FILEDIR = os.path.dirname(os.path.realpath(__file__)) + '/'
DOCUMENTS = FILEDIR+'data-documents/'
PYTHONVERSION = sys.version[0]


def write(msg, comm=None):
  '''writes to std out
  Args:
    msg: string
    comm: MPI Communicator (will not write if not root process)
  Returns:
    length of msg
  '''
  
  if comm is None or not comm.rank:
    sys.stdout.write(msg)
    sys.stdout.flush()
  return len(msg)


def ranksize(comm=None):
  '''returns rank and size of MPI Communicator
  Args:
    comm: MPI Communicator
  Returns:
    int, int
  '''

  if comm is None:
    return 0, 1
  return comm.rank, comm.size


def isroot(comm=None):
  '''checks whether process is root process
  Args:
    comm: MPI Communicator
  Returns:
    bool
  '''

  return comm is None or not comm.rank


def checkpoint(comm=None):
  '''waits until all processes have reached this point
  Args:
    comm: MPI Communicator
  '''

  if not comm is None:
    comm.allgather(0)


def splitlist(L, comm=None):
  '''splits list among processes
  Args:
    L: list to split
    comm: MPI Communicator
  Returns:
    list of size len(L)/nproc
  '''

  rank, size = ranksize(comm)
  if size == 1:
    return L
  return L[round(rank/size*len(L)):round((rank+1)/size*len(L))]


def txt2unicode(text):
  '''converts text to unicode
  Args:
    text: string
  Returns:
    unicode
  '''

  return unicode(unicodedata.normalize('NFKD', unicode(text, 'utf-8')).encode('ascii', 'ignore'))


def csv2clf(filename, unsup=False, splitlabel=False, delimiter='\t'):
  '''loads CSV file of form label\tdocument
  Args:
    filename: string of CSV filepath
  Returns:
    [list of documents, list of labels]
  '''

  with open(filename, 'r') as f:
    if PYTHONVERSION == '3':
      if splitlabel:
        return list(zip(*((document, label.split(':')[0]) for label, document in csv.reader(f, delimiter=delimiter))))
      return list(zip(*(row[::-1] for row in csv.reader(f, delimiter=delimiter))))
    if splitlabel:
      return list(zip(*((txt2unicode(document), label.split(':')[0]) for label, document in csv.reader(f, delimiter=delimiter))))
    return list(zip(*((txt2unicode(document), label) for label, document in csv.reader(f, delimiter=delimiter))))


def sst(partitions=['train', 'test']):
  '''loads binary Stanford Sentiment Treebank sentiment classification dataset
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings
  Returns:
    ((list of documents, list of labels) for each partition)
  '''

  if type(partitions) == str:
    return csv2clf(DOCUMENTS+'sst_'+partitions+'.csv')
  return [csv2clf(DOCUMENTS+'sst_'+partition+'.csv') for partition in partitions]


def sst_fine(partitions=['train', 'test']):
  '''loads fine-grained Stanford Sentiment Treebank sentiment classification dataset
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings
  Returns:
    ((list of documents, list of labels) for each partition)
  '''
  
  return sst(['fine_'+partition for partition in partitions])


def imdb(partitions=['train', 'test']):
  '''loads Internet Movie Database sentiment classification dataset
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings; if 'unsup' loads unsupervised corpus
  Returns:
    ((list of documents, list of labels) for each partition)
  '''

  if partitions == 'unsup':
    with open(DOCUMENTS+'imdb_unsup.txt', 'r') as f:
      return [line.strip() for line in f]
  if type(partitions) == str:
    return csv2clf(DOCUMENTS+'imdb_'+partitions+'.csv')
  return [csv2clf(DOCUMENTS+'imdb_'+partition+'.csv') for partition in partitions]


def ng(partitions=['train', 'test']):
  '''loads 20 NewsGroups topic classification dataset
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings
  Returns:
    ((list of documents, list of labels) for each partition)
  '''

  if type(partitions) == str:
    data = fetch_20newsgroups(subset=partitions)
    return data['data'], list(data['target'])
  output = []
  for partition in partitions:
    data = fetch_20newsgroups(subset=partition)
    output.append((data['data'], list(data['target'])))
  return output


def trec(partitions=['train', 'test'], splitlabel=True):
  '''loads TREC question classification dataset
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings
    splitlabel: whether to use only first part of label
  Returns:
    ((list of documents, list of labels) for each partition)
  '''

  if type(partitions) == str:
    return csv2clf(DOCUMENTS+'trec_'+partitions+'.csv', splitlabel=splitlabel)
  return [csv2clf(DOCUMENTS+'trec_'+partition+'.csv', splitlabel=splitlabel) for partition in partitions]


def txt2clf(*args):
  '''loads datasets with labels split by filename
  Args:
    filenames: list of filenames
  Returns:
    (list of documents, list of labels)
  '''

  documents = []
  labels = []
  for i, filename in enumerate(args):
    try:
      with open(filename, 'r') as f:
        if PYTHONVERSION == '3':
          documents.extend(line.strip() for line in f)
        else:
          documents.extend(txt2unicode(line.strip()) for line in f)
    except UnicodeDecodeError:
      with open(filename, 'rb') as f:
        documents.extend(line.decode(errors='ignore').strip() for line in f)
    labels.extend([i]*(len(documents)-len(labels)))
  return documents, labels


def mr():
  '''loads Customer Review dataset
  Returns:
    (list of documents, list of labels)
  '''

  return txt2clf(DOCUMENTS+'rt-polarity.neg', DOCUMENTS+'rt-polarity.pos')


def cr():
  '''loads Customer Review dataset
  Returns:
    (list of documents, list of labels)
  '''

  return txt2clf(DOCUMENTS+'custrev.neg', DOCUMENTS+'custrev.pos')


def subj():
  '''loads subjectivity dataset
  Returns:
    (list of documents, list of labels)
  '''

  return txt2clf(DOCUMENTS+'subj.objective', DOCUMENTS+'subj.subjective')


def mpqa():
  '''loads MPQA Opinion Corpus dataset
  Returns:
    (list of documents, list of labels)
  '''

  return txt2clf(DOCUMENTS+'mpqa.neg', DOCUMENTS+'mpqa.pos')


def sick(partition, similarity=False):
  '''loads data from single SICK partition
  Args:
    partition: component of data to load
    similarity: load similarity labels (as float); otherwise loads entailment labels (as str)
  Returns:
    (list of documents, list of documents, list of labels)
  '''

  if similarity:
    row2label = lambda row: float(row[3])
  else:
    row2label = lambda row: row[4]

  with open(DOCUMENTS + 'SICK_' + partition + '.txt', 'r') as f:
    f.readline()
    if PYTHONVERSION == '3':
      return list(zip(*((row[1], row[2], row2label(row)) for row in csv.reader(f, delimiter='\t'))))
    return list(zip(*((txt2unicode(row[1]), txt2unicode(row[2]), row2label(row)) for row in csv.reader(f, delimiter='\t'))))


def sick_e(partitions=['train', 'test']):
  '''loads data for SICK-Entailment task
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings
  Returns:
    ((list of documents, list of documents, list of labels) for each partition)
  '''

  if type(partitions) == str:
    return sick(partitions)
  return [sick(partition) for partition in partitions]
  

def sick_r(partitions=['train', 'test']):
  '''loads data for SICK-Entailment task
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings
  Returns:
    ((list of documents, list of documents, list of labels) for each partition)
  '''

  if type(partitions) == str:
    return sick(partitions, True)
  return [sick(partition, True) for partition in partitions]


def mrpc(partitions=['train', 'test']):
  '''loads data for MRPC task
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings
  Returns:
    (list of documents, list of documents, list of labels)
  '''

  if type(partitions) == str:
    with open(DOCUMENTS + 'msr_paraphrase_' + partitions + '.txt', 'r') as f:
      f.readline()
      if PYTHONVERSION == '3':
        return list(zip(*((row[-2], row[-1], row[0]) for row in csv.reader(f, delimiter='\t'))))
      return list(zip(*((txt2unicode(row[-2]), txt2unicode(row[-1]), row[0]) for row in csv.reader(f, delimiter='\t'))))
  return [mrpc(partition) for partition in partitions]


def sts(partitions=['train', 'test']):
  '''loads data for STS 2012-2017 collected sentence similarity tasks
  Args:
    partitions: component(s) of data to load; can be a string (for one partition) or list of strings
  Returns:
    ((list of documents, list of documents, list of labels) for each partition)
  '''

  if type(partitions) == str:
    with open(DOCUMENTS + 'sts-' + partitions + '.csv', 'r') as f:
      if PYTHONVERSION == '3':
        return list(zip(*((row[5], row[6], row[4]) for row in (line.strip().split('\t') for line in f))))
      return list(zip(*((txt2unicode(row[5]), txt2unicode(row[6]), row[4]) for row in (line.strip().split('\t') for line in f))))
  return [sts(partition) for partition in partitions]


TASKMAP = {'train-test split': {'sst': sst, 'sst_fine': sst_fine, 'imdb': imdb, 'ng': ng, 'trec': trec},
           'cross-validation': {'mr': mr, 'cr': cr, 'subj': subj, 'mpqa': mpqa},
           'pairwise task': {'sick_e': sick_e, 'sick_r': sick_r, 'mrpc': mrpc, 'sts': sts}}


def batched_build(documents, transform, info=(), root='', batchsize=None):
  '''constructs document representations
  Args:
    documents: list of strings
    transform: function that transforms list of documents to a matrix with len(documents) rows
    info: auxiliary info to pass to transform
    root: root of message to print to StdOut
    batchsize: number of documents to process at a time; if None processes all documents at once
  Returns:
    matrix of document representations with len(documents) rows
  '''
  
  if batchsize is None:
    if root:
      write(root+20*' ')
    return transform(documents, *info)
  offsets = np.arange(0, len(documents), batchsize)
  return np.vstack(transform(documents[offset:offset+batchsize], *info) for i, offset in enumerate(offsets) if not root or write(root+' Batch '+str(i+1)+'/'+str(len(offsets))+20*' '))


def evaluate(task, represent, prepare=None, batchsize=None, invariant=False, verbose=False, params=[10**i for i in range(-2, 3)], intercept=False, n_folds=2, n_jobs=-1, random_state=0):
  '''evaluates representation method on given task
  Args:
    task: string name of task
    represent: function that transforms list of documents to a matrix with len(documents) rows
    prepare: returns aggregate information used by represent (should be limited to n-gram vocab, NOT feature counts, etc.)
    batchsize: number of documents the represent should process at a time
    invariant: representation method does not depend on the batch (unlike e.g. SIF weighted features); if False must have batchsize is None
    verbose: print progress information
    params: cross-validation parameters
    intercept: whether to fit intercept in linear model
    n_folds: number of folds to use when cross-validating
    n_jobs: number of threads to run when cross-validating
    random_state: cross-validation seed
  Returns:
    if accuracy task: (train acc, test acc); if regression: (Pearson r, Spearman rho); if retrieval: (acc, F1)
  '''

  assert batchsize is None or invariant, "cannot construct in batches if not invariant"

  if task in TASKMAP['train-test split']:
    (dtrain, ltrain), (dtest, ltest) = TASKMAP['train-test split'][task]()
    info = () if prepare is None else prepare(dtrain+dtest)
    root = '\rBuilding '+task.upper()+' Train' if verbose else ''
    Xtrain = batched_build(dtrain, represent, info, root, batchsize)
    Ytrain = np.array(ltrain)
    root = '\rBuilding '+task.upper()+' Test' if verbose else ''
    Xtest = batched_build(dtest, represent, info, root, batchsize)
    Ytest = np.array(ltest)
    clf = LogitCV(Cs=params, fit_intercept=intercept, cv=n_folds, dual=np.less(*Xtrain.shape), solver='liblinear', n_jobs=n_jobs, random_state=random_state)
    if verbose:
      write('\rCross-Validating and Fitting '+task.upper()+10*' ')
    clf.fit(Xtrain, Ytrain)
    train = 100.0*clf.score(Xtrain, Ytrain)
    test = 100.0*clf.score(Xtest, Ytest)

  elif task in TASKMAP['cross-validation']:
    documents, labels = TASKMAP['cross-validation'][task]()
    info = () if prepare is None else prepare(documents)
    train = 0.0
    test = 0.0
    Y = np.array(labels)
    if invariant:
      root = '\rBuilding '+task.upper() if verbose else ''
      X = batched_build(documents, represent, info, root, batchsize)
      for i, (tr, te) in enumerate(StratifiedKFold(n_splits=10, random_state=random_state).split(X, Y)):
        if verbose:
          write('\rCross-Validating and Fitting '+task.upper()+' Fold '+str(i+1)+10*' ')
        clf = LogitCV(Cs=params, fit_intercept=intercept, cv=n_folds, dual=np.less(*X.shape), solver='liblinear', n_jobs=n_jobs, random_state=random_state)
        clf.fit(X[tr], Y[tr])
        train += clf.score(X[tr], Y[tr])
        test += clf.score(X[te], Y[te])
    else:
      for i, (tr, te) in enumerate(StratifiedKFold(n_splits=10, random_state=random_state).split(documents, Y)):
        root = '\rBuilding '+task.upper()+' Fold '+str(i+1)+' Train' if verbose else ''
        Xtrain = batched_build([documents[i] for i in tr], represent, info, root, batchsize)
        root = '\rBuilding '+task.upper()+' Fold '+str(i+1)+' Test' if verbose else ''
        Xtest = batched_build([documents[i] for i in te], represent, info, root, batchsize)
        if verbose:
          write('\rCross-Validating and Fitting '+task.upper()+' Fold '+str(i+1)+10*' ')
        clf = LogitCV(Cs=params, fit_intercept=intercept, cv=n_folds, dual=np.less(*Xtrain.shape), solver='liblinear', n_jobs=n_jobs, random_state=random_state)
        clf.fit(Xtrain, Y[tr])
        train += clf.score(Xtrain, Y[tr])
        test += clf.score(Xtest, Y[te])
    train *= 10.0
    test *= 10.0

  elif task in TASKMAP['pairwise task']:
    (d1train, d2train, ltrain), (d1test, d2test, ltest) = TASKMAP['pairwise task'][task]()
    info = () if prepare is None else prepare(d1train+d2train+d1test+d2test)
    root = '\rBuilding '+task.upper()+' Train' if verbose else ''
    Xtrain = batched_build(d1train+d2train, represent, info, root, batchsize)
    m = int(Xtrain.shape[0]/2)
    if task == 'sts':
      Ptrain = np.zeros(m)
      nz = norm(Xtrain[:m], axis=1) * norm(Xtrain[m:], axis=1) > 0.0
      Ptrain[nz] = np.sum(normalize(Xtrain[:m][nz]) * normalize(Xtrain[m:][nz]), axis=1)
    else:
      Xtrain = np.hstack([abs(Xtrain[:m]-Xtrain[m:]), Xtrain[:m]*Xtrain[m:]])
    root = '\rBuilding '+task.upper()+' Test' if verbose else ''
    Xtest = batched_build(d1test+d2test, represent, info, root, batchsize)
    m = int(Xtest.shape[0]/2)
    if task == 'sts':
      Ptest = np.zeros(m)
      nz = norm(Xtest[:m], axis=1) * norm(Xtest[m:], axis=1) > 0.0
      Ptest[nz] = np.sum(normalize(Xtest[:m][nz]) * normalize(Xtest[m:][nz]), axis=1)
    else:
      Xtest= np.hstack([abs(Xtest[:m]-Xtest[m:]), Xtest[:m]*Xtest[m:]])
    if verbose:
      write('\rCross-Validating and Fitting '+task.upper()+10*' ')
    if task in {'sick_r', 'sts'}:
      if task == 'sts':
        Ytest = np.array([float(y) for y in ltrain+ltest])
        P = np.concatenate([Ptrain, Ptest])
      else:
        Ytrain = np.array([float(y) for y in ltrain])
        Ytest = np.array([float(y) for y in ltest])
        reg = RidgeCV(alphas=params, fit_intercept=intercept)
        reg.fit(Xtrain, Ytrain)
        P = reg.predict(Xtest)
      r = 100.0*pearsonr(Ytest, P)[0]
      rho = 100.0*spearmanr(Ytest, P)[0]
      if verbose:
        write('\r'+task.upper()+': r='+str(r)+', rho='+str(rho)+10*' '+'\n')
      return r, rho
    else:
      clf = LogitCV(Cs=params, fit_intercept=intercept, cv=n_folds, dual=np.less(*Xtrain.shape), solver='liblinear', n_jobs=n_jobs, random_state=random_state)
      if task == 'mrpc':
        Ytrain = np.array([int(y) for y in ltrain])
        Ytest = np.array([int(y) for y in ltest])
        clf.fit(Xtrain, Ytrain)
        acc = 100.0*clf.score(Xtest, Ytest)
        f1 = 100.0*f1_score(Ytest, clf.predict(Xtest))
        if verbose:
          write('\r'+task.upper()+': Acc='+str(acc)+', F1='+str(f1)+10*' '+'\n')
        return acc, f1
      else:
        Ytrain = np.array(ltrain)
        Ytest = np.array(ltest)
        clf.fit(Xtrain, Ytrain)
        train = 100.0*clf.score(Xtrain, Ytrain)
        test = 100.0*clf.score(Xtest, Ytest)

  else:
    raise(NotImplementedError)

  if verbose:
    write('\r'+task.upper()+': Train Acc='+str(train)+', Test Acc='+str(test)+10*' '+'\n')
  return train, test
