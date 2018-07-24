import h5py
import numpy as np
from numpy.linalg import norm
from scipy.linalg import svd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize


FLOAT = np.float32
# NOTE: filepath for Common Crawl GloVe embeddings goes here
VECTORFILES = {('CC', 'GloVe', 300): '/n/fs/nlpdatasets/glove.840B/glove.840B.300d.txt'}


# NOTE: Some files have 2d or 2d+2 numbers on each line, with the last d of them being meaningless; avoid loading them by setting dimension=d
def load(vectorfile, vocabulary=None, dimension=None):
  '''generates word embeddings from file
  Args:
    vectorfile: word embedding text file or HDF5 file with keys 'words' and 'vectors'
    vocabulary: dict/set of strings, or int specifying number of words to load; if None loads all words from file
    dimension: number of dimensions to load
  Returns:
    (word, vector) generator
  '''

  try:
    f = h5py.File(vectorfile, 'r')
    for word, vector in zip(f['words'], f['vectors']):
      if vocabulary is None or word in vocabulary:
        yield word, vector
    f.close()

  except OSError:
    if vocabulary is None:
      V = float('inf')
    elif type(vocabulary) == int:
      V = vocabulary
      vocabulary = None
    else:
      V = len(vocabulary)

    with open(vectorfile, 'r') as f:
      n = 0
      for line in f:
        index = line.index(' ')
        word = line[:index]
        if vocabulary is None or word in vocabulary:
          yield word, np.array([FLOAT(entry) for entry in line[index+1:].split()[:dimension]])
          n += 1
        if n == V:
          break


def vocab2mat(vocabulary=None, random=None, vectorfile=None, corpus='CC', objective='GloVe', dimension=300, unit=True):
  '''constructs matrix of word vectors
  Args:
    vocabulary: dict mapping strings to indices, or iterable of strings, or int specifying vocab size; if None loads all words in vectorfile
    random: type ('Gaussian' or 'Rademacher') of random vectors to use; if None uses pretrained vectors
    vectorfile: word embedding text file; ignored if not random is None
    corpus: corpus used to train embeddings; ignored if not random is None or not vectorfile is None
    objective: objective used to train embeddings; ignored if not random is None or not vectorfile is None
    dimension: embedding dimension
    unit: normalize embeddings
  Returns:
    numpy matrix of size (len(vocabulary), dimension)
  '''

  assert random is None or not vocabulary is None, "needs vocabulary size information for random vectors"

  if random is None:

    if vectorfile is None:
      vectorfile = VECTORFILES[(corpus, objective, dimension)]
    if type(vocabulary) == set:
      vocabulary = sorted(vocabulary)
    if type(vocabulary) == list:
      vocabulary = {word: i for i, word in enumerate(vocabulary)}
    if type(vocabulary) == dict:
      matrix = np.zeros((len(vocabulary), dimension), dtype=FLOAT)
      for word, vector in load(vectorfile, vocabulary, dimension):
        matrix[vocabulary[word]] = vector
    else:
      matrix = np.vstack(load(vectorfile, vocabulary, dimension))
  
  else:

    if not type(vocabulary) == int:
      vocabulary = len(vocabulary)
    if random.lower() == 'gaussian':
      matrix = np.random.normal(scale=1.0/np.sqrt(dimension), size=(vocabulary, dimension)).astype(FLOAT)
    elif random.lower() == 'rademacher':
      return (2.0*np.random.randint(2, size=(vocabulary, dimension)).astype(FLOAT)-1.0)/np.sqrt(dimension)
    else:
      raise(NotImplementedError)

  if unit:
    return normalize(matrix)
  return matrix


def vocab2vecs(vocabulary=None, random=None, vectorfile=None, corpus='CC', objective='GloVe', dimension=300, unit=True):
  '''constructs dict mapping words to vectors
  Args:
    vocabulary: iterable of strings, or int specifying vocab size; if None loads all words in vectorfile
    random: type ('Gaussian' or 'Rademacher') of random vectors to use; if None uses pretrained vectors
    vectorfile: word embedding text file; ignored if not random is None
    corpus: corpus used to train embeddings; ignored if not random is None or not vectorfile is None
    objective: objective used to train embeddings; ignored if not random is None or not vectorfile is None
    dimension: embedding dimension
    unit: normalize embeddings
  Returns:
    {word: vector} dict; words not in vectorfile are not included
  '''

  assert random is None or not (vocabulary is None or type(vocabulary) == int), "needs word information for random vectors"

  if random is None:
    if vectorfile is None:
      vectorfile = VECTORFILES[(corpus, objective, dimension)]
    if unit:
      return {word: vector/norm(vector) for word, vector in load(vectorfile, vocabulary, dimension)}
    return dict(load(vectorfile, vocabulary, dimension))
  return dict(zip(vocabulary, vocab2mat(vocabulary, random=random, dimension=dimension, unit=unit)))


def docs2vecs(documents, f2v=None, weights=None, default=1.0, avg=False, **kwargs):
  '''computes document embeddings from documents
  Args:
    documents: iterable of lists of hashable features
    f2v: dict mapping features to vectors; if None will compute this using vocab2vecs
    weights: dict mapping features to weights; unweighted if None
    default: default weight to assign if feature not in weights; ignored if weights is None
    avg: divide embeddings by the document length
    kwargs: passed to vocab2vecs; ignored if not f2v is None
  Returns:
    matrix of size (len(documents), dimension)
  '''

  if f2v is None:
    f2v = vocab2vecs({word for document in documents for word in documents}, **kwargs)
    dimension = kwargs.get('dimension', 300)
  else:
    dimensions = {v.shape for v in f2v.values()}
    assert len(dimensions) == 1, "all feature vectors must have same dimension"
    dimension = dimensions.pop()
  if not weights is None:
    f2v = {feat: weights.get(feat, default)*vec for feat, vec in f2v.items()}
    
  z = np.zeros(dimension, dtype=FLOAT)
  if avg:
    return np.vstack(sum((f2v.get(feat, z) for feat in document), z) / max(1.0, len(document)) for document in documents)
  return np.vstack(sum((f2v.get(feat, z) for feat in document), z) for document in documents)


class OrthogonalProcrustes:
  '''sklearn-style class for solving the Orthogonal Procrustes problem
  '''

  def __init__(self, fit_intercept=False):
    '''initializes object
    Args:
      fit_intercept: whether to find best transformation after translation
    Returns:
      None
    '''

    self.fit_intercept = fit_intercept

  def fit(self, X, Y):
    '''finds orthogonal matrix M minimizing |XM^T-Y|
    Args:
      X: numpy array of shape (n, d)
      Y: numpy array of shape (n, d)
    Returns:
      self (with attribute coef_, a numpy array of shape (d, d)
    '''

    if self.fit_intercept:
      Xbar, Ybar = np.mean(X, axis=0), np.mean(Y, axis=0)
      X, Y = X-Xbar, Y-Ybar
    U, _, VT = svd(Y.T.dot(X))
    self.coef_ = U.dot(VT)
    if self.fit_intercept:
      self.intercept_ = Ybar - self.coef_.dot(Xbar)
    else:
      self.intercept_ = np.zeros(self.coef_.shape[0], dtype=self.coef_.dtype)
    return self


def align_vocab(func):
  '''wrapper to align vocab to allow word-to-vector dict inputs to functions taking two word-vector matrices as inputs
  '''

  def wrapper(X, Y, **kwargs):
    assert type(X) == type(Y), "first two arguments must be the same type"
    if type(X) == dict:
      vocab = sorted(set(X.keys()).intersection(Y.keys()))
      X = np.vstack(X[w] for w in vocab)
      Y = np.vstack(Y[w] for w in vocab)
    else:
      assert type(X) == np.ndarray, "first two arguments must be 'dict' or 'numpy.ndarray'"
    return func(X, Y, **kwargs)

  return wrapper


@align_vocab
def best_transform(source, target, orthogonal=True, fit_intercept=False):
  '''computes best matrix between two sets of word embeddings in terms of least-squares error
  Args:
    source: numpy array of size (len(vocabulary), dimension) or dict mapping words to vectors; must be same type as target
    target: numpy array of size (len(vocabulary), dimension) or dict mapping words to vectors; must be same type as source
    orthogonal: if True constrains best transform to be orthogonal
    fit_intercept: whether to find best transformation after translation
  Returns:
    numpy array of size (dimension, dimension)
  '''

  if orthogonal:
    transform = OrthogonalProcrustes(fit_intercept=fit_intercept).fit(source, target)
  else:
    transform = LinearRegression(fit_intercept=fit_intercept).fit(source, target)
    if not fit_intercept:
      transform.intercept_ = np.zeros(target.shape[1])
  return transform.coef_.astype(target.dtype), transform.intercept_.astype(target.dtype)


@align_vocab
def average_cosine_similarity(X, Y):
  '''computes the average cosine similarity between two sets of word embeddings
  Args:
    X: numpy array of size (len(vocabulary), dimension) or dict mapping words to vectors; must be same type as target
    Y: numpy array of size (len(vocabulary), dimension) or dict mapping words to vectors; must be same type as source
  Returns:
    average cosine similarity as a float
  '''

  return np.mean((normalize(X) * normalize(Y)).sum(1))
