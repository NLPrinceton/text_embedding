import sys
import nltk
import numpy as np
from numpy.fft import fft
from numpy.fft import ifft
from scipy import sparse as sp
from text_embedding.documents import *
from text_embedding.features import *
from text_embedding.vectors import *


# NOTE: filepath for Amazon GloVe embeddings goes here
VECTORFILES[('Amazon', 'GloVe', 1600)] = '/n/fs/nlpdatasets/AmazonProductData/amazon_glove1600.txt'


def BonC(n, min_count=1):
  prepare = lambda documents: ([True],)
  def represent(documents, vocab):
    docs = [tokenize(doc.lower() for doc in documents)]
    for k in range(1, n+1):
      docs.append([[tuple(sorted(gram)) for gram in nltk.ngrams(doc, k)] for doc in docs[0]])
      if vocab[0]:
        vocab.append(sorted({cooc for cooc, count in feature_counts(docs[k]).items() if count >= min_count}))
    vocab[0] = False
    return sp.hstack([docs2bofs(docs[k], vocabulary=vocab[k]) for k in range(1, n+1)], format='csr')
  return represent, prepare, True


def pointwise_mult(cooc, w2v):
  for i, word in enumerate(cooc):
    vec = w2v.get(word)
    if vec is None:
      return 0.0
    if i:
      output = output * scaling * vec
    else:
      output = vec
      scaling = np.sqrt(vec.shape[0])
  return output


def circular_conv(cooc, w2v):
  for i, word in enumerate(cooc):
    vec = w2v.get(word)
    if vec is None:
      return 0.0
    if i:
      output = output * fft(vec)
    else:
      output = fft(vec)
  return np.real(ifft(output))


def DisC(n, composition, scaling=True, vectorfile=None, corpus='Amazon', objective='GloVe', dimension=1600):
  prepare = lambda documents: (vocab2vecs({word for doc in documents for word in split_on_punctuation(doc.lower())}, vectorfile=vectorfile, corpus=corpus, objective=objective, dimension=dimension), np.zeros(dimension))
  compose = {'mult': pointwise_mult, 'conv': circular_conv}[composition]
  def represent(documents, w2v, z):
    docs = tokenize(doc.lower() for doc in documents)
    if scaling:
      return np.hstack(np.vstack(sum((compose(gram, w2v) for gram in nltk.ngrams(doc, k)), z) for doc in docs)/k for k in range(1, n+1))
    return np.hstack(np.vstack(sum((compose(gram, w2v) for gram in nltk.ngrams(doc, k)), z) for doc in docs) for k in range(1, n+1))
  return represent, prepare, True


if __name__ == '__main__':

  try:
    represent, prepare, invariant = DisC(int(sys.argv[2]), sys.argv[3])
  except IndexError:
    represent, prepare, invariant = BonC(int(sys.argv[2]))
  for task in sys.argv[1].split(','):
    evaluate(task, represent, prepare=prepare, invariant=invariant, verbose=True, intercept=task in TASKMAP['pairwise task'])
