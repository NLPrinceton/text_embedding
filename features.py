from collections import Counter
from itertools import chain
from itertools import groupby
from operator import itemgetter
#from string import punctuation
from unicodedata import category
import nltk
import numpy as np
from scipy import sparse as sp


#PUNCTUATION = set(punctuation)
PUNCTUATION = {'M', 'P', 'S'}
UINT = np.uint16


def split_on_punctuation(document):
  '''tokenizes string by splitting on spaces and punctuation
  Args:
    document: string
  Returns:
    str generator
  '''

  for token in document.split():
    if len(token) == 1:
      yield token
    else:
      chunk = token[0]
      for char0, char1 in zip(token[:-1], token[1:]):
        #if (char0 in PUNCTUATION) == (char1 in PUNCTUATION):
        if (category(char0)[0] in PUNCTUATION) == (category(char1)[0] in PUNCTUATION):
          chunk += char1
        else:
          yield chunk
          chunk = char1
      if chunk:
        yield chunk


def tokenize(documents):
  '''tokenizes documents
  Args:
    documents: iterable of strings
  Returns:
    list of list of strings
  '''

  return [list(split_on_punctuation(doc)) for doc in documents]


def feature_counts(documents):
  '''computes feature counts from featurized documents
  Args:
    documents: iterable of lists of hashable features
  Returns:
    dict mapping features to counts
  '''

  return Counter(feat for doc in documents for feat in doc)


def feature_vocab(documents, min_count=1, sorted_features=sorted):
  '''gets feature vocabulary from featurized documents
  Args:
    documents: iterable of lists of hashable features
    min_count: minimum number of times feature must appear to be included in the vocabulary
    sorted_features: function that sorts the features
  Returns:
    {feature: index} dict
  '''
  
  return {feat: i for i, feat in enumerate(sorted_features(feat for feat, count in feature_counts(documents).items() if count >= min_count))}


def docs2bofs(documents, vocabulary=None, weights=None, default=1.0, format='csr', **kwargs):
  '''constructs sparse BoF representations from featurized documents
  Args:
    documents: iterable of lists of hashable features
    vocabulary: dict mapping features to indices (nonnegative ints) or a list of features; if None will compute automatically from documents
    weights: dict mapping features to weights (floats) or a list/np.ndarray of weights; if None will compute unweighted BoFs
    default: default feature weight if not feature in weights; ignored if weights is None
    format: sparse matrix format
    kwargs: passed to feature_vocab; ignored if not vocabulary is None
  Returns:
    sparse BoF matrix in CSR format of size (len(documents), len(vocabulary))
  '''

  if vocabulary is None:
    vocabulary = feature_vocab(documents, **kwargs)
  elif type(vocabulary) == list:
    vocabulary = {feat: i for i, feat in enumerate(vocabulary)}

  rows, cols, values = zip(*((row, col, count) for (row, col), count in Counter((i, vocabulary.get(feat, -1)) for i, doc in enumerate(documents) for feat in doc).items() if not col==-1))
  m = len(documents)
  V = len(vocabulary)
  if weights is None:
    return sp.coo_matrix((values, (rows, cols)), shape=(m, V), dtype=UINT).asformat(format)
  bofs = sp.coo_matrix((values, (rows, cols)), shape=(m, V)).tocsr()

  if type(weights) == dict:
    diag = np.empty(V)
    for feat, i in vocabulary.items():
      diag[i] = weights.gets(feat, default)
  else:
    assert len(weights) == V, "if weights passed as a list/np.ndarray, length must be same as vocabulary size"
    if type(weights) == list:
      diag = np.array(weights)
    else:
      diag = weights
  return bofs.dot(sp.diags(diag, 0)).asformat(format)


def sif_weights(documents_or_counts, a=1E-2):
  '''computes SIF weights from featurized documents
  Args:
    documents_or_counts: iterable of lists of hashable features or dict mapping features to counts or count vector
    a: SIF parameter
  Returns:
    if passed documents of count dict: dict mapping features to weights (floats); else a weight vector
  '''

  if type(documents_or_counts) == np.ndarray:
    axtotal = a*sum(documents_or_counts)
    return axtotal/(axtotal+documents_or_counts) 
  if type(documents_or_counts) == list:
    documents_or_counts = feature_counts(documents_or_counts)
  axtotal = a*sum(documents_or_counts.values())
  return {feat: axtotal/(axtotal+count) for feat, count in documents_or_counts.items()}
