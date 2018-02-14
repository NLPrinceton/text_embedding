import sys
import nltk
from scipy import sparse as sp
from text_embedding.documents import *
from text_embedding.features import *
from text_embedding.vectors import *


# NOTE: filepath for Amazon GloVe embeddings goes here
VECTORFILES[('Amazon', 'GloVe', 1600)] = '/n/fs/nlpdatasets/AmazonProductData/amazon_glove1600.txt'


def BonG(n, min_count=1):
  prepare = lambda documents: ([True],)
  def represent(documents, vocab):
    docs = [tokenize(doc.lower() for doc in documents)]
    for k in range(1, n+1):
      docs.append([list(nltk.ngrams(doc, k)) for doc in docs[0]])
      if vocab[0]:
        vocab.append(sorted({gram for gram, count in feature_counts(docs[k]).items() if count >= min_count}))
    vocab[0] = False
    return sp.hstack([docs2bofs(docs[k], vocabulary=vocab[k]) for k in range(1, n+1)], format='csr') 
  return represent, prepare, True


def SIF(a, vectorfile=None, corpus='Amazon', objective='GloVe', dimension=1600):
  prepare = lambda documents: (vocab2vecs({word for doc in documents for word in split_on_punctuation(doc.lower())}, vectorfile=vectorfile, corpus=corpus, objective=objective, dimension=dimension), [True, None])
  def represent(documents, w2v, weights):
    docs = tokenize(doc.lower() for doc in documents)
    if weights[0]:
      weights[0] = False
      weights[1] = sif_weights(docs, a)
    else:
      weights[0] = True
    return docs2vecs(docs, f2v=w2v, weights=weights[1])
  return represent, prepare, False


if __name__ == '__main__':

  try:
    represent, prepare, invariant = BonG(int(sys.argv[2]))
  except ValueError:
    represent, prepare, invariant = SIF(float(sys.argv[2]))
  for task in sys.argv[1].split(','):
    evaluate(task, represent, prepare=prepare, invariant=invariant, verbose=True, intercept=task in TASKMAP['pairwise task'])
