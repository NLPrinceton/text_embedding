import argparse
import sys
from collections import Counter
import numpy as np
from text_embedding.documents import *
from text_embedding.features import *
from text_embedding.vectors import *


def unigram_baseline(w2v, task, n_jobs=-1, sif=False):
  z = np.zeros(w2v[next(iter(w2v))].shape[0])
  if sif:
    def rep(docs):
      docs = [[w for w in split_on_punctuation(doc.lower())] for doc in docs]
      wei = sif_weights(Counter(w for doc in docs for w in doc))
      return np.vstack(sum((wei.get(w, 0.0)*w2v.get(w, z) for w in docs), z) for doc in docs)
  else:
    rep = lambda docs: np.vstack(sum((w2v.get(w, z) for w in split_on_punctuation(doc.lower())), z) for doc in docs)
  return evaluate(task.lower(), rep, invariant=True, params=[10**i for i in range(-4, 5)], n_jobs=n_jobs)


@align_vocab
def linear_alignment(source, target, orthogonal=True):
  transform = best_transform(source, target, orthogonal=orthogonal)
  return average_cosine_similarity(source.dot(transform.T), target)


def parse():
  parser = argparse.ArgumentParser(prog='python text_embedding/testvecs.py')
  parser.add_argument('vectorfiles', nargs='+', help='one or two word embedding text files (space-separated)', type=str)
  parser.add_argument('-d', '--dimension', default=None, help='embedding dimension (defaults to using entire row)', type=int)
  return parser.parse_args()


if __name__ == '__main__':

  args = parse()
  files = args.vectorfiles
  d = args.dimension

  if len(files) == 1:
    write('Loading Word Embeddings')
    w2v = vocab2vecs(vectorfile=files[0], dimension=d)

    write('\rClassification Evaluation: Test Accuracy using Logit over Sum-of-Embeddings\n')
    for task in ['SST', 'IMDB']:
      write(task+' Acc: ')
      write(str(document_classification(w2v, task.lower())[1]) + '\n')

  else:
    write('Loading Source Embeddings')
    src = vocab2vecs(vectorfile=files[0], dimension=d, unit=False)
    write('\rLoading Target Embeddings')
    tgt = vocab2vecs(vectorfile=files[1], dimension=d, unit=False)

    write('\rAlignment Evaluation: Mean Cosine Similarity of Best Orthogonal Transform\n')
    write('Avg Sim: ')
    write(str(linear_alignment(src, tgt, orthogonal=True)) + '\n')
