import argparse
import sys
from collections import Counter
import numpy as np
from text_embedding.documents import *
from text_embedding.features import *
from text_embedding.vectors import *


def unigram_baseline(w2v, task, n_jobs=-1):
  z = np.zeros(w2v[next(iter(w2v))].shape[0])
  rep = lambda docs: np.vstack(sum((w2v.get(w, z) for w in split_on_punctuation(doc.lower())), z) for doc in docs)
  return evaluate(task.lower(), rep, invariant=True, params=[10**i for i in range(-4, 5)], n_jobs=n_jobs)


@align_vocab
def linear_alignment(source, target, orthogonal=True, fit_intercept=False):
  M, b = best_transform(source, target, orthogonal=orthogonal, fit_intercept=fit_intercept)
  return average_cosine_similarity(source.dot(M.T)+b, target)


def parse():
  parser = argparse.ArgumentParser(prog='python text_embedding/testvecs.py')
  parser.add_argument('vectorfiles', nargs='+', help='one or two word embedding text files (space-separated)')
  parser.add_argument('-d', '--dimension', default=None, help='embedding dimension (defaults to using entire row)', type=int)
  parser.add_argument('-t', '--tasks', nargs='*', help='embedding evaluation tasks (space-separated)')
  return parser.parse_args()


if __name__ == '__main__':

  args = parse()
  files = args.vectorfiles
  d = args.dimension

  if len(files) == 1:
    write('Loading Word Embeddings')
    w2v = vocab2vecs(vectorfile=files[0], dimension=d)
    tasks = args.tasks if args.tasks else ['SST', 'IMDB']

    write('\rClassification Evaluation: Test Accuracy using Logit over Sum-of-Embeddings\n')
    for task in tasks:
      write(task+' Acc: ')
      write(str(unigram_baseline(w2v, task.lower())[1]) + '\n')

  else:
    write('Loading Source Embeddings')
    src = vocab2vecs(vectorfile=files[0], dimension=d, unit=False)
    write('\rLoading Target Embeddings')
    tgt = vocab2vecs(vectorfile=files[1], dimension=d, unit=False)

    write('\rAlignment Evaluation: Mean Cosine Similarity of Best Orthogonal Transform\n')
    write('Avg Sim: ')
    write(str(linear_alignment(src, tgt, orthogonal=True, fit_intercept=False)) + '\n')

    write('\rAlignment Evaluation: Mean Cosine Similarity of Best Orthogonal Transform with Translation\n')
    write('Avg Sim: ')
    write(str(linear_alignment(src, tgt, orthogonal=True, fit_intercept=True)) + '\n')

    write('\rAlignment Evaluation: Mean Cosine Similarity of Best Linear Transform\n')
    write('Avg Sim: ')
    write(str(linear_alignment(src, tgt, orthogonal=False, fit_intercept=False)) + '\n')

    write('\rAlignment Evaluation: Mean Cosine Similarity of Best Linear Transform with Translation\n')
    write('Avg Sim: ')
    write(str(linear_alignment(src, tgt, orthogonal=False, fit_intercept=True)) + '\n')
