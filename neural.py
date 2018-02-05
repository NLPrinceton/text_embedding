import importlib
import sys
from text_embedding.documents import *


def byte_mLSTM():
  assert PYTHONVERSION == '3', "Python 3 must be used for byte_mLSTM"
  Model = importlib.import_module('model-byte_mLSTM.encoder').Model
  return Model().transform, None, True


def skipthoughts(direction='concat'):
  assert PYTHONVERSION == '2', "Python 2 must be used for skipthoughts"
  EncoderManager = importlib.import_module('model-skipthoughts.encoder_manager').EncoderManager
  encoder = EncoderManager()
  if direction in {'uni', 'concat'}:
    encoder.load_uni()
  if direction in {'bi', 'concat'}:
    encoder.load_bi()
  return lambda documents: np.nan_to_num(encoder.encode(documents, use_eos=True)), None, True


def Sent2Vec(order='uni'):
  assert PYTHONVERSION == '3', "Python 3 must be used for Sent2Vec"
  get_sentence_embeddings = importlib.import_module('model-Sent2Vec.build').get_sentence_embeddings
  return lambda documents: get_sentence_embeddings(documents, ngram=order), None, True


if __name__ == '__main__':

  model = globals()[sys.argv[2]]
  try:
    represent, prepare, invariant = model(sys.argv[3])
  except IndexError:
    represent, prepare, invariant = model()
  for task in sys.argv[1].split(','):
    evaluate(task, represent, prepare=prepare, invariant=invariant, batchsize=200, verbose=True, intercept=task in TASKMAP['pairwise task'])
