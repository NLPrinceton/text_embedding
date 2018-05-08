import os
import time
import sys
import re
from subprocess import call
import numpy as np
from nltk import TweetTokenizer
from nltk.tokenize import StanfordTokenizer


# NOTE: Set necessary paths here
NLPDATA = '/n/fs/nlpdatasets/'
FASTTEXT_EXEC_PATH = NLPDATA+'fastText/fasttext'
MODEL_TORONTOBOOKS_UNIGRAMS = NLPDATA+'sent2vec/torontobooks_unigrams.bin'
MODEL_TORONTOBOOKS_BIGRAMS = NLPDATA+'sent2vec/torontobooks_bigrams.bin'
SNLP_TAGGER_JAR = NLPDATA+'stanford-corenlp-full-2016-10-31/stanford-postagger-2017-06-09/stanford-postagger-3.8.0.jar'


def tokenize(tknzr, sentence, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentence: a string to be tokenized
        - to_lower: lowercasing or not
    """
    sentence = sentence.strip()
    sentence = ' '.join([format_token(x) for x in tknzr.tokenize(sentence)])
    if to_lower:
        sentence = sentence.lower()
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
    sentence = re.sub('(\@[^\s]+)','<user>',sentence) #replace @user268 by <user>
    filter(lambda word: ' ' not in word, sentence)
    return sentence

def format_token(token):
    """"""
    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token

def tokenize_sentences(tknzr, sentences, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentences: a list of sentences
        - to_lower: lowercasing or not
    """
    return [tokenize(tknzr, s, to_lower) for s in sentences]


def get_embeddings_for_preprocessed_sentences(sentences, model_path, fasttext_exec_path):
    """Arguments:
        - sentences: a list of preprocessed sentences
        - model_path: a path to the sent2vec .bin model
        - fasttext_exec_path: a path to the fasttext executable
    """
    timestamp = str(time.time())
    test_path = NLPDATA+'sent2vec/'+timestamp+'_fasttext.test.txt'
    embeddings_path = NLPDATA+'sent2vec/'+timestamp+'_fasttext.embeddings.txt'
    dump_text_to_disk(test_path, sentences)
    call(fasttext_exec_path+
          ' print-sentence-vectors '+
          model_path + ' < '+
          test_path + ' > ' +
          embeddings_path, shell=True)
    embeddings = read_embeddings(embeddings_path)
    os.remove(test_path)
    os.remove(embeddings_path)
    assert(len(sentences) == len(embeddings))
    return np.array(embeddings)

def read_embeddings(embeddings_path):
    """Arguments:
        - embeddings_path: path to the embeddings
    """
    with open(embeddings_path, 'r') as in_stream:
        embeddings = []
        for line in in_stream:
            line = '['+line.replace(' ',',')+']'
            embeddings.append([float(entry) for entry in line.split(',')[-701:-1]])
            #embeddings.append(eval(line))
        return embeddings
    return []

def dump_text_to_disk(file_path, X, Y=None):
    """Arguments:
        - file_path: where to dump the data
        - X: list of sentences to dump
        - Y: labels, if any
    """
    with open(file_path, 'w') as out_stream:
        if Y is not None:
            for x, y in zip(X, Y):
                out_stream.write('__label__'+str(y)+' '+x+' \n')
        else:
            for x in X:
                out_stream.write(x+' \n')

def get_sentence_embeddings(sentences, ngram='uni'):
    tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
    s = ' <delimiter> '.join(sentences) #just a trick to make things faster
    tokenized_sentences_SNLP = tokenize_sentences(tknzr, [s])
    tokenized_sentences_SNLP = tokenized_sentences_SNLP[0].split(' <delimiter> ')
    if len(tokenized_sentences_SNLP) != len(sentences):
      print('SENT2VEC TOKENIZATION FAILED')
      tokenized_sentences_SNLP = sentences
    #assert(len(tokenized_sentences_SNLP) == len(sentences))
    if ngram == 'uni':
        embeddings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_SNLP, \
                                 MODEL_TORONTOBOOKS_UNIGRAMS, FASTTEXT_EXEC_PATH)
    elif ngram == 'bi':
        embeddings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_SNLP, \
                                 MODEL_TORONTOBOOKS_BIGRAMS, FASTTEXT_EXEC_PATH)
    else:
        raise(NotImplementedError)
    return embeddings


def _get_sentence_embeddings(sentences, ngram='bigrams', model='concat_wiki_twitter'):
    """ Returns a numpy matrix of embeddings for one of the published models. It
    handles tokenization and can be given raw sentences.
    Arguments:
        - ngram: 'unigrams' or 'bigrams'
        - model: 'wiki', 'twitter', or 'concat_wiki_twitter'
        - sentences: a list of raw sentences ['Once upon a time', 'This is another sentence.', ...]
    """
    wiki_embeddings = None
    twitter_embbedings = None
    tokenized_sentences_NLTK_tweets = None

      
    tokenized_sentences_SNLP = None
    if model == "wiki" or model == 'concat_wiki_twitter':
        tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
        s = ' <delimiter> '.join(sentences) #just a trick to make things faster
        tokenized_sentences_SNLP = tokenize_sentences(tknzr, [s])
        tokenized_sentences_SNLP = tokenized_sentences_SNLP[0].split(' <delimiter> ')
        assert(len(tokenized_sentences_SNLP) == len(sentences))
        if ngram == 'unigrams':
            wiki_embeddings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_SNLP, \
                                     MODEL_WIKI_UNIGRAMS, FASTTEXT_EXEC_PATH)
        else:
            wiki_embeddings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_SNLP, \
                                     MODEL_WIKI_BIGRAMS, FASTTEXT_EXEC_PATH)
    if model == "twitter" or model == 'concat_wiki_twitter':
        tknzr = TweetTokenizer()
        tokenized_sentences_NLTK_tweets = tokenize_sentences(tknzr, sentences)
        if ngram == 'unigrams':
            twitter_embbedings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_NLTK_tweets, \
                                     MODEL_TWITTER_UNIGRAMS, FASTTEXT_EXEC_PATH)
        else:
            twitter_embbedings = get_embeddings_for_preprocessed_sentences(tokenized_sentences_NLTK_tweets, \
                                     MODEL_TWITTER_BIGRAMS, FASTTEXT_EXEC_PATH)
    if model == "twitter":
        return twitter_embbedings
    elif model == "wiki":
        return wiki_embeddings
    elif model == "concat_wiki_twitter":
        return np.concatenate((wiki_embeddings, twitter_embbedings), axis=1)
    sys.exit(-1)
