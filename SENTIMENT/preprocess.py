from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from six.moves import urllib

import sys

import tensorflow as tf
from tensorflow.python.platform import gfile


import pandas as pd
import os
import numpy as np

def basic_tokenizer(sentence):
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]



def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 1000 == 0:
          print("  processing line %d" % counter)
        tokens = basic_tokenizer(line)

        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w)
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1

      print('>> Dataset number of samples :', counter)
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      print('>> Full Vocabulary Size :',len(vocab_list))
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


#private
def initialize_vocabulary(vocabulary_path):

  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


#private
def sentence_to_token_ids(sentence, vocabulary):
  words = basic_tokenizer(sentence)
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]




def data_to_token_ids(data_path, target_path, vocabulary_path):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 1000 == 0:
            print("    Tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
 





_PAD = b"_PAD"
_EOS = b"_EOS"
_UNK = b"_UNK"
_GO = b"_GO"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


_WORD_SPLIT = re.compile(b'([.,!?"\':;)(])')
_DIGIT_RE = re.compile(br'\d')


working_dir = '/Users/smokey/COGS118A/SENTIMENT/'
input_file = 'data.csv'
vocab_file = 'vocab.csv'
train_file = 'train.csv'

input_path = os.path.join(working_dir, input_file)
train_path = os.path.join(working_dir, train_file)
vocab_path = os.path.join(working_dir, vocab_file)


create_vocabulary(vocab_path, input_path, 10000000)
data_to_token_ids(input_path, train_path, vocab_path)
