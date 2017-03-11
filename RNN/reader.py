# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("UTF-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)                   # data is simply reading all the train data and tokenizing it , note its ordered

  counter = collections.Counter(data)            # returns a dict of "word":word count
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))  # items() converts to a list of (element, count) pairs
  words, _ = list(zip(*count_pairs))    # zip(*data) where data is a list of tuples[(x1, y1), (x2,y2)] gives two separate lists of [x1,x2] and [y1,y2]

  word_to_id = dict(zip(words, range(len(words))))  # so every word in the corpus has an id now

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_producer(raw_data, batch_size, num_steps, name=None):


  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  print("PTB Producer")
  print(len(raw_data))
  with tf.name_scope("PTBProducer"):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)   # now the raw data is simply a tensor of 19,29,589

    data_len = tf.size(raw_data)
    batch_len = data_len // batch_size    # basically the number of batches that can be formed given the data size of 19, 29, 589 and batch_size =20


    data = tf.reshape(tensor=raw_data[0 : batch_size * batch_len], shape =  [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps
    assertion = tf.assert_positive(epoch_size,message="epoch_size == 0, decrease batch_size or num_steps")

    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps], tf.ones_like([0, i * num_steps]))
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1], tf.ones_like([0, i * num_steps]))
    y.set_shape([batch_size, num_steps])
    return x, y



def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """fidsfjfddskfjlsdjfldjflsdjflksdjflkdsjflkdsjflkdsjf

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)    # a dictionary of 10, 000 words, which is all the unique words in the text corpus
  train_data = _file_to_word_ids(train_path, word_to_id)  # having indexed each word by its id in the word_to_id dict we replace the train data text with the id's of the words
  valid_data = _file_to_word_ids(valid_path, word_to_id)   # same take the validation data, and see what all words in the validation data are present in the dicitonary, make a list of ids of those words
  test_data = _file_to_word_ids(test_path, word_to_id)

  vocabulary = len(word_to_id)
  print(len(train_data), vocabulary)
  return train_data, valid_data, test_data, vocabulary


