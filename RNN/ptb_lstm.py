
"""

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader


flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "/home/sarthak/Mydata/Projects/UdacityDeepLearning/PTB/simple-examples/data/",
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", '/home/sarthak/PycharmProjects/udacity-deep learning/RNN/',
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000

class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35        # the number of unrolled time steps of LSTM
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def get_config():
  if(FLAGS.model=="small"):
    return SmallConfig()
  if (FLAGS.model == "medium"):
    return SmallConfig()
  if (FLAGS.model == "large"):
    return SmallConfig()
  if(FLAGS.model=="test"):
    return TestConfig()

class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps

    self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)


class PTBModel(object):

  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    def lstm_cell():
      return tf.nn.rnn_cell.BasicLSTMCell(num_units=size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    # since dropout is only implemented during training and keep prob <1  [keep prob =1 effectively means, drop not used]
    if(is_training and config.keep_prob < 1):
      def attn_cell():
        return tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell(), output_keep_prob=config.keep_prob)

    cell = tf.nn.rnn_cell.MultiRNNCell(cells= [attn_cell() for _ in range(config.num_layers)] , state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size=batch_size, dtype=data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(name="embedding", shape=[vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if(is_training and config.keep_prob<1):
      inputs = tf.nn.dropout(x=inputs, keep_prob=config.keep_prob)

    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0:
          tf.get_variable_scope().reuse_variables()  # indicate to reuse all the varialbes under the scope model/RNN ie BasicLSTMCell, MultiRNNCell ..

        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

      # Concatenates the list of tensors `values` along dimension `concat_dim`.

    output = tf.reshape(tf.concat(concat_dim=1, values=outputs), [-1, size])

    softmax_w = tf.get_variable(name="softmax_w", shape=[size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable(name="softmax_b", shape=[vocab_size], dtype=data_type())

    logits = tf.matmul(output, softmax_w) + softmax_b

    loss = tf.nn.seq2seq.sequence_loss_by_example(logits=[logits], targets= [tf.reshape(input_.targets, [-1]) ] ,
                                                    weights= [tf.ones(shape=[batch_size*num_steps],dtype=data_type() )] )

    self._cost = cost = tf.reduce_sum(loss)/batch_size
    self._final_state = state

    if not is_training:
      return
    # note all the above part of the PTBModel were being reused by test and valid models to compute logits, but from now on nothing is needed
    # so if not training , simply return else continue with your usual Gradient Descent using clipping gradient norms by a certain threshold to
    # prevent the problem of vanishing or exploding gradients in RNN

    '''
    trainable: If `True`, the default, also adds the variable to the graph collection `GraphKeys.TRAINABLE_VARIABLES`. This collection is used as
    the default list of variables to use by the `Optimizer` classes.
    '''
    self._lr = tf.Variable(initial_value=0.0, trainable=False)  # learning rate
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=self._lr, name="GradientDescent")
    global_step = tf.Variable(0, name='global_step', trainable=False)
    self._train_op = optimizer.apply_gradients(grads_and_vars=zip(grads, tvars), global_step=global_step)

    self._new_lr = tf.placeholder(
      tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)


  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

###############################  PTB model ends ##########################

def run_epoch(session, model, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)



def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data   # what we get here is normal text corpus, where each word is replaced by its ID


  config = get_config()

  # test configuration
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps =1


  with tf.Graph().as_default():
    initializer =  tf.random_uniform_initializer(-config.init_scale, config.init_scale)

    with tf.name_scope("Train"):

      '''
      train_input is an object of the class type PTBInput
      By giving the train_data and configuration [small in this case] we get train_input
      train_input has : input_data [X's] and labels/targets [Y's]

      '''
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")

      '''
      configuration provides batch_size, epochs, num_of_steps and we also pass the train_input to now construct our model for training
      '''
      with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.scalar_summary("training loss", m.cost)
      tf.scalar_summary("Learning Rate", m.lr)


    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("model", reuse=True, initializer=initializer):
          mvalid = PTBModel(is_training=False, config=config, input_= valid_input)
      tf.scalar_summary("Validation Loss",mvalid.cost )

    with tf.name_scope("Test"):
      test_input = PTBInput(config=config, data=test_data, name="TestInput")
      with tf.variable_scope("model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)

    # running the graph on a session
    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i+1 - config.max_epoch, 0.0)  # decaying LR based on the current epoch
        m.assign_lr(session=session, lr_value= config.learning_rate*lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, eval_op=m.train_op,verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

        valid_perplexity = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

      test_perplexity = run_epoch(session=session, model=mtest)
      print("Valid Perplexity: %.3f" % valid_perplexity)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

if __name__ == "__main__":
  tf.app.run()


'''
https://www.tensorflow.org/versions/master/how_tos/variable_scope/
when we create a variable in TF using tf.variable() and pass a tensor into it of a certain shape and a stdev to draw the initial values
from some distribution.
and all such variables are actually initialized with values using tf.global_variables_initializer()

############################
Also, Variable Scope mechanism in TensorFlow consists of two main functions:

tf.get_variable(<name>, <shape>, <initializer>): Creates or returns a variable with a given name.
tf.variable_scope(<scope_name>): Manages namespaces for names passed to tf.get_variable().

But if we want to share these variables you might want to initialize them all in one place using tf.variable_scope()
The function tf.get_variable() is used to get or create a variable instead of a direct call to tf.Variable().
It uses an initializer instead of passing the value directly, as in tf.Variable. An initializer is a function
that takes the shape and provides a tensor with that shape.

Note: The reuse parameter is inherited. So when you open a reusing variable scope, all sub-scopes will be reusing too.
##############################
 Reshape Example

 # tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]]

so this tensor has 3 matrices each of dim 2X3
Matrix 1  [
           [1,1,1],
           [2,2,2]
          ]

# -1 can also be used to infer the shape

# -1 is inferred to be 9:
reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 2:
reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                         [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# -1 is inferred to be 3:
reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]],
                             [[4, 4, 4],
                              [5, 5, 5],
                              [6, 6, 6]]]

lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=False)
stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm] * number_of_layers,
    state_is_tuple=False)

'''