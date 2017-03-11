from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
import math
from time import time

IMAGE_SIZE = 28
NUM_OF_CLASSES = 10
BATCH_SIZE = 100

TRAIN_SIZE = 280000    # 280255
CV_SIZE = 69000        # 69745
TEST_SIZE = 18000      # 18000

def reformat(dataset, labels, setsize):

    dataset = np.reshape(dataset,(-1,IMAGE_SIZE*IMAGE_SIZE)).astype(np.float32)
    labels = (np.arange(NUM_OF_CLASSES) == labels[:,None]).astype(np.float32) # one hot encoding

    return dataset[:setsize, ], labels[:setsize,]

try:
    print(" ### Getting Data from pickle ### \n\n")
    with open("notMNIST.pickle", 'rb') as f:
        data_map = pickle.load(f)

        train_dataset = data_map['train_dataset']
        train_label = data_map['train_label']
        print("current TRAIN shapes : ", train_dataset.shape, train_label.shape)
        train_dataset, train_labels = reformat(dataset=train_dataset, labels=train_label, setsize=TRAIN_SIZE)
        print("NEW TRAIN shapes = ", train_dataset.shape , train_labels.shape)

        cv_dataset = data_map['cv_dataset']
        cv_labels = data_map['cv_label']
        print("current TRAIN shapes : ", cv_dataset.shape, cv_labels.shape)
        cv_dataset, cv_labels = reformat(dataset=cv_dataset, labels=cv_labels, setsize=CV_SIZE)
        print("NEW CV shapes = ", cv_dataset.shape , cv_labels.shape)

        test_dataset = data_map['test_dataset']
        test_labels = data_map['test_label']
        print("current test shapes : ", test_dataset.shape, test_dataset.shape)
        test_dataset, test_labels = reformat(dataset=test_dataset, labels=test_labels, setsize=TEST_SIZE)
        print("NEW TEST shapes = ", test_dataset.shape , test_labels.shape)


except Exception as e:
    print("Sorry couldn't read file : ", e)

# ################################################################################################

'''
Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

'''

# Basic model parameters which can be tuned here as flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, "this is the learning rate")
flags.DEFINE_integer('hidden1', 1024, "number of nodes in hidden layer 1")
flags.DEFINE_integer('batch_size', 100, "input batch size for input layer of the nn ... must be div by m")
flags.DEFINE_integer('MAX_STEPS', 2800, "max step for the gradient descent")
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')

def inference(images_placeholder, hidden_layer1):

    """
    Build the MNIST model up to where it may be used for inference.

    Args:
    images_placeholder: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.

    Returns:
    softmax_linear: Output tensor with the computed logits.

    """

    # HIDDEN 1
    # weights shape --> (784,1024) ;  biases shape --> (1024,)
    # hidden 1 will be (100,1024) relu outputs
    with tf.name_scope("hidden1"):
        with tf.name_scope("weights_1"):
            weights_1 = tf.Variable(tf.truncated_normal(shape =[IMAGE_SIZE*IMAGE_SIZE, hidden_layer1],
                                                      stddev=1.0/ math.sqrt(float(IMAGE_SIZE*IMAGE_SIZE)),
                                                      name="weights_1"
                                                      ))
            tf.histogram_summary("[1->2]Weights ", weights_1)

            biases_1 = tf.Variable(tf.zeros([hidden_layer1]), name="biases_1")



        # activation output of layer 1 : 100*784  X 784*1024 == 100*1024
        try:
            hidden1 = tf.nn.relu(tf.matmul(images_placeholder, weights_1)+biases_1)
        except Exception as e:
            print("matmul failed check matrix mul compatability :", e)


    # LINEAR-Softmax output layer
    # weights shape --> (1024,10) ;  biases shape --> (10,)
    # hidden logits will be (100,1024) relu outputs
    with tf.name_scope("softmax_linear"):
        with tf.name_scope("weights_2"):
            weights_2 = tf.Variable(tf.truncated_normal([hidden_layer1, NUM_OF_CLASSES],
                                                      stddev=1.0/math.sqrt(float(hidden_layer1)),
                                                      name="weights_2"
                                                      ))
            tf.histogram_summary("[2->3]weights ", weights_2)

            biases_2 = tf.Variable(tf.zeros([NUM_OF_CLASSES]), name="biases_2")

        # so we return logits and then compute soft-max of this logits tensor
        logits = tf.matmul(hidden1, weights_2) + biases_2

    return logits

def lossFun(logits, labels_placeholder):

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_placeholder,
                                                                   name='xentropy')
    # since labels_placeholder stores the size of the current batch ie value of m to sum over to compute
    # reduced mean
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.scalar_summary("loss value", loss)

    return loss

def training(learning_rate, loss):

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss)

    return train_op

def evaluation(logits, labels):

  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  # here correct is of shape 100 for the current batch of images and for each image we have 10 contenders
  # ie 10 classes and we choose the top 1
  print("evaluation .. start")
  print(logits)
  print(labels)
  correct = tf.nn.in_top_k(predictions= logits, targets= labels, k =  1)


  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def do_eval(session, eval_correct, images_placeholder, labels_placeholder, dataset, labels, setsize):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval. where an epoch is defined as TestSize/BatchSize therefore we have 2800 steps in 1 epoch

  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = int(setsize/BATCH_SIZE)

  print("steps per epoch: ", steps_per_epoch)
  num_of_examples = steps_per_epoch*FLAGS.batch_size
  for step in range(steps_per_epoch):
    start_time = time()
    '''
    create an offset for the batch and then fill the feed dictionary with the actual
    set  of images and labels for this particular batch of training step
    '''
    offset = (step*BATCH_SIZE)%(labels.shape[0]-BATCH_SIZE)
    # generate the batch
    batch_images_data = dataset[offset:(offset+BATCH_SIZE), :]
    batch_labels = labels[offset:(offset+BATCH_SIZE):]

    feed_dict = {images_placeholder: batch_images_data, labels_placeholder: batch_labels}
    true_count += session.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_of_examples

    print('  Num examples: %d  Num correct: %d  Accuracy @ 1: %0.04f' %(num_of_examples, true_count, precision))


def accuracy(predictions, labels):

    '''
    :param predictions: shape is 10*100 where 100 is the batch size and 10 is the predicted prob list for a given input image
                        we use argmax to pick out the max probability for all the 100 lists, argmax returns the index of max value[predicted label value]
    :param labels: know values for that batch
    :return: , compare them to the predictions and return the batch accuracy %

    '''

    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))  / predictions.shape[0])


def run_training():


    with tf.Graph().as_default():
        # generate place holders for the images and labels, we'll fill the placeholder later as usual
        # batch size images at a time each of size 784 (28*28)
        train_images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE))
        train_labels_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_OF_CLASSES))

        cv_images_placeholder = tf.constant(cv_dataset)
        test_images_placeholder = tf.constant(test_dataset)

        '''
        inference is where we create the structure of the desired NN , ie weights & biases for each layer,
        number of units , relu , softmax everything and now the nn is defined.
        '''

        # HIDDEN 1
        # weights shape --> (784,1024) ;  biases shape --> (1024,)
        # hidden 1 will be (100,1024) relu outputs
        with tf.name_scope("hidden1"):
            with tf.name_scope("weights_1"):
                weights_1 = tf.Variable(tf.truncated_normal(shape=[IMAGE_SIZE * IMAGE_SIZE, FLAGS.hidden1],
                                                            stddev=1.0 / math.sqrt(float(IMAGE_SIZE * IMAGE_SIZE)),
                                                            name="weights_1"
                                                            ))
                tf.histogram_summary("[1->2]Weights ", weights_1)
            with tf.name_scope("biases_1"):
                biases_1 = tf.Variable(tf.zeros([FLAGS.hidden1]), name="biases_1")

            # activation output of layer 1 : 100*784  X 784*1024 == 100*1024
            try:
                hidden1 = tf.nn.relu(tf.matmul(train_images_placeholder, weights_1) + biases_1)
            except Exception as e:
                print("matmul failed check matrix mul compatability :", e)

        # LINEAR-Softmax output layer
        # weights shape --> (1024,10) ;  biases shape --> (10,)
        # hidden logits will be (100,1024) relu outputs
        with tf.name_scope("softmax_linear"):
            with tf.name_scope("weights_2"):
                weights_2 = tf.Variable(tf.truncated_normal([FLAGS.hidden1, NUM_OF_CLASSES],
                                                            stddev=1.0 / math.sqrt(float(FLAGS.hidden1)),
                                                            name="weights_2"
                                                            ))

                tf.histogram_summary("[2->3]weights ", weights_2)

            with tf.name_scope("biases_2"):
                biases_2 = tf.Variable(tf.zeros([NUM_OF_CLASSES]), name="biases_2")

            # so we return logits and then compute soft-max of this logits tensor
            logits = tf.matmul(hidden1, weights_2) + 0.01*tf.nn.l2_loss(biases_2)


        #logits = inference(train_images_placeholder, FLAGS.hidden1)

        loss = lossFun(logits, train_labels_placeholder) # Cost(h(x),y) = -{ ylog(h(x)) - (1-y)(1-log(h(x)) }


        train_op = training(FLAGS.learning_rate, loss)

        train_pred = tf.nn.softmax(logits= logits)

        cv_pred = tf.nn.softmax(
            tf.matmul(tf.nn.relu((tf.matmul(cv_images_placeholder, weights_1)+ biases_1)), weights_2) + biases_2
        )

        test_pred = tf.nn.softmax(
            tf.matmul(tf.nn.relu((tf.matmul(test_images_placeholder, weights_1) + biases_1)), weights_2) + biases_2
        )

        #eval_correct = evaluation(logits, labels_placeholder)

        merged = tf.merge_all_summaries()
        init = tf.initialize_all_variables()

        session = tf.InteractiveSession()


        session.run(init)

        #tf.image_summary("myimage", [100, 28, 28, 1])

        summary_writer = tf.train.SummaryWriter("/home/sarthak/Mydata/Projects/UdacityDeepLearning/TD_GD_logs/",
                                                session.graph)

        # starting the training loop
        for step in range(FLAGS.MAX_STEPS):
            start_time = time()
            '''
            create an offset for the batch and then fill the feed dictionary with the actual
            set  of images and labels for this particular batch of training step
            '''
            offset = (step*BATCH_SIZE)%(train_labels.shape[0]-BATCH_SIZE)
            # generate the batch
            batch_images_data = train_dataset[offset:(offset+BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset+BATCH_SIZE):]

            feed_dict = {train_images_placeholder: batch_images_data, train_labels_placeholder: batch_labels}

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value, batch_predictions, summary = session.run([train_op, loss,train_pred, merged], feed_dict=feed_dict)

            summary_writer.add_summary(summary, step)
            duration = time() - start_time

            if step%100 == 0:
                print("loss at step %d : %f"%(step, loss_value))
                print("Batch accuracy @ step", step, ": ",  accuracy(batch_predictions, batch_labels))


        print("Evaluating corss validation data\n")
        print(accuracy(cv_pred.eval(), cv_labels))

        print("Evaluating Test data\n")
        print(accuracy(test_pred.eval(), test_labels))

        final_weights = weights_2.eval()


        print(final_weights[0], final_weights.shape)


def main():
    print("........ not mnist traning via 1 Layer NN using relu ......\n")
    run_training()

main()

