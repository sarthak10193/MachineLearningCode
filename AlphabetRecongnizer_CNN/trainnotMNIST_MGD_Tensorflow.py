import tensorflow as tf
import numpy as np
from time import time
from six.moves import cPickle as pickle

IMAGE_SIZE = 28
NUM_OF_LABELS = 10
BATCH_SIZE=100

def reformat(dataset,labels):
    dataset = np.reshape(dataset,(-1,IMAGE_SIZE*IMAGE_SIZE)).astype(np.float32)
    labels = (np.arange(NUM_OF_LABELS) == labels[:,None]).astype(np.float32) # one hot encoding

    return dataset, labels

try:
    with open("notMNIST.pickle", 'rb') as f:
        data_map = pickle.load(f)
        train_dataset = data_map['train_dataset']
        train_label = data_map['train_label']
        print("current TRAIN shapes : ", train_dataset.shape, train_label.shape)
        train_dataset, train_label = reformat(train_dataset,train_label)
        print("NEW TRAIN shapes = ", train_dataset.shape , train_label.shape)

        cv_dataset = data_map['cv_dataset']
        cv_label = data_map['cv_label']
        print("\ncurrent CV Shape : ",cv_dataset.shape, cv_label.shape)
        cv_dataset,cv_label = reformat(cv_dataset,cv_label)
        print("New CV Shape : ",cv_dataset.shape, cv_label.shape)

        test_dataset = data_map['test_dataset']
        test_label = data_map['test_label']
        print("\ncurrent test shape ", test_dataset.shape, test_label.shape)
        test_dataset,test_label = reformat(test_dataset,test_label)
        print("New test Shape : ",test_dataset.shape, test_label.shape)

except Exception as e:
    print("sorry pickle data couldn't be read: ", e)



graph = tf.Graph()
with graph.as_default():

    ''' instead  of holding all the training data in a constant node we create a placeholder
     node which will be feed the data to every call of session.run()'''
    tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_OF_LABELS))

    tf_cv_dataset = tf.constant(cv_dataset)
    tf_test_dataset  = tf.constant(test_dataset)

    # variables
    # num_of labels here because we are intrinsically running 10 1vs all models so 10 different weight sets each len = 784
    with tf.name_scope('Weights'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE*IMAGE_SIZE, NUM_OF_LABELS]))
        tf.histogram_summary('Weights', weights)

    biases = tf.Variable(tf.zeros(NUM_OF_LABELS))

    logits = tf.matmul(tf_train_dataset, weights) + biases
    print("logits", logits)
    print(logits, tf_train_labels)

    with tf.name_scope("Loss"):
        lossFunction = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
        tf.scalar_summary("Loss", lossFunction)

    # we'll now minimize the above cross entropy loss function using MGD
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(lossFunction)

    # predictions for training, validation, and test data
    train_predicitons = tf.nn.softmax(logits=logits)
    cv_predictions  = tf.nn.softmax(tf.matmul(a=tf_cv_dataset, b=weights) + biases)
    test_predicitons  = tf.nn.softmax(tf.matmul(a=tf_test_dataset, b=weights) + biases)
    print("test pred : ", test_predicitons)

    merged = tf.merge_all_summaries()

NUM_OF_STEPS = 4001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
      / predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("variables initialized")

    summary_writer = tf.train.SummaryWriter("/home/sarthak/Mydata/Projects/UdacityDeepLearning/TD_GD_logs",session.graph)

    for step in range(NUM_OF_STEPS):
        # create an offset for every batch and then feed it
        offset = (step*BATCH_SIZE)%(train_label.shape[0]-BATCH_SIZE)
        # generate the batch
        batch_data = train_dataset[offset:(offset+BATCH_SIZE), :]
        batch_labels = train_label[offset:(offset+BATCH_SIZE), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

        _, l, predicitions, summary = session.run([optimizer, lossFunction, test_predicitons, merged], feed_dict=feed_dict)

        summary_writer.add_summary(summary, step)

        if (step % 100 == 0):

            print("mini-batch loss at %d step : %f"%(step,l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predicitions, batch_labels))

            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph dependencies.
            print("Validation accuracy: %.1f%%" % accuracy(cv_predictions.eval(), cv_label))

    print("Test accuracy: %.1f%%" % accuracy(test_predicitons.eval(), test_label))








