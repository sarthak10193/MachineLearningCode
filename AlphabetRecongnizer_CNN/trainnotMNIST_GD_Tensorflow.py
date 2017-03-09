from six.moves import cPickle as pickle
import numpy as np
from time import time
import tensorflow as tf

IMAGE_SIZE = 28
NUM_OF_LABELS = 10
MAX_STEPS = 1001   # max steps for Gradient descent

def reformat(dataset, labels):
    dataset = np.reshape(dataset, (-1,IMAGE_SIZE*IMAGE_SIZE)).astype(np.float32)
    labels = (np.arange(NUM_OF_LABELS) == labels[:,None]).astype(np.float32)
    return dataset, labels

try:
    with open('notMNIST.pickle','rb') as f:
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
    print("unable to process data map pickle : ", e)

'''
Now lets perform some of the same and new tasks using the tensorflow library
1. usual logistic regression classifier using gd   -- did this using sklearn before , though used lfgs and not gd
2. usual logistic regression classifier using sgd  -- did using sklearn before
3. our first attempt at a basic 1 hidden layer neural network
'''
print("\n##### welcome to tensorflow #######\n")

start_time = time()
# note out of the available approx 3,50,000 training set data we are running GD on only 20,000
# plot a graph and see how loss varies with number of traning data set to decide the best value
TRAIN_SUBSET = 15000

# creating a model graph for data flow
graph = tf.Graph()
with graph.as_default():

    '''Creates a constant tensor of given shape and datatype '''
    tf_train_dataset = tf.constant(train_dataset[:TRAIN_SUBSET, ])
    tf_train_labels = tf.constant(train_label[:TRAIN_SUBSET, ])
    print(tf_train_dataset.get_shape())

    tf_cv_dataset  = tf.constant(cv_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    '''
    Note : previously no biases were considered in trainnotMNIST1 and 2
    and this randomization was internally handled by the sklearn function for logreg
    '''
    with tf.name_scope('weights'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE*IMAGE_SIZE, NUM_OF_LABELS]))
        print("weights : ", weights.get_shape())

    with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([NUM_OF_LABELS]))


    ''' logits represents (theta)`X  = z '''
    logits = tf.matmul(tf_train_dataset, weights) + biases
    print("theta transpose X : ", logits)

    ''' Defining the cross entropy loss function to compute loss btw logits and labels '''
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)

        with tf.name_scope('total'):
            loss = tf.reduce_mean(cross_entropy)


    ''' Using the gradient descent optimizer to compute the minimum of the loss funtion '''
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    '''
    remember we had h(x|parameterized by theta) = 1/(1+(e)^-theta'x)
    '''
    train_pred = tf.nn.softmax(logits)  # softmax converts it into probabilities

    cv_pred = tf.nn.softmax(tf.matmul(tf_cv_dataset, weights) + biases)

    test_pred = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    init = tf.initialize_all_variables()
# lets run this graph we modelled with all the placeholders etc on a session
'''
After we've initialized the SummaryWriters, we have to add summaries to the SummaryWriters as we train and test the model.
'''

NUM_OF_STEP = 1001

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
      / predictions.shape[0])


with tf.Session(graph=graph) as session:
    '''
    this is a one time operation that ensures that the parameters get initialized as we described in the
    graph above, ie random weights for the for the matrix and zero bias values
    '''
    session.run(init)
    print("All variables initialized")

    for step in range(NUM_OF_STEP):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy arrays
        _, current_loss, predictions = session.run([optimizer, loss, train_pred])

        if step % 100 == 0:
            print("loss at step %d: %f " % (step, current_loss))
            print("Training accuracy : %.1f" % accuracy(predictions, train_label[:TRAIN_SUBSET, ]))

            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph dependencies.
            print('Validation accuracy: %.1f%%' % accuracy(cv_pred.eval(), cv_label))

    print('Test accuracy: %.1f%%' % accuracy(test_pred.eval(), test_label))








