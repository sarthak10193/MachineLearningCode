from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
import math
from time import time

IMAGE_SIZE = 28
NUM_OF_CLASSES = 10
BATCH_SIZE = 100

TRAIN_SIZE = 280000    # 280255
CV_SIZE = 19000        # 69745
TEST_SIZE = 18000      # 18000

NUM_OF_CHANNELS = 1
DEPTH = 16
PATCH_SIZE = 5
FC_LAYER = 64
DROPOUT = 0.50



def reformat(dataset, labels, setsize):


    dataset = np.reshape(dataset,(-1,IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS)).astype(np.float32)
    labels = (np.arange(NUM_OF_CLASSES) == labels[:,None]).astype(np.float32) # one hot encoding , Note without the asType we would get bool values obv

    return dataset[:setsize, ], labels[:setsize,]

try:
    print(" ### Getting Data from pickle ### \n\n")
    with open("/home/sarthak/PycharmProjects/udacity-deep learning/notMNIST.pickle", 'rb') as f:
        data_map = pickle.load(f)

        train_dataset = data_map['train_dataset']
        train_label = data_map['train_label']

        print("current TRAIN shapes : ", train_dataset.shape, train_label.shape)
        train_dataset, train_labels = reformat(dataset=train_dataset, labels=train_label, setsize=TRAIN_SIZE)
        print("NEW TRAIN shapes = ", train_dataset.shape , train_labels.shape, "\n")

        cv_dataset = data_map['cv_dataset']
        cv_labels = data_map['cv_label']
        print("current TRAIN shapes : ", cv_dataset.shape, cv_labels.shape)
        cv_dataset, cv_labels = reformat(dataset=cv_dataset, labels=cv_labels, setsize=CV_SIZE)
        print("NEW CV shapes = ", cv_dataset.shape, cv_labels.shape, "\n")

        test_dataset = data_map['test_dataset']
        test_labels = data_map['test_label']
        print("current test shapes : ", test_dataset.shape, test_labels.shape)
        test_dataset, test_labels = reformat(dataset=test_dataset, labels=test_labels, setsize=TEST_SIZE)
        print("NEW TEST shapes = ", test_dataset.shape, test_labels.shape, "\n\n")


except Exception as e:
    print("unable to read data from pickle")


def accuracy(predictions, labels):


    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

mygraph = tf.Graph()

with mygraph.as_default():

    tf_train_images_placeholder = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS))
    tf_train_labels_placeholder = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE, NUM_OF_CLASSES))

    tf_cv_images_placeholder = tf.constant(cv_dataset)
    tf_test_images_placeholder = tf.constant(test_dataset)


    '''
    we have 16 filters/weights at conv1. each filter is 5*5*1  , therefore the weights matrix is 5*5*1*16.
    The output these filters produce is a 14*14 image of depth/new channels 16
    '''
    with tf.name_scope(name="layer12"):

        with tf.name_scope(name="weights_12"):
            weights_12 = tf.Variable(tf.truncated_normal(shape=[PATCH_SIZE, PATCH_SIZE, NUM_OF_CHANNELS, DEPTH], stddev=0.1))
            tf.histogram_summary("weights_12", weights_12)

            x_min = tf.reduce_min(weights_12)
            x_max = tf.reduce_max(weights_12)
            kernel_0_to_1 = (weights_12 - x_min) / (x_max - x_min)
            print("kernel shape:", kernel_0_to_1.get_shape())

            #shape: (5, 5, 1, 16)  ------ > transposed: (16, 5, 5, 1) image summary format of 4D tensor []

            kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
            print("kernel transposed:", kernel_transposed.get_shape())
            tf.image_summary('conv1/filters', tensor=kernel_transposed, max_images=16)


        with tf.name_scope(name="biases_12"):
            biases_12 = tf.Variable(tf.zeros(shape=[DEPTH]))
            tf.histogram_summary("biases_12", biases_12)

    """
        The 3rd layer again has 16 filters/ set of weights , each filter being 5*5*16
        the output of 3rd layer is a 7*7 image with again 16 depth/feature maps, but then we take max pool of this output to be fed into FC layer
        Maxpool2X2 of 7*7 gives a 4*4 image with same padding with the same depth of course
    """
    with tf.name_scope(name="layer23"):

        with tf.name_scope(name="weights_23"):
            weights_23= tf.Variable(tf.truncated_normal(shape=[PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
            tf.histogram_summary("weights_23", weights_23)

        with tf.name_scope(name="biases_23"):
            biases_23 = tf.Variable(tf.constant(value=1.0, shape=[DEPTH]))
            tf.histogram_summary("biases_23", biases_23)

    with tf.name_scope(name="layer3FC"):

        with tf.name_scope(name="weights_3_FC"):
            weights_3_FC = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[4*4*DEPTH, FC_LAYER]))
            tf.histogram_summary("weights_3_FC", weights_3_FC)

        with tf.name_scope(name="biases_3_FC"):
            biases_3_FC = tf.Variable(tf.constant(value=1.0, shape = [FC_LAYER]))
            tf.histogram_summary(tag="biases_3_FC", values=biases_3_FC)

    with tf.name_scope(name="layerFC_output"):

        with tf.name_scope(name="weights_FC_output"):
            weights_FC_output = tf.Variable(tf.truncated_normal(shape=[FC_LAYER, NUM_OF_CLASSES], stddev=0.1))
            tf.histogram_summary("weights_FC_output", weights_FC_output)

        with tf.name_scope(name="biases_FC_output"):
            biases_FC_output = tf.Variable(tf.constant(value=1.0, shape=[NUM_OF_CLASSES]))
            tf.histogram_summary("biases_FC_output", biases_FC_output)

    def model(data):

        conv1 = tf.nn.conv2d(input=data, filter=weights_12, strides=[1, 2, 2, 1], padding="SAME")
        hidden1 = tf.nn.relu(conv1 + biases_12)  # basically conv1 is theta`X but only on a part of the image, then add the biases , take relu
        print("hidden 1 shape : ", hidden1.get_shape())

        conv2 = tf.nn.conv2d(input=hidden1, filter=weights_23, strides=[1, 2, 2, 1], padding="SAME")
        hidden2 = tf.nn.relu(conv2+ biases_23)
        print("Hidden 2 shape: ", hidden2.get_shape())

        """
        Note : The o/p of the second conv is a 7*7 image [49 pixels] with depth 16 ie we have 16 feature maps each responsible for detecting something
        """
        maxpool = tf.nn.max_pool(hidden2, padding="SAME", strides=[1, 2, 2 , 1], ksize=[1, 2, 2, 1])  # ksize means 2*2 maxpool
        print("Hidden 2 is 2X2 maxpooled to give: ", maxpool.get_shape(), "\n\n")

        '''
        Now, the output of the maxpool has to be reshaped to fit to the Fully connected layer with 64 units
        4*4*16 = 256
        '''
        shape = maxpool.get_shape().as_list()

        reshape = tf.reshape(tensor=maxpool, shape=[shape[0], shape[1]*shape[2]*shape[3]])

        hidden3 = tf.nn.relu(tf.matmul(reshape,weights_3_FC) + biases_3_FC)  # so this is out connection btw maxpool and FC, a simple relu of theta`X + B

        #hidden3 = tf.nn.dropout(hidden3, DROPOUT)  # adding dropout regularization to the fully connected layer

        logits = tf.matmul(hidden3, weights_FC_output) + biases_FC_output

        return logits

    logits = model(tf_train_images_placeholder)

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels_placeholder))
        tf.scalar_summary("loss", loss)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss=loss)

    train_pred = tf.nn.softmax(logits=logits)
    cv_pred = tf.nn.softmax(logits=model(data=tf_cv_images_placeholder))
    test_pred = tf.nn.softmax(logits=model(data=tf_test_images_placeholder))

    merged = tf.merge_all_summaries()

MAX_STEPS = 3000

with tf.Session(graph=mygraph) as session:
    tf.initialize_all_variables().run()
    print("variables initialized ..........")

    summary_writer = tf.train.SummaryWriter("/home/sarthak/Mydata/Projects/UdacityDeepLearning/TD_GD_logs/",
                                            session.graph)

    for step in range(MAX_STEPS):

        offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
        # generate the batch
        batch_images_data = train_dataset[offset:(offset + BATCH_SIZE), :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE):]

        feed_dict = {tf_train_images_placeholder: batch_images_data, tf_train_labels_placeholder: batch_labels}

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value, batch_predictions, summary = session.run([optimizer, loss, train_pred, merged], feed_dict=feed_dict)

        summary_writer.add_summary(summary=summary, global_step=step)

        if(step%100) ==0:
            print("batch accuracy @ step ", step , " : ", accuracy(batch_predictions, batch_labels))
            print("loss value @ step  ", step, "     : ", loss_value)
            print("\n")

    print("Evaluating cross validation data\n")
    print(accuracy(cv_pred.eval(), cv_labels))

    print("Evaluating Test data\n")
    print(accuracy(test_pred.eval(), test_labels))







