from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np




IMAGE_SIZE = 28
NUM_OF_CLASSES = 10
BATCH_SIZE = 100

NUM_OF_CHANNELS = 1
DEPTH = 16
PATCH_SIZE = 5
FC_LAYER = 64


TRAIN_SIZE = 280000    # 280255
CV_SIZE = 29000        # 69745  using 69000 makes the system run out of memory
TEST_SIZE = 18000      # 18000

def reformat(dataset, labels, setsize):

    dataset = np.reshape(dataset,(-1,IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS)).astype(np.float32)
    labels = (np.arange(NUM_OF_CLASSES) == labels[:,None]).astype(np.float32) # one hot encoding vector rep [0,1,0]

    return dataset[:setsize, ], labels[:setsize,]

try:
    print(" ### Getting Data from pickle ### \n\n")
    with open("/home/sarthak/PycharmProjects/udacityML/MachineLearningCode/AlphabetRecongnizer_CNN/notMNIST.pickle", 'rb') as f:
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
        print("NEW CV shapes = ", cv_dataset.shape , cv_labels.shape, "\n")

        test_dataset = data_map['test_dataset']
        test_labels = data_map['test_label']
        print("current test shapes : ", test_dataset.shape, test_labels.shape)
        test_dataset, test_labels = reformat(dataset=test_dataset, labels=test_labels, setsize=TEST_SIZE)
        print("NEW TEST shapes = ", test_dataset.shape , test_labels.shape)


except Exception as e:
    print("Sorry couldn't read file : ", e)

# ################################################################################################

def accuracy(predictions, labels):
    '''
    :param predictions: shape is 10*100 where 100 is the batch size and 10 is the predicted prob list for a given input image
                        we use argmax to pick out the max probability for all the 100 lists, argmax returns the index of max value[predicted label value]
    :param labels: know values for that batch
    :return: , compare them to the predictions and return the batch accuracy %

    '''

    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


mygraph = tf.Graph()
with mygraph.as_default():

    tf_train_images_placeholder = tf.placeholder(dtype=tf.float32, shape= (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_OF_CHANNELS))
    tf_train_labels_placeholder = tf.placeholder(dtype=tf.int32, shape = (BATCH_SIZE, NUM_OF_CLASSES))

    tf_cv_images_placeholder = tf.constant(cv_dataset)
    tf_test_images_placeholder = tf.constant(test_dataset)

    '''
    From I/P to CN1 :
    we have a 28*28 image with depth/no_of_channels  = 1 (grey scale)
    select patch size = 5*5
    select stride = 2
    we are creating 16 filters. each filter is responsible for detecting some feature
    therefore the layer 2 has depth/channels = 16 . these are also knows as feature maps , we have 16 feature maps at layer 2.

    Note : the number of neurons for a given feature map at layer 2 will depend on stride value and padding.
    and it will be the same for all 64 feature maps.

    for a stride of 2 and same padding we end up with 28*28 images with depth of 64
    '''
    weights_12 = tf.Variable(tf.truncated_normal(shape=[PATCH_SIZE, PATCH_SIZE, NUM_OF_CHANNELS, DEPTH],stddev=0.1 ))
    print("[1-->2]", weights_12.get_shape())
    biases_12 = tf.Variable(tf.zeros(shape=[DEPTH]))

    '''
    From CN1 to CN2 :
    redo above step , from cn1 to cn2
    THis time your input is cn1 with depth/no of channels = 16 mapped to cn2. For cn2 we again select 16 feature maps
    take feature map 1 of cn1 apply filter 1, 2, ,3, ...... 16
    take feature map 2 of cn1 apply filter 1, 2, 3, ....... 16
    .
    .
    .
    take feature map 16 of cn1 apply filter 1, 2, 3, ....... 16
    Therefore patch_size, patch_size * current feature maps * no of filters

    '''
    weights_23 = tf.Variable(tf.truncated_normal(shape=[PATCH_SIZE, PATCH_SIZE, DEPTH, DEPTH], stddev=0.1))
    print("[2-->3] :", weights_23.get_shape())
    biases_23 = tf.Variable(tf.constant(1.0, shape=[DEPTH]))

    '''
    Cn2 to hidden layer : fully connected layer , just reshape your last conv output to fit a fully connected layer
    since last conv output is 7*7*16 we get 784
    '''
    weights_3_FC = tf.Variable(tf.truncated_normal(shape=[IMAGE_SIZE//4 * IMAGE_SIZE//4 * DEPTH, FC_LAYER], stddev=0.1))
    print("3 to Fullyconnected : ", weights_3_FC.get_shape())
    biases_3_FC = tf.Variable(tf.constant(1.0, shape=[FC_LAYER]))

    weights_hidden_op = tf.Variable(tf.truncated_normal(shape=[FC_LAYER, NUM_OF_CLASSES], stddev=0.1))
    print("Fully connected to output : ", weights_hidden_op.get_shape())
    biases_hidden_op = tf.Variable(tf.constant(1.0, shape=[NUM_OF_CLASSES]))


    def model(data):
        '''
        conv2d() does the convolution that is takes, the image data, filters , stride values , and padding and computes the
            matrix multiplication of filters weights and image pixel.
        Then as usual we add the bias and do a relu() to compute the output activation values out of this layer
        :param data: image data
        :return: logits
        '''
        conv1 = tf.nn.conv2d(input=data, filter= weights_12, strides=[1,2,2,1], padding="SAME")
        hidden1 = tf.nn.relu(conv1 + biases_12)
        '''
        output_1 = (28.00 - 5.00 - (-2.00)) / 2.00 + 1.00 = floor(13.5) +1 = 14
        O= (W−K−2P)/S+1
        so using same padding and stride = 2 we get h = 14 , w = 14 , depth = 16   [batch size = 100]
        so basically second layer has 16 levels of  14*14=296  stacked on top of each other
        '''
        print(hidden1)

        conv2 = tf.nn.conv2d(input=hidden1, filter=weights_23, strides=[1,2,2,1], padding="SAME")
        hidden2 = tf.nn.relu((conv2 + biases_23))
        '''
        SO  using SAME padding and stride = 2 we now get h = 7 , w= 7 , depth 16   [batch size = 100]
        '''
        print(hidden2)

        shape = hidden2.get_shape().as_list()
        reshape = tf.reshape(tensor=hidden2, shape = [shape[0], shape[1]*shape[2]*shape[3]])   # [100, 7*&*16] = 100, 784  reshaping hidden 2 , the op of conv2

        hidden3 = tf.nn.relu(tf.matmul(reshape, weights_3_FC) + biases_3_FC)   # hidden 3 is the FC layer we have

        logits = tf.matmul(hidden3, weights_hidden_op) + biases_hidden_op

        return logits

    logits = model(tf_train_images_placeholder)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels_placeholder))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

    train_pred = tf.nn.softmax(logits=logits) # applying the activation function , which in this case is softmax activation
    cv_pred = tf.nn.softmax(logits=model(tf_cv_images_placeholder))
    test_pred = tf.nn.softmax(logits=model(tf_test_images_placeholder))


MAX_STEPS = 2000

with tf.Session(graph=mygraph) as session:
    tf.initialize_all_variables().run()
    print("variables initialized ..........")

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
        _, loss_value, batch_predictions = session.run([optimizer, loss, train_pred], feed_dict=feed_dict)


        if(step%100) ==0:
            print("batch accuracy @ step ", step , " : ", accuracy(batch_predictions, batch_labels))
            print("loss value @ step  ", step, "     : ", loss_value)
            print("\n")

    print("Evaluating cross validation data\n")
    print(accuracy(cv_pred.eval(), cv_labels))

    print("Evaluating Test data\n")
    print(accuracy(test_pred.eval(), test_labels))


















