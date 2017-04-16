import numpy as np
from six.moves import cPickle as pickle

import notMnistmakeData2 as mnistMD

TRAIN_SIZE = 350000 # so we'll basically pick 35000 random samples from each of the 10 pickles  , not each pickle has 45000

TEST_SIZE = 18000


def make_array(train_size, image_size):
    dataset = np.ndarray((train_size, image_size, image_size), dtype=np.float32)  # a 3-D array of size 3,50,000 * (28*28)
    label = np.ndarray(train_size, dtype=np.int32)  # change to one hot encoding

    return dataset, label


def merge_dataset(train_size, allpickleslist):
    num_classes=len(allpickleslist)  # quite obvious we have 10 output classes

    train_dataset, train_label = make_array(train_size,mnistMD.IMAGE_SIZE)
    SIZE_PER_CLASS = int(train_size/num_classes)
    start_train=0
    end_train = SIZE_PER_CLASS

    for label, pickle_i in enumerate(allpickleslist):   # each pickle is 3D
        try:
            with open(pickle_i,'rb') as f:
                letter_set = pickle.load(f)
                np.random.shuffle(letter_set)
                train_dataset[start_train:end_train, :, :] = letter_set[:SIZE_PER_CLASS, :, :]  # picking 35,000
                train_label[start_train:end_train] = label
                start_train = end_train
                end_train = start_train + SIZE_PER_CLASS

        except Exception as e:
            print("unable to process data from ", pickle_i," :", e)

    return train_dataset, train_label


train_dataset, train_label = merge_dataset(TRAIN_SIZE, mnistMD.train_nonMNIST)

test_dataset, test_label = merge_dataset(TEST_SIZE, mnistMD.test_nonMNIST)

'''
train_dataset now has 3,50,000 samples 35,000 each for A,B ... J and note they'll all be sorted from A to J ... so shuffle them and then train
train_label now has 3,50,000 corresponding o/p labels 35000 each for A,B ... J and note they'll all be sorted from A to J ...

now that we have out training dataset, we can use this to split into training dataset and validation data set
'''

print('Training Data ------> \n X  = ' , train_dataset.shape , 'Y=', train_label.shape , '\n\n ')
print('Test Data ------> \n X  = ' , test_dataset.shape , 'Y=', test_label.shape , '\n\n ')

# lets shuffle them synch first

def randomizeData(train_dataset, train_label):
    permutation = np.random.permutation(train_label.shape[0])
    shuffled_dataset = train_dataset[permutation, : , :]  # note we are obviously not shuffling the pixels here
    shuffled_labels = train_label[permutation]

    return shuffled_dataset, shuffled_labels

train_dataset, train_label = randomizeData(train_dataset, train_label)

# split the training data into Train and Cross Validation
msk = np.random.rand(len(train_label)) <= 0.80  # creates a list of true false values

traindataset = train_dataset[msk]
trainlabel = train_label[msk]

validdataset = train_dataset[~msk]
validlabel = train_label[~msk]

testdataset = test_dataset  # just for the sake of naming convention
testlabel = test_label


print(' X [Train] = ' , traindataset.shape , 'Y [Train] =', trainlabel.shape)
print(' X [CV] = ' , validdataset.shape , 'Y [CV] =', validlabel.shape)
print(' X [Test] = ' , testdataset.shape , 'Y [Test] =', testlabel.shape)
'''
now that we have completely pre-processed the raw image files into training and cross validation sets , lets save them into pickles
'''

pickleFile = 'notMNIST.pickle'
try:
    f = open(pickleFile, 'wb')
    save = {
        'train_dataset' : traindataset,
        'train_label' : trainlabel,
        'cv_dataset' : validdataset,
        'cv_label' : validlabel,
        'test_dataset' : testdataset,
        'test_label' : testlabel
    }
    pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
    f.close()

except Exception as e:
    print('Unable to save pickle file : ' ,e )
