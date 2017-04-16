import os

import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle

import extractNotMNIST1 as exnmnist

'''
Next important task is to convert the data set into 3D numpy array storing the :
actual label mapped to the 2D pixel representation of the image
'''
IMAGE_SIZE = 28
PIXEL_DEPTH = 255.0

def getPixelRepresentation(current_letter_folder, min_size):

    '''processing the folder for the currently passed letter eg A'''

    image_files = os.listdir(current_letter_folder) # this give all the images (.png) of that folder in the list
    print("the number of images  [.png] in this folder are : ", len(image_files))
    dataset = np.ndarray(shape=(len(image_files), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    num_images  = 0
    for image in image_files:
        distinct_imagefile  = os.path.join(current_letter_folder,image)  # defines the path for 1 particular image in that letter folder
        try:
            distinct_imagefile_data = (ndimage.imread(distinct_imagefile).astype(float)-PIXEL_DEPTH / 2)/ PIXEL_DEPTH
            if distinct_imagefile_data.shape != (IMAGE_SIZE, IMAGE_SIZE):  # basically must be a 28 X 28 image
                raise Exception('Unexpected Image shape')
            dataset[num_images, :, :] = distinct_imagefile_data
            num_images+=1

        except IOError as e:
            print('Could not read', distinct_imagefile, ':' ,e , 'skipping this failed case')

     # so finally this is what contains pixel rep of each image indexed from 0 to num_images
    dataset = dataset[0:num_images, : , :]
    if num_images < min_size:
        raise Exception(" not all images were proceseed : " , 'total : ', min_size, '  processed : ', num_images)

    print("full dataset Tensor created for the current letter : ", dataset.shape)
    print("mean : ", np.mean(dataset))
    print("std deviation : ", np.std(dataset))


    return dataset


def getData(data_folder, min_size, force=False):
    dataset = []
    for folder in data_folder:
        filename = folder + '.pik'
        dataset.append(filename)
        if os.path.exists(filename) and not force:
            print('pixel representation matrix for %s already exists ' %filename)
        else:
            print('pixel representation matrix for %s NOT present .... processing  ' %filename)
            dataset_per_Letter = getPixelRepresentation(folder, min_size)   # passing a letter folder
            try:
                with open(filename,'wb') as f:
                    pickle.dump(dataset_per_Letter, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print("unable to save data")

    return dataset


# passing the folders list  : A, B ..... J  where each folder has some .jpg images
print("training data : \n")
train_nonMNIST = getData(exnmnist.train_folders, 45000)  # 45000 min images per letter folder is what we want to transform : training data (/52913)
print("testing data : \n")
test_nonMNIST = getData(exnmnist.test_folders, 1800)    # 1800 min images per letter folder is what we want to transform : training data  (/1874)




