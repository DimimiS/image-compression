import numpy as np
import os
from keras.preprocessing import image


# import the dataset from local directory
# 1. CIFAR-10 dataset
# 2. STL-10 dataset

# CIFAR-10 dataset

from keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, _), (x_test, _) = cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

# Normalize the images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# reshape data
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

# these are all the unlabeled data
stl_train_path = 'C:\\Users\\dimis\\Desktop\\image-compression\\img\\'

stl_train = []
for filename in os.listdir(stl_train_path):
    img = image.load_img(stl_train_path+filename, target_size=(96, 96, 3))  # Add the full path to the image file
    stl_train.append(image.img_to_array(img))
    print(filename)
stl_train = np.array(stl_train)

# all the labelled data are tests now
stl_test_path = 'C:\\Users\\dimis\\Desktop\\image-compression\\stl_test\\'

stl_test = []
for filename in os.listdir(stl_test_path):
    img = image.load_img(stl_test_path+filename, target_size=(96, 96, 3))  # Add the full path to the image file
    stl_test.append(image.img_to_array(img))
    print(filename)
stl_test = np.array(stl_test)


# Normalize the images
slt_train = x_train.astype('float32') / 255
slt_test = x_test.astype('float32') / 255

# reshape data
stl_train = np.reshape(stl_train, (len(stl_train), 96, 96, 3))
stl_test = np.reshape(stl_test, (len(stl_test), 96, 96, 3))
print(stl_train.shape)
print(stl_test.shape)