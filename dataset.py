import numpy as np
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt


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

stl_train_path = 'C:\\Users\\Α\\Desktop\\image-compression\\stl_train\\'

stl_train = []
for filename in os.listdir(stl_train_path):
    img = plt.imread(stl_train_path+filename)  # Add the full path to the image file
    stl_train.append(image.img_to_array(img))
    print(filename)
stl_train = np.array(stl_train)
print(stl_train.shape)

stl_test_path = 'C:\\Users\\Α\\Desktop\\image-compression\\stl_test\\'

stl_test = []
for filename in os.listdir(stl_test_path):
    img = plt.imread(stl_test_path+filename)  # Add the full path to the image file
    stl_test.append(image.img_to_array(img))
    print(filename)
stl_test = np.array(stl_test)

