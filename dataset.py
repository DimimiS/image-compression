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

# STL-10 dataset from img file
stl_train_path = "img"

# import kodak dataset


stl_train = []
for filename in os.listdir(stl_train_path):
    if filename.endswith(".png"):
        img = image.load_img(filename)
        stl_train.append(image.img_to_array(img))
stl_train = np.array(stl_train)


print(stl_train.shape)