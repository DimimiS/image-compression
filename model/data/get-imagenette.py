from fastai.vision.all import untar_data, URLs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import cv2

# Helper function to convert RGB to YUV
def rgb_to_yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

# Helper function to preprocess images and convert them to the desired color space
def preprocess_image(image):
    image = image / 255.0  # Normalize to [0, 1]
    image = rgb_to_yuv(image)  # Convert to YUV
    return image

# Download the Imagenette Dataset
path = untar_data(URLs.IMAGENETTE_320)
train_dir = os.path.join(path, 'train')
valid_dir = os.path.join(path, 'val')

# Convert images from RGB to YUV
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)

