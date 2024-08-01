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

# Prepare the Data Loaders using ImageDataGenerator
train_generator = ImageDataGenerator(preprocessing_function=preprocess_image).flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='input'
)

validation_generator = ImageDataGenerator(preprocessing_function=preprocess_image).flow_from_directory(
    valid_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='input'
)
#  Display images

# Display a few images from the training set
train, _ = next(train_generator)
valid, _ = next(validation_generator)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(train[i])
    plt.axis("off")

# plt.show()

#  Show images dimensions
print(f"Training Image Shape: {train[0].shape}")
# Training Image Shape: (320, 320, 3)
