from fastai.vision.all import untar_data, URLs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# Download the Imagenette Dataset
path = untar_data(URLs.IMAGENETTE_320)
train_dir = os.path.join(path, 'train')
valid_dir = os.path.join(path, 'val')

# Prepare the Data Loaders using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(320, 320),
    batch_size=32,
    class_mode='input'
)

valid_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(320, 320),
    batch_size=32,
    class_mode='input'
)

# #  Display images
# import matplotlib.pyplot as plt

# # Display a few images from the training set
# images, _ = next(train_generator)
# plt.figure(figsize=(10, 10))
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i])
#     plt.axis("off")

# plt.show()

