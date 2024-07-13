from fastai.vision.all import untar_data, URLs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


# Download the Imagenette Dataset
path = untar_data(URLs.IMAGENETTE)
train_dir = os.path.join(path, 'train')
valid_dir = os.path.join(path, 'val')

# Prepare the Data Loaders using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='input',  # Set to 'input' for image-to-image autoencoding
    subset='training'
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    valid_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='input',  # Set to 'input' for image-to-image autoencoding
    subset='validation'
)
