import tensorflow as tf
import tensorflow_compression as tfc
from tensorflow.keras import layers, models
from fastai.vision.all import untar_data, URLs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Generalized Divisive Normalization (GDN) Layer
class GDN(layers.Layer):
    def __init__(self, inverse=False, **kwargs):
        super(GDN, self).__init__(**kwargs)
        self.inverse = inverse
        self.gdn = tfc.GDN(inverse=inverse)
    
    def call(self, inputs):
        return self.gdn(inputs)

# Analysis Block: This block encodes the input image into a latent space representation
def analysis_block(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(inputs)
    x = GDN()(x)
    x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
    x = GDN()(x)
    x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
    x = GDN()(x)
    x = layers.Conv2D(128, kernel_size=5, strides=2, padding='same')(x)
    x = GDN()(x)
    return models.Model(inputs, x, name='analysis_block')

# Synthesis Block: This block decodes the latent space representation back into the image space
def synthesis_block(output_shape):
    inputs = tf.keras.Input(shape=output_shape)
    x = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(inputs)
    x = GDN(inverse=True)(x)
    x = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x)
    x = GDN(inverse=True)(x)
    x = layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(x)
    x = GDN(inverse=True)(x)
    x = layers.Conv2DTranspose(3, kernel_size=5, strides=2, padding='same')(x)
    return models.Model(inputs, x, name='synthesis_block')

# Binarizer: This layer quantizes the encoded latent space representation to binary values
class Binarizer(layers.Layer):
    def __init__(self, **kwargs):
        super(Binarizer, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.sign(inputs)

# Recurrent Neural Network (RNN) Block: This block applies LSTM to the encoded representation
class RNNBlock(layers.Layer):
    def __init__(self, num_units=128, **kwargs):
        super(RNNBlock, self).__init__(**kwargs)
        self.lstm = layers.LSTM(num_units, return_sequences=True)
    
    def call(self, inputs):
        return self.lstm(inputs)

# Image Compression Model: This is the main model that integrates all the components
class ImageCompressionModel(tf.keras.Model):
    def __init__(self, input_shape, rnn_units=128, **kwargs):
        super(ImageCompressionModel, self).__init__(**kwargs)
        self.analysis_block = analysis_block(input_shape)
        self.binarizer = Binarizer()
        self.rnn_block_enc = RNNBlock(rnn_units)
        self.synthesis_block = synthesis_block((32, 32, 128))
        self.rnn_block_dec = RNNBlock(rnn_units)
    
    def call(self, inputs):
        x = self.analysis_block(inputs)  # Encode the image
        x = self.binarizer(x)  # Quantize the encoded representation
        x = self.rnn_block_enc(x)  # Apply LSTM encoding
        x = self.synthesis_block(x)  # Decode the quantized representation
        x = self.rnn_block_dec(x)  # Apply LSTM decoding
        return x

# Custom Loss Function: This function calculates the mean absolute error between the true and predicted images
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

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

# Compile the Model with Adam optimizer and custom loss
input_shape = (256, 256, 3)
model = ImageCompressionModel(input_shape)
model.compile(optimizer='adam', loss=custom_loss)

# Train the Model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)

# Evaluate the Model on the validation set
loss = model.evaluate(validation_generator)
print(f"Validation Loss: {loss}")
