import tensorflow as tf
from tensorflow.keras import layers, models
from model_components.GDN import GDN

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
