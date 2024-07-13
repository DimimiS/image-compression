import tensorflow as tf
from tensorflow.keras import layers, models
from model_components.GDN import GDN

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
