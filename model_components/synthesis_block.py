import tensorflow as tf
from tensorflow.keras import layers, models
from .GDN import GDN

# Synthesis Block: This block decodes the latent space representation back into the image space
def synthesis_block(output_shape):
    inputs = tf.keras.Input(shape=output_shape)
    print(inputs.shape)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(inputs)
    # x = GDN(inverse=True)(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    # x = GDN(inverse=True)(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    # x = GDN(inverse=True)(x)
    x = layers.Conv2DTranspose(3, (3, 3), activation='relu', padding='same')(x)
    return models.Model(inputs, x, name='synthesis_block')
