import tensorflow as tf
from tensorflow.keras import layers, models
from .GDN import GDN

# Synthesis Block: This block decodes the latent space representation back into the image space
def synthesis_block(output_shape):
    
    inputs = tf.keras.Input(shape=output_shape)
    print(inputs.shape)
    x = layers.Conv2DTranspose(128, (3, 3), padding='same')(inputs)
    x = layers.LeakyReLU()(x)
    x = GDN( inverse=True)(x)

    for _ in range(1):  # Simplified loop for repetitive blocks
        x = layers.Conv2DTranspose(64, (3, 3), padding='same')(x)
        x = layers.LeakyReLU()(x)
        x = GDN( inverse=True)(x)  # Ensure channels is an integer

    x = layers.Conv2DTranspose(3, (3, 3), padding='same')(x)
    x = layers.LeakyReLU()(x)
    return models.Model(inputs, x, name='synthesis_block')
