import tensorflow as tf
from tensorflow.keras import layers, models
from .GDN import GDN

# Synthesis Block: This block decodes the latent space representation back into the image space
def synthesis_block(output_shape):
    channels = output_shape[-1]  # Get the number of channels from the input tensor
    
    inputs = tf.keras.Input(shape=output_shape)
    print(inputs.shape)
    x = layers.Conv2DTranspose(128, (3, 3), padding='same')(inputs)
    x = GDN(channels=channels, inverse=True)(x)
    x = layers.ReLU()(x)

    # for _ in range(2):  # Simplified loop for repetitive blocks
    #     x = layers.Conv2DTranspose(128, (3, 3), padding='same')(x)
    #     x = GDN(channels=128, inverse=True)(x)  # Ensure channels is an integer
    #     x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(3, (3, 3), activation='relu', padding='same')(x)
    return models.Model(inputs, x, name='synthesis_block')
