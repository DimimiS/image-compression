import tensorflow as tf
from tensorflow.keras import layers, models
from .GDN import GDN

# Analysis Block: This block encodes the input image into a latent space representation
def analysis_block(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    channels = input_shape[-1] 
    x = layers.Conv2D(3, (3, 3), padding='same')(inputs)
    x = GDN(channels=channels)(x)
    x = layers.ReLU()(x)

    # for _ in range(2):  # Simplified loop for repetitive blocks
    #     x = layers.Conv2DTranspose(128, (3, 3), padding='same')(x)
    #     x = GDN(channels=128)(x)  # Ensure channels is an integer
    #     x = layers.ReLU()(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    print(x.shape)
    return models.Model(inputs, x, name='analysis_block')
