import tensorflow as tf
from tensorflow.keras import layers

# Binarizer: This layer quantizes the encoded latent space representation to binary values
class Binarizer(layers.Layer):
    def __init__(self, **kwargs):
        super(Binarizer, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.sign(inputs)
