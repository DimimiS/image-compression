import tensorflow as tf
from tensorflow.keras import layers

class Sign(layers.Layer):
    def call(self, inputs):
        return tf.sign(inputs)

class Binarizer(layers.Layer):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = layers.Conv2D(128, kernel_size=1, use_bias=False)
        self.sign = Sign()

    def call(self, inputs):
        feat = self.conv(inputs)
        x = tf.tanh(feat)
        return self.sign(x)
