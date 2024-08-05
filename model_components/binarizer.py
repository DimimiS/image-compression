import tensorflow as tf
from tensorflow.keras import layers

class Sign(layers.Layer):
    def __init__(self):
        super(Sign, self).__init__()

    def call(self, inputs):
        try:
            prob = tf.random.uniform(tf.shape(inputs))
            x = tf.where((1 - inputs) / 2 <= prob, tf.ones_like(inputs), -tf.ones_like(inputs))
            return x
        except:
            return self.sign(inputs)

class Binarizer(layers.Layer):
    def __init__(self, num_channels=128):
        super(Binarizer, self).__init__()
        self.conv = layers.Conv2D(
            filters=num_channels,
            kernel_size=(1, 1),
            use_bias=False,
            kernel_initializer='he_normal'
        )
        self.sign = Sign()

    def call(self, inputs):
        feat = self.conv(inputs)
        x = tf.tanh(feat)
        return self.sign(x)