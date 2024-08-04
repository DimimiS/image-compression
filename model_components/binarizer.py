import tensorflow as tf
from tensorflow.keras import layers

class Sign(layers.Layer):
    def call(self, inputs):
        return tf.sign(inputs)

class Binarizer(layers.Layer):
    def __init__(self, num_channels=128):
        super(Binarizer, self).__init__()
        self.conv = layers.Conv2D(
            num_channels,
            kernel_size=1,
            use_bias=False,
            kernel_initializer='he_normal'  # Using He normal initialization
        )
        self.batch_norm = layers.BatchNormalization()  # Adding Batch Normalization
        self.sign = Sign()

    def call(self, inputs):
        feat = self.conv(inputs)
        feat = self.batch_norm(feat)  # Normalize the feature maps
        x = tf.tanh(feat)

        # Binarization with Straight-Through Estimator
        binary_output = self.sign(x)
        # Using straight-through estimator for gradients
        return tf.where(tf.abs(x) > 0.5, tf.ones_like(x), tf.zeros_like(x)) + (inputs - tf.where(tf.abs(x) > 0.5, tf.ones_like(x), tf.zeros_like(x)))
