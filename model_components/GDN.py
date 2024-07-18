import tensorflow as tf

class GDN(tf.keras.layers.Layer):
    def __init__(self, channels, gamma_init=0.1, inverse=False, **kwargs):
        super(GDN, self).__init__(**kwargs)
        self.channels = channels
        self.gamma_init = gamma_init
        self.inverse = inverse

    def build(self, input_shape):
        if self.inverse:
            # Initialize weights for inverse GDN if needed
            self.gamma = self.add_weight(
                name='gamma_inverse',
                shape=(1, 1, self.channels, self.channels),
                initializer=tf.keras.initializers.RandomNormal(mean=self.gamma_init, stddev=0.05),
                trainable=True
            )
        else:
            # Initialize weights for original GDN
            self.gamma = self.add_weight(
                name='gamma',
                shape=(1, 1, self.channels, self.channels),
                initializer=tf.keras.initializers.RandomNormal(mean=self.gamma_init, stddev=0.05),
                trainable=True
            )
        super(GDN, self).build(input_shape)

    def call(self, inputs):
        if self.inverse:
            # Implement the inverse GDN operation
            x = tf.nn.relu(tf.nn.conv2d(inputs, tf.linalg.inv(self.gamma), strides=[1, 1, 1, 1], padding='SAME'))
        else:
            # Original GDN operation
            x = tf.nn.relu(tf.nn.conv2d(inputs, self.gamma, strides=[1, 1, 1, 1], padding='SAME'))
        return x
