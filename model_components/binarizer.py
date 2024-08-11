import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

class Sign(layers.Layer):
    def call(self, inputs):
        return tf.sign(inputs)

class Binarizer(layers.Layer):
    def __init__(self, num_channels=128):
        super(Binarizer, self).__init__()
        self.sign = Sign()

    def call(self, inputs):
        # Apply tanh activation
        feat = tf.tanh(inputs)
        
        # Check if we are in training mode
        training = K.learning_phase()  # Returns 1 for training and 0 for inference
        
        if training:
            # Quantization noise during training
            prob = tf.random.uniform(shape=tf.shape(feat))
            binary_output = tf.where(tf.math.less_equal((1 - feat) / 2, prob), tf.ones_like(feat), -tf.ones_like(feat))
        else:
            # Deterministic sign function during evaluation
            binary_output = self.sign(feat)
        
        # Straight-Through Estimator: gradient approximation
        return binary_output + (inputs - binary_output)