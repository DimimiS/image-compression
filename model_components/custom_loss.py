import tensorflow as tf


def compute_entropy(x):
    x_flat = tf.reshape(x, [-1])
    histogram = tf.histogram_fixed_width(x_flat, [tf.reduce_min(x_flat), tf.reduce_max(x_flat)], nbins=256)
    probabilities = histogram / tf.reduce_sum(histogram)
    probabilities = tf.clip_by_value(probabilities, 1e-10, 1.0)  # Avoid log(0)
    entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities))
    return entropy

class RateDistortionLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_param=0.0001, **kwargs):
        super().__init__(**kwargs)
        self.lambda_param = lambda_param
        self.mse = tf.keras.losses.MeanSquaredError()

    def loss(self, y_true, y_pred):
        # Distortion (D): Mean Squared Error
        distortion = tf.mse(y_true - y_pred)
        
        # Rate (R): Calculate based on the number of bits
        rate = compute_entropy(y_pred)  # Using computed entropy directly

        distortion = tf.cast(distortion, tf.float32)
        rate = tf.cast(rate, tf.float32)
        
        # Combine the losses
        return rate + self.lambda_param * distortion

    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred)
