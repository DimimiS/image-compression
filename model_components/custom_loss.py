import tensorflow as tf

def compute_entropy(x):
    x_flat = tf.reshape(x, [-1])
    histogram = tf.histogram_fixed_width(x_flat, [tf.reduce_min(x_flat), tf.reduce_max(x_flat)], nbins=256)
    probabilities = histogram / tf.reduce_sum(histogram)
    entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-10))
    return entropy

class RateDistortionLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_param = lambda_param
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        distortion = self.mse(y_true, y_pred)
        rate = compute_entropy(y_pred)
        rate = tf.cast(rate, tf.float32)
        distortion = tf.cast(distortion, tf.float32)
        return rate + self.lambda_param * distortion