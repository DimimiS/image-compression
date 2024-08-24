import tensorflow as tf


def compute_entropy(x):
    x_flat = tf.reshape(x, (-1,))
    histogram = tf.histogram_fixed_width(x_flat, [tf.reduce_min(x_flat), tf.reduce_max(x_flat)], nbins=256)
    probabilities = histogram / tf.reduce_sum(histogram)
    probabilities = tf.clip_by_value(probabilities, 1e-10, 1.0)  # Avoid log(0)
    entropy = -tf.cast(tf.reduce_sum(probabilities * tf.math.log(probabilities)), tf.float32)/tf.math.log(2.0)
    return entropy

class RateDistortionLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_param=0.01, **kwargs):
        super().__init__(**kwargs)
        self.lambda_param = lambda_param
        self.mse = tf.keras.losses.MeanSquaredError()


    def call(self, y_true, y_pred):
        # Calculate the number of pixels in the image
        num_pixels = tf.cast(tf.reduce_prod(tf.shape(y_true)[:-1]), tf.float32)

        # Calculate compression ratio of a single image of the batch
        bpp = compute_entropy(y_pred) / num_pixels

        # Distortion (D): Mean Squared Error
        distortion = self.mse(y_true, y_pred)
        
        # Rate (R): Entropy of the compressed image
        rate = bpp

        # Ensure rate is of type float32
        rate = tf.cast(rate, tf.float32)
        
        # Combine the losses
        return rate + self.lambda_param * distortion
