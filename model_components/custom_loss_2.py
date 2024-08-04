import tensorflow as tf

class RateDistortionLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambda_param = lambda_param
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        # Calculate distortion using Mean Squared Error
        distortion = self.mse(y_true, y_pred)

        # Estimate bit rate based on the output tensor's shape
        batch_size = tf.cast(tf.shape(y_pred)[0], tf.float32)  # Ensure batch_size is float
        num_elements = tf.cast(tf.reduce_prod(tf.shape(y_pred)[1:]), tf.float32)  # Ensure num_elements is float
        bit_rate = (num_elements * 8) / batch_size  # Bits per image (assumes 8 bits per element)

        # Combine the rate and distortion
        return bit_rate + self.lambda_param * distortion
