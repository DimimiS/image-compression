import tensorflow as tf


# Function that calculates the average bits per pixel (bpp) of the predicted batch of images after calculating the bpp of each image accurately
import tensorflow as tf

def calculate_bit_rate(y_true, y_pred):
    # Get the batch size dynamically
    batch_size = tf.shape(y_pred)[0]
    
    # Assuming y_pred has shape (batch_size, height, width, channels)
    height = tf.shape(y_pred)[1]
    width = tf.shape(y_pred)[2]
    
    compression_ratio = 0.0
    for i in range(batch_size):
        # Cast the image to uint8
        image_uint8 = tf.cast(y_true[i], tf.uint8)
        compressed_uint8 = tf.cast(y_pred[i], tf.uint8)

        # Get number of bytes for the compressed image without encoding to png
        compressed_size = tf.strings.length(tf.image.encode_png(compressed_uint8))
        # Get number of bytes for the original image without encoding to png
        original_size = tf.strings.length(tf.image.encode_png(image_uint8))
        # Calculate the compression ratio
        compression_ratio += tf.cast(original_size, tf.float32) / tf.cast(compressed_size, tf.float32)

    # Calculate the average compression ratio
    avg_compression_ratio = compression_ratio / tf.cast(batch_size, tf.float32)
    # Calculate the bits per pixel (bpp)
    bpp = 8 / avg_compression_ratio

    return 1/avg_compression_ratio

class RateDistortionLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_param=0.001, **kwargs):
        super().__init__(**kwargs)
        self.lambda_param = lambda_param
        self.mse = tf.keras.losses.MeanSquaredError()


    def call(self, y_true, y_pred):

        # Distortion (D): Mean Squared Error
        distortion = self.mse(y_true, y_pred)

        # Rate (R): Bits per Pixel (bpp)
        rate = calculate_bit_rate(y_true, y_pred)
        
        # Combine the losses
        return rate + self.lambda_param * distortion
