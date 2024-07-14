import tensorflow as tf
from tensorflow.keras.layers import Reshape
from tensorflow.keras import layers
from .analysis_block import analysis_block
from .binarizer import Binarizer
from .rnn_block import RNNBlock
from .synthesis_block import synthesis_block

# Image Compression Model: This is the main model that integrates all the components

class ImageCompressionModel(tf.keras.Model):
    def __init__(self, input_shape, rnn_units=128, **kwargs):
        super(ImageCompressionModel, self).__init__(**kwargs)
        self.analysis_block = analysis_block(input_shape)
        self.binarizer = Binarizer()
        self.rnn_block_enc = RNNBlock(rnn_units)
        # Add a reshape layer here if the rnn_block_enc doesn't output the expected shape
        self.reshape_before_dim_reduction = layers.Reshape((32, 256, 128))  # Adjust the target shape as needed
        self.dimension_reduction_layer = layers.Conv2D(128, (1, 1), strides=(8, 1), padding='same')
        self.reshape_layer = Reshape((32, 32, 128))
        self.synthesis_block = synthesis_block(self.analysis_block.output_shape[1:])
        self.rnn_block_dec = RNNBlock(rnn_units)

    def call(self, inputs):
        x = self.analysis_block(inputs)
        # x = self.binarizer(x)
        # x = self.rnn_block_enc(x)
        # # Apply the reshape before dimension reduction if needed
        # x = self.reshape_before_dim_reduction(x)
        # x = self.dimension_reduction_layer(x)
        # x = self.reshape_layer(x)
        x = self.synthesis_block(x)
        # Clipping the output to ensure values are within [0, 1]
        x = tf.clip_by_value(x, 0.0, 1.0)
        return x
