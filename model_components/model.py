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
        self.synthesis_block = synthesis_block(self.analysis_block.output_shape[1:])
        self.rnn_block_dec = RNNBlock(rnn_units)

    def call(self, inputs):
        x = self.analysis_block(inputs)
        x = self.binarizer(x)
        x = self.synthesis_block(x)
        return x
