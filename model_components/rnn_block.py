import tensorflow as tf
from tensorflow.keras.layers import LSTM, Reshape

# Recurrent Neural Network (RNN) Block: This block applies LSTM to the encoded representation
class RNNBlock(tf.keras.layers.Layer):
    def __init__(self, rnn_units, **kwargs):
        super(RNNBlock, self).__init__(**kwargs)
        self.lstm = LSTM(units=rnn_units, return_sequences=True, return_state=False)
        # Assuming the input shape to the block is (batch_size, 16, 16, 128)
        # Reshape it to (batch_size, 256, 128) where 256 is the new timestep dimension
        self.reshape = Reshape((16*16, 128))  # Adjust according to your specific dimensions

    def call(self, inputs):
        x = self.reshape(inputs)  # Reshape the input to match LSTM's expected input shape
        x = self.lstm(x)
        return x