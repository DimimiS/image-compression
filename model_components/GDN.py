import tensorflow_compression as tfc
from tensorflow.keras import layers
import os

# Generalized Divisive Normalization (GDN) Layer
class GDN(layers.Layer):
    def __init__(self, inverse=False, **kwargs):
        super(GDN, self).__init__(**kwargs)
        self.inverse = inverse
        self.gdn = tfc.GDN(inverse=inverse)
    
    def call(self, inputs):
        return self.gdn(inputs)