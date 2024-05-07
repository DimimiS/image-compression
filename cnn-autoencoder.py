from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Input, Conv2DTranspose
from keras.models import Model
import matplotlib.pyplot as plt
from dataset import stl_train, stl_test
import numpy as np


# reshape data
stl_train = np.reshape(stl_train, (len(stl_train), 96, 96, 3))
stl_test = np.reshape(stl_test, (len(stl_test), 96, 96, 3))

input_layer = Input(shape=(96, 96, 3), name="INPUT")
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

code_layer = MaxPooling2D((2, 2), name="Code")(x)

x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(code_layer)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
output_layer = Conv2D(3, (3, 3), padding='same', name="OUTPUT")(x)

model = Model(input_layer, output_layer)

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
model.fit(stl_train, stl_train, epochs=10, batch_size=32, shuffle=True, validation_data=(stl_test, stl_test))

# Save model
model.save('autoencoder.keras')
