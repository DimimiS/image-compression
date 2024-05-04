import numpy as np
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Input, Conv2DTranspose
from keras.models import Model
import matplotlib.pyplot as plt


# import the dataset from local directory

from keras.datasets import cifar10

# Load the CIFAR-10 dataset
(x_train, _), (x_test, _) = cifar10.load_data()

print(x_train.shape)
print(x_test.shape)

# Normalize the images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# reshape data
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

input_layer = Input(shape=(32, 32, 3), name="INPUT")
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

code_layer = MaxPooling2D((2, 2), name="Code")(x)

x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(code_layer)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
output_layer = Conv2D(3, (3, 3), padding='same', name="OUTPUT")(x)

model = Model(input_layer, output_layer)

model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

# Train the model
model.fit(x_train, x_train, epochs=50, batch_size=32, shuffle=True, validation_data=(x_test, x_test))

# Save model
model.save('autoencoder.keras')
