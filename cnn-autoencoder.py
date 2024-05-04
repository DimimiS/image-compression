from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Input, Conv2DTranspose
from keras.models import Model
import matplotlib.pyplot as plt
from dataset import x_train, x_test, stl_train, stl_test


input_layer = Input(shape=(96, 96, 3), name="INPUT")
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

model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model
model.fit(stl_train, stl_train, epochs=30, batch_size=32, shuffle=True, validation_data=(stl_test, stl_test))

# Save model
model.save('autoencoder.keras')
