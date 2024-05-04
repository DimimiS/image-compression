import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.datasets import mnist


# -------------------------------------
# image preprocessing
# -------------------------------------

# load mnist data
(x_train, _), (x_test, _) = mnist.load_data()

# data normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# reshape data
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# -------------------------------------
# exploratory data analysis
# -------------------------------------

# randomly select input image
index = np.random.randint(0, len(x_test))
image = x_test[index].reshape(28, 28)
# plt.imshow(image, cmap='gray')


# -------------------------------------
# model definition
# -------------------------------------

model = Sequential([
    # encoder
    Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D(2, padding='same'),
    Conv2D(16, 3, activation='relu', padding='same'),
    MaxPooling2D(2, padding='same'),
    Conv2D(8, 3, activation='relu', padding='same'),
    MaxPooling2D(2, padding='same'),
    
    # decoder
    Conv2D(8, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    Conv2D(16, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    Conv2D(32, 3, activation='relu'),
    UpSampling2D(2),

    # output layer
    Conv2D(1, 3, activation='sigmoid', padding='same')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()

# -------------------------------------
# model training
# -------------------------------------

model.fit(x_train, x_train, epochs=1, batch_size=128, validation_data=(x_test, x_test))

# -------------------------------------
# prediction (get compressed images)
# -------------------------------------

compressed_images = model.predict(x_test)

# -------------------------------------
# visualize compressed images
# -------------------------------------
comp_image = compressed_images[index].reshape(28, 28)
# plt.imshow(comp_image, cmap='gray')

plt.imshow(comp_image, cmap='gray')
plt.show()