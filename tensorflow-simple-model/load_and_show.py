import keras
import matplotlib.pyplot as plt
from dataset import x_test, stl_test

# Load the model
model = keras.models.load_model('autoencoder.keras')

# Predict the test images
decoded_imgs = model.predict(stl_test)

# Display the first 10 images from the test set
plt.figure(figsize=(10, 1))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(stl_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
# Display the first 10 reconstructed images
plt.figure(figsize=(10, 1))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(decoded_imgs[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
