import keras
import matplotlib.pyplot as plt
from model_components.dataset import train_generator, validation_generator
import numpy as np


# Load the model
model = keras.models.load_model('/home/dimitra/Desktop/image-compression/cnn-gdn.keras')

#  Display images
train = train_generator.next()
compressed_train = model.predict(train[0])

#  Manage GDN issue with scaling
# Assuming compressed_train contains the model's output
compressed_train_rescaled = (compressed_train - compressed_train.min()) / (compressed_train.max() - compressed_train.min())
# Clipping the rescaled output to ensure it's within the valid range
compressed_train_clipped = np.clip(compressed_train_rescaled, 0, 1)

#  Display images RGB ranges
print(f"Training Image RGB Range: {compressed_train[0].min(), compressed_train[0].max()}")

#  Show just one image in both original and compressed form in the same subplot
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(train[0][0])
plt.subplot(1, 2, 2)
plt.title('Compressed Image')
plt.imshow(compressed_train[0])

plt.show()
#  Show images dimensions
print(f"Training Image Shape: {train[0].shape}")
# Training Image Shape: (320, 320, 3)
