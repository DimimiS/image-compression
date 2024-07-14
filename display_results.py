import keras
import matplotlib.pyplot as plt
from model_components.dataset import train_generator, validation_generator


# Load the model
model = keras.models.load_model('/home/dimitra/Desktop/image-compression/cnn-only.keras')

#  Display images
train = train_generator.next()
compressed_train = model.predict(train[0])

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
