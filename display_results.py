import keras
import matplotlib.pyplot as plt
from model_components.dataset import train_generator, validation_generator
import numpy as np
import cv2
from model_components.model_metrics import psnr, ms_ssim
from model_components.custom_loss import RateDistortionLoss
import tensorflow as tf

# Load the model
model = keras.models.load_model('/home/dimitra/Desktop/image-compression/cnn-gdn.keras', custom_objects={'psnr': psnr, 'ms_ssim': ms_ssim})

# Display images
train_batch = next(train_generator)
compressed_train = model.predict(train_batch[0])

# Convert the images back to RGB from YUV and normalize the pixel values
def yuv_to_rgb(yuv_image):
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    return rgb_image

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow((train_batch[0][0]))
plt.subplot(2, 2, 3)
plt.title('Original Image | Bit Rate: {:.2f} bits/pixel')
plt.imshow(yuv_to_rgb(train_batch[0][0]))
plt.subplot(2, 2, 2)
plt.title('Compressed Image')
plt.imshow((compressed_train[0]))
plt.subplot(2, 2, 4)
plt.title('Compressed Image | Bit Rate: {:.2f} bits/pixel')
plt.imshow(yuv_to_rgb(compressed_train[0]))

# plt.show()

def save_tensor_as_png(tensor, filename):
    tensor = tf.image.convert_image_dtype(tensor, tf.uint8)
    encoded_png = tf.image.encode_png(tensor)
    tf.io.write_file(filename, encoded_png)

# Save a fixed number of images from the generator
num_images_to_save = 32
for i in range(num_images_to_save):
    image_original = yuv_to_rgb(train_batch[0][i])  # Assuming batch size of 1
    image_compressed = yuv_to_rgb(model.predict(train_batch[0])[i])  # Assuming batch size of 1

    filename_original = f'data/original/output_image_{i}.png'
    save_tensor_as_png(image_original, filename_original)
    filename_compressed = f'data/compressed/output_image_{i}.png'
    save_tensor_as_png(image_compressed, filename_compressed)
    print(f"Image saved as '{filename_original} and '{filename_compressed}'")

    # Calculate compression ratio
    original_size = tf.io.gfile.stat(filename_original).length
    compressed_size = tf.io.gfile.stat(filename_compressed).length
    compression_ratio = original_size / compressed_size
    print(f"Compression Ratio for image {i}: {compression_ratio:.2f}")
    