import keras
import matplotlib.pyplot as plt
from model_components.dataset import train_generator, validation_generator
import numpy as np
import cv2
from model_components.model_metrics import psnr, ms_ssim
from model_components.custom_loss import RateDistortionLoss

# loss = RateDistortionLoss()

# Load the model
model = keras.models.load_model('/home/dimitra/Desktop/image-compression/cnn-gdn.keras', custom_objects={'psnr': psnr, 'ms_ssim': ms_ssim})

#  Display images
train = train_generator.next()
compressed_train = model.predict(train[0])

#  Display images
# Convert the images back to RGB from YUV and normalize the pixel values
def yuv_to_rgb(yuv_image):
    rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    # Clip or normalize the RGB image to be within the valid range
    # rgb_image = np.clip(rgb_image, 0, 255)  # Assuming the image is in 8-bit format
    return rgb_image


def calculate_bit_rate(image_array):
    # Get image dimensions and number of channels
    height, width, num_channels = image_array.shape
    
    # For most images, each channel will be 8 bits (1 byte)
    bits_per_channel = 8
    
    # Calculate the bit rate
    total_pixels = width * height
    bit_rate = total_pixels * num_channels * bits_per_channel
    
    return bit_rate

original_bit_rate = calculate_bit_rate(train[0][0])
compressed_bit_rate = calculate_bit_rate(compressed_train[0])

# Get image's bit rate per pixel
original_bit_rate_per_pixel = original_bit_rate / (train[0][0].shape[0] * train[0][0].shape[1])
compressed_bit_rate_per_pixel = compressed_bit_rate / (compressed_train[0].shape[0] * compressed_train[0].shape[1])

#  Display images RGB ranges
print(f"Training Image RGB Range: {yuv_to_rgb(compressed_train[0]).min(), yuv_to_rgb(compressed_train[0]).max()}")

#  Show just one image in both original and compressed form in the same subplot
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow((train[0][0]))
plt.subplot(2, 2, 3)
plt.title('Original Image | Bit Rate: {:.2f} bits'.format(original_bit_rate_per_pixel))
plt.imshow(yuv_to_rgb(train[0][0]))
plt.subplot(2, 2, 2)
plt.title('Compressed Image')
plt.imshow((compressed_train[0]))
plt.subplot(2, 2, 4)
plt.title('Compressed Image | Bit Rate: {:.2f} bits'.format(compressed_bit_rate_per_pixel))
plt.imshow(yuv_to_rgb(compressed_train[0]))

plt.show()