import keras
import matplotlib.pyplot as plt
from model_components.dataset import train_generator, validation_generator
import numpy as np
import cv2
from model_components.model_metrics import psnr, ms_ssim

# Load the model
model = keras.models.load_model('/home/dimitra/Desktop/image-compression/cnn-gdn.keras', custom_objects={'psnr': psnr, 'ms_ssim': ms_ssim})

#  Display images
train = train_generator.next()
compressed_train = model.predict(train[0])

#  Manage GDN issue with scaling
# Assuming compressed_train contains the model's output
compressed_train_rescaled = (compressed_train - compressed_train.min()) / (compressed_train.max() - compressed_train.min())
# Clipping the rescaled output to ensure it's within the valid range
compressed_train_clipped = np.clip(compressed_train_rescaled, 0, 1)

#  Display images
# Convert the images back to RGB from YUV and normalize the pixel values
def yuv_to_rgb(yuv_image):
    # yuv_image = yuv_image.astype(np.uint8)
    return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

#  Display images RGB ranges
print(f"Training Image RGB Range: {yuv_to_rgb(compressed_train[0]).min(), yuv_to_rgb(compressed_train[0]).max()}")

#  Show just one image in both original and compressed form in the same subplot
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow((train[0][0]))
plt.subplot(2, 2, 3)
plt.title('Original Image')
plt.imshow(yuv_to_rgb(train[0][0]))
plt.subplot(2, 2, 2)
plt.title('Compressed Image')
plt.imshow((compressed_train[0]))
plt.subplot(2, 2, 4)
plt.title('Compressed Image')
plt.imshow(yuv_to_rgb(compressed_train[0]))

plt.show()