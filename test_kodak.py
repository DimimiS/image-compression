import matplotlib.pyplot as plt
import keras
import numpy as np
import cv2
from model_components.model_metrics import psnr, ms_ssim
from model_components.custom_loss import RateDistortionLoss
import tensorflow as tf
import os
from display_results import save_tensor_as_png

# Load the model
model = keras.models.load_model('/home/dimitra/Desktop/image-compression/cnn-gdn.keras', custom_objects={'psnr': psnr, 'ms_ssim': ms_ssim})

# Initialize an empty list to store the images
kodak_images = []

# Function to crop the center of the image
def center_crop(image, crop_size):
    h, w, _ = image.shape
    start_x = w // 2 - (crop_size // 2)
    start_y = h // 2 - (crop_size // 2)
    return image[start_y:start_y + crop_size, start_x:start_x + crop_size]

# Load the Kodak dataset images in RGB and crop to 256x256
for i in range(24):
    if i + 1 < 10:
        image_path = f'./kodak/kodim0{i + 1}.png'
    else:
        image_path = f'./kodak/kodim{i + 1}.png'
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = center_crop(image, 256)  # Crop the center to 256x256
        kodak_images.append(image)
        print(f"Loaded and cropped image {image_path}")
    else:
        print(f"Failed to load image {image_path}")

# Check if kodak_images is not empty
if kodak_images:
    print("All images loaded and cropped successfully.")
else:
    print("No images loaded.")

# Display the first image
plt.imshow(kodak_images[0])
plt.axis('off')
plt.show()

# Normalize the pixel values to [0, 1]
kodak_images = np.array(kodak_images) / 255.0

# Create a directory to save the predictions if it doesn't exist
output_dir = './predictions'
os.makedirs(output_dir, exist_ok=True)

predictions = []
# Save the predictions
for i, image in enumerate(kodak_images):
    # Predict the compressed image
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    # Denormalize the image (convert back to [0, 255])
    prediction = (prediction * 255).astype(np.uint8)
    predictions.append(prediction)
    # Save the image
    output_path = os.path.join(output_dir, f'kodim{i + 1}_pred.png')
    save_tensor_as_png(prediction, output_path)
    print(f"Saved prediction to {output_path}")

# Calculate the PSNR and MS-SSIM for the images
psnr_values = []
ms_ssim_values = []
for i, image in enumerate(kodak_images):
    prediction = predictions[i]
    # Convert images to float32 and normalize to [0, 1]
    image_float32 = image.astype(np.float32) / 255.0
    prediction_float32 = prediction.astype(np.float32) / 255.0
    # Calculate PSNR
    psnr_value = tf.image.psnr(image_float32, prediction_float32, max_val=1.0).numpy()
    # Calculate MS-SSIM
    ms_ssim_value = tf.image.ssim_multiscale(image_float32, prediction_float32, max_val=1.0).numpy()
    psnr_values.append(psnr_value)
    ms_ssim_values.append(ms_ssim_value)
    print(f"Image {i + 1} - PSNR: {psnr_value:.2f}, MS-SSIM: {ms_ssim_value:.4f}")

# Calculate the average PSNR and MS-SSIM
avg_psnr = np.mean(psnr_values)
avg_ms_ssim = np.mean(ms_ssim_values)
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average MS-SSIM: {avg_ms_ssim:.4f}")

# Calculate bits per pixel (bpp) for the images
bpp_values = []
for i, image in enumerate(kodak_images):
    # Calculate the size of the compressed image
    compressed_size = os.path.getsize(os.path.join(output_dir, f'kodim{i + 1}_pred.png'))
    # Calculate the bpp
    bpp = (compressed_size * 8) / (256 * 256)
    bpp_values.append(bpp)
    print(f"Image {i + 1} - BPP: {bpp:.4f}")

# Calculate the average bpp
avg_bpp = np.mean(bpp_values)
print(f"Average BPP: {avg_bpp:.4f}")

# Display the original and predicted images
plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(5, 2, 2 * i + 1)
    plt.imshow(kodak_images[i])
    plt.title(f'Original Image {i + 1}')
    plt.axis('off')
    plt.subplot(5, 2, 2 * i + 2)
    plt.imshow(predictions[i])
    plt.title(f'Predicted Image {i + 1}')
    plt.axis('off')
plt.tight_layout()
plt.show()