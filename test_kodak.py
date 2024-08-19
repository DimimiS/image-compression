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

# Display the first predictions vs original images
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(kodak_images[0])
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Predicted Image')
plt.imshow(predictions[0])
plt.axis('off')
plt.show()
