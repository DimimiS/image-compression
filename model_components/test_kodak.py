import matplotlib.pyplot as plt
import keras
import numpy as np
import cv2
from model_components.model_metrics import psnr, ms_ssim
from model_components.custom_loss import RateDistortionLoss
import tensorflow as tf
import os

# Load the model
model = keras.models.load_model('/home/dimitra/Desktop/image-compression/cnn-gdn.keras', custom_objects={'psnr': psnr, 'ms_ssim': ms_ssim})
# Initialize an empty list to store the images
kodak_images = []

# Load the Kodak dataset images in RGB
for i in range(24):
    image_path = f'./kodak/kodim{i + 1}.png'
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (256, 256))  # Resize the image to 256x256
        kodak_images.append(image)
        print(f"Loaded image {image_path}")
    else:
        print(f"Failed to load image {image_path}")

# Check if kodak_images is not empty
if kodak_images:
    print("All images loaded successfully.")
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

# Save the predictions
for i, image in enumerate(kodak_images):
    # Predict the compressed image
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    # Denormalize the image (convert back to [0, 255])
    prediction = (prediction * 255).astype(np.uint8)
    # Save the image
    output_path = os.path.join(output_dir, f'kodim{i + 1}_pred.png')
    cv2.imwrite(output_path, prediction)
    print(f"Saved prediction to {output_path}")
