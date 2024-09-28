import os
from PIL import Image

def move_images_to_subdirectories(image_dir, final_dir):
    for root, _, files in os.walk(image_dir):
        for file in files:
            image_path = os.path.join(root, file)
            new_path = os.path.join(final_dir, file)
            os.rename(image_path, new_path)
            print(f"Moved {image_path} to {new_path}")

# Example usage
for directories in os.listdir('data/train'):
    move_images_to_subdirectories('data/train/' + directories, 'data/train')
    print(directories)
for directories in os.listdir('data/validation'):
    move_images_to_subdirectories('data/validation/' + directories, 'data/validation')
