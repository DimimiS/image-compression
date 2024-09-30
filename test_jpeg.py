import os
import subprocess
import numpy as np
from PIL import Image

def compress_image_jpeg(image_path, quality, output_dir):
    img = Image.open(image_path)
    compressed_path = os.path.join(output_dir, os.path.basename(image_path).replace('.png', '.jpg'))
    os.makedirs(os.path.dirname(compressed_path), exist_ok=True)
    img.save(compressed_path, quality=quality)
    return compressed_path

def calculate_bpp(original_path, compressed_path):
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    bpp = (compressed_size * 8) / (Image.open(original_path).size[0] * Image.open(original_path).size[1])
    return bpp

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def find_quality_for_specific_bpp(image_path, target_bpp, tolerance=0.01, output_dir='data/compressed'):
    low, high = 1, 100
    best_quality = None
    best_bpp = None
    best_psnr = None

    while low <= high:
        mid = (low + high) // 2
        compressed_path = compress_image_jpeg(image_path, mid, output_dir)
        bpp = calculate_bpp(image_path, compressed_path)
        img1 = np.array(Image.open(image_path))
        img2 = np.array(Image.open(compressed_path))
        psnr_value = psnr(img1, img2)

        if abs(bpp - target_bpp) < tolerance:
            best_quality = mid
            best_bpp = bpp
            best_psnr = psnr_value
            break
        elif bpp > target_bpp:
            high = mid - 1
        else:
            low = mid + 1

    return best_quality, best_bpp, best_psnr

def main():
    test_dir = 'data/test'
    output_dir = 'data/compressed'
    target_bpp = 0.95  # Specify the target BPP value
    tolerance = 0.01  # Tolerance for BPP matching

    # Get a list of all image files in the test directory
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("No images found in the test directory.")
        return

    for filename in image_files:
        original_path = os.path.join(test_dir, filename)
        quality, bpp, psnr_value = find_quality_for_specific_bpp(original_path, target_bpp, tolerance, output_dir)
        if quality is not None:
            print(f'File: {filename}, Target BPP: {target_bpp}, Quality: {quality}, BPP: {bpp:.2f}, PSNR: {psnr_value:.2f}')
        else:
            print(f'File: {filename} - Could not match target BPP {target_bpp} within tolerance.')

if __name__ == "__main__":
    main()