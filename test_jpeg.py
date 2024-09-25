import os
import subprocess
import random
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def calculate_bpp(image_path, compressed_path):
    original_img = Image.open(image_path)
    original_size = original_img.size[0] * original_img.size[1]
    compressed_size = os.path.getsize(compressed_path) * 8
    bpp = compressed_size / original_size
    return bpp

def compress_image_jpeg(image_path, quality):
    img = Image.open(image_path)
    compressed_path = image_path.replace('data/test', 'data/compressed').replace('.png', '.jpg')
    os.makedirs(os.path.dirname(compressed_path), exist_ok=True)
    img.save(compressed_path, quality=quality)
    return compressed_path

def compress_image_bpg(image_path, qp):
    compressed_path = image_path.replace('data/test', 'data/compressed').replace('.png', '.bpg')
    os.makedirs(os.path.dirname(compressed_path), exist_ok=True)
    subprocess.call(['bpgenc', '-q', str(qp), '-m', '9', '-f', '444', '-o', compressed_path, image_path])
    return compressed_path

def find_quality_for_bpp(image_path, max_bpp, tolerance=0.01, codec='jpeg'):
    low, high = 1, 100
    best_quality = None
    best_bpp = None
    best_psnr = None
    best_msssim = None

    while low <= high:
        mid = (low + high) // 2
        if codec == 'jpeg':
            compressed_path = compress_image_jpeg(image_path, mid)
        elif codec == 'bpg':
            compressed_path = compress_image_bpg(image_path, mid)
        else:
            raise ValueError("Unsupported codec")

        bpp = calculate_bpp(image_path, compressed_path)
        psnr_value = psnr(np.array(Image.open(image_path)), np.array(Image.open(compressed_path)))
        msssim_value = ssim(np.array(Image.open(image_path)), np.array(Image.open(compressed_path)), multichannel=True, win_size=3, channel_axis=-1)

        if abs(bpp - max_bpp) < tolerance:
            best_quality = mid
            best_bpp = bpp
            best_psnr = psnr_value
            best_msssim = msssim_value
            break
        elif bpp > max_bpp:
            high = mid - 1
        else:
            low = mid + 1

    return best_quality, best_bpp, best_psnr, best_msssim

def main():
    test_dir = 'data/test'
    max_bpp_values = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    tolerance = 0.01  # Tolerance for BPP matching
    codec = 'jpeg'  # Change to 'bpg' for BPG compression

    results = {bpp: {'psnr': [], 'msssim': []} for bpp in max_bpp_values}

    # Get a list of all image files in the test directory
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("No images found in the test directory.")
        return

    for max_bpp in max_bpp_values:
        # Randomly select one image for each bpp value
        filename = random.choice(image_files)
        original_path = os.path.join(test_dir, filename)

        quality, bpp, psnr_value, msssim_value = find_quality_for_bpp(original_path, max_bpp, tolerance, codec)
        if quality is not None:
            results[max_bpp]['psnr'].append(psnr_value)
            results[max_bpp]['msssim'].append(msssim_value)
            print(f'File: {filename}, BPP: {max_bpp}, Quality: {quality}, PSNR: {psnr_value:.2f}, MS-SSIM: {msssim_value:.4f}')
        else:
            print(f'File: {filename} - Could not match target BPP {max_bpp} within tolerance.')

    # Calculate and print average values
    for max_bpp in max_bpp_values:
        avg_psnr = np.mean(results[max_bpp]['psnr']) if results[max_bpp]['psnr'] else None
        avg_msssim = np.mean(results[max_bpp]['msssim']) if results[max_bpp]['msssim'] else None
        print(f'Average for BPP {max_bpp}: PSNR: {avg_psnr}, MS-SSIM: {avg_msssim}')

if __name__ == "__main__":
    main()