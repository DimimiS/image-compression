# A file to automatically runs the commands python3 model.py --verbose compress file
# and python3 model.py --verbose decompress file for png in data/test
# and save printed output to results.txt

import os
import subprocess
import sys

# Get the current working directory
cwd = os.getcwd()

# Get the path to the data directory
data_dir = os.path.join(cwd, "data", "test")

# Get the path to the results file
results_file = os.path.join(cwd, "results.txt")

# Get the path to the model.py file
model_file = os.path.join(cwd, "model.py")

# Get the path to the png files in the data directory
png_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

# Open the results file
with open(results_file, "w") as f:
    # Loop through the png files
    for png_file in png_files:
        # Compress the png file
        compress_command = ["python3", model_file, "--verbose", "compress", os.path.join(data_dir, png_file)]
        compress_output = subprocess.run(compress_command, capture_output=True, text=True)
        f.write(compress_output.stdout)
        f.write("\n")
        print("Compressed", png_file)

        # Decompress the png file
        decompress_command = ["python3", model_file, "decompress", os.path.join(data_dir, png_file + ".tfci")]
        decompress_output = subprocess.run(decompress_command, capture_output=True, text=True)
        f.write(decompress_output.stdout)
        f.write("\n")
        print("Decompressed", png_file)
        # Delete the compressed file and the decompressed file
        os.remove(os.path.join(data_dir, png_file + ".tfci"))
        print("Deleted", png_file + ".tfci")
        os.remove(os.path.join(data_dir, png_file)+".tfci.png") 
        print("Deleted", png_file + ".tfci.png")

# Print the results file path
print(f"Results saved to {results_file}")

# Calculate the average values of each specific metric returned by the model.py file that are of the form:
# Mean squared error: 6.4104
# PSNR (dB): 40.06
# Multiscale SSIM: 0.9890
# Multiscale SSIM (dB): 19.57
# Bits per pixel: 0.3986

# Open the results file
with open(results_file, "r") as f:
    # Initialize the variables to store the sum of each metric
    mse_sum = 0
    psnr_sum = 0
    msssim_sum = 0
    msssim_db_sum = 0
    bpp_sum = 0
    # Initialize the variable to store the number of metrics
    num_metrics = 0
    # Loop through the lines in the results file
    for line in f:
        # Check if the line contains the mean squared error
        if "Mean squared error:" in line:
            # Get the value of the mean squared error
            mse = float(line.split(":")[1].strip())
            # Add the value to the sum
            mse_sum += mse
            # Increment the number of metrics
            num_metrics += 1
        # Check if the line contains the PSNR
        elif "PSNR (dB):" in line:
            # Get the value of the PSNR
            psnr = float(line.split(":")[1].strip())
            # Add the value to the sum
            psnr_sum += psnr
        # Check if the line contains the Multiscale SSIM
        elif "Multiscale SSIM:" in line:
            # Get the value of the Multiscale SSIM
            msssim = float(line.split(":")[1].strip())
            # Add the value to the sum
            msssim_sum += msssim
        # Check if the line contains the Multiscale SSIM (dB)
        elif "Multiscale SSIM (dB):" in line:
            # Get the value of the Multiscale SSIM (dB)
            msssim_db = float(line.split(":")[1].strip())
            # Add the value to the sum
            msssim_db_sum += msssim_db
        # Check if the line contains the Bits per pixel
        elif "Bits per pixel:" in line:
            # Get the value of the Bits per pixel
            bpp = float(line.split(":")[1].strip())
            # Add the value to the sum
            bpp_sum += bpp

    # Calculate the average values of each metric
    mse_avg = mse_sum / num_metrics
    psnr_avg = psnr_sum / num_metrics
    msssim_avg = msssim_sum / num_metrics
    msssim_db_avg = msssim_db_sum / num_metrics
    bpp_avg = bpp_sum / num_metrics

    # Save the average values to the results file
    with open(results_file, "a") as f:
        f.write(f"\nAverage Mean squared error: {mse_avg}\n")
        f.write(f"Average PSNR (dB): {psnr_avg}\n")
        f.write(f"Average Multiscale SSIM: {msssim_avg}\n")
        f.write(f"Average Multiscale SSIM (dB): {msssim_db_avg}\n")
        f.write(f"Average Bits per pixel: {bpp_avg}\n")
