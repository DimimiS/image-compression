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
        decompress_command = ["python3", model_file, "--verbose", "decompress", os.path.join(data_dir, png_file + ".tfci")]
        decompress_output = subprocess.run(decompress_command, capture_output=True, text=True)
        f.write(decompress_output.stdout)
        f.write("\n")
        print("Decompressed", png_file)

# Print the results file path
print(f"Results saved to {results_file}")