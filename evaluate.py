import torch
from torch.utils.data import DataLoader
from utils.dataset import ImageDataset
from utils.transforms import data_transforms
import numpy as np
from compressai.losses import RateDistortionLoss
from pytorch_msssim import MS_SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models.autoencoder import Encoder, Decoder, Autoencoder
import sys
import os

cwd = os.getcwd()
checkpoint = sys.argv[1] if len(sys.argv) > 1 else "01"
model_path = os.path.join(cwd, f"checkpoints/{checkpoint}")

# Load the saved model
encoder_path = model_path + '/encoder.pth'
entropy_bottleneck_path = model_path + '/entropy_bottleneck.pth'
decoder_path = model_path + '/decoder.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Autoencoder().to(device)
model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
model.entropy_bottleneck.load_state_dict(torch.load(entropy_bottleneck_path, map_location=device))
model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))

model.eval()

# Update the entropy bottleneck
model.entropy_bottleneck.update()

# Path to the test dataset
test_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/test/'

# Create the test dataset and dataloader
test_dataset = ImageDataset(test_dir, transform=data_transforms['test'])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define loss functions
criterion = RateDistortionLoss(lmbda=0.01 * int(checkpoint))
ms_ssim_loss = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)

test_loss = 0

psnr_values = []
ms_ssim_values = []
ssim_values = []
bpp_values = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)  # Move inputs to the same device as the model

        latent = model.encoder(inputs)
        latent_hat, likelihoods = model.entropy_bottleneck(latent)
        output = model(inputs)
        outputs = output["x_hat"]
        likelihoods = output["likelihoods"]

        # Calculate the loss
        loss = criterion({"x_hat": outputs, "likelihoods": likelihoods}, inputs)
        test_loss += loss["loss"].item()

        # Calculate the MS-SSIM loss
        ms_ssim_val = ms_ssim_loss(outputs, inputs)
        ssim_val = ssim(inputs[0].permute(1, 2, 0).cpu().numpy(), outputs[0].permute(1, 2, 0).cpu().numpy(), multichannel=True, win_size=3, data_range=1.0)
        psnr_val = psnr(inputs[0].permute(1, 2, 0).cpu().numpy(), outputs[0].permute(1, 2, 0).cpu().numpy(), data_range=1.0)
        bpp_val = loss["bpp_loss"].item()

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        ms_ssim_values.append(ms_ssim_val)
        bpp_values.append(bpp_val)

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Save the output images
import os
import torchvision
from PIL import Image

output_dir = model_path + '/output'
os.makedirs(output_dir, exist_ok=True)

model.eval()
with torch.no_grad():
    for i, (inputs) in enumerate(test_loader):
        inputs = inputs.to(device)  # Move inputs to the same device as the model
        latent = model.encoder(inputs)
        latent_hat, likelihoods = model.entropy_bottleneck(latent)
        outputs = model.decoder(latent_hat)

        for j in range(inputs.size(0)):
            torchvision.utils.save_image(inputs[j], os.path.join(output_dir, f'input_{i * inputs.size(0) + j}.png'))
            torchvision.utils.save_image(outputs[j], os.path.join(output_dir, f'output_{i * inputs.size(0) + j}.png'))
            
            if i == 0 and j == 0:
                # Ensure outputs[j] is a single image tensor
                if outputs[j].dim() == 3:
                    input_img = inputs[j].permute(1, 2, 0).cpu().numpy()
                    output_img = outputs[j].permute(1, 2, 0).cpu().numpy()  # Explicitly select the image tensor
                    
                    # Image.fromarray((input_img * 255).astype('uint8')).show()
                    Image.fromarray((output_img * 255).astype('uint8')).show()

# Output: Test Loss: 0.0162
# The above code snippet evaluates the trained model on the test dataset and saves the input and output images in the 'output' directory.
# It also calculates the test loss using Mean Squared Error (MSE) and Multi-Scale Structural Similarity Index (MS-SSIM) loss. 
# The test loss is printed at the end of the evaluation process.

# Save all the values from the bpp, PSNR, and SSIM arrays in a text file
with open(model_path + '/evaluation_results.txt', 'w') as f:
    for bpp_val, psnr_val, ms_ssim_val in zip(bpp_values, psnr_values, ms_ssim_values):
        f.write(f'{bpp_val:.4f}, {psnr_val:.2f}, {ms_ssim_val:.4f}\n')

avg_psnr = np.mean(psnr_values)
ms_ssim_values_cpu = [val.cpu().numpy() for val in ms_ssim_values]  # Move all tensors to CPU and convert to NumPy arrays
avg_ms_ssim = np.mean(ms_ssim_values_cpu)
avg_bpp = np.mean(bpp_values)
print(f'Average PSNR: {avg_psnr:.2f}')
print(f'Average MSSSIM: {avg_ms_ssim:.4f}')
print(f'Average bpp: {avg_bpp:.4f}')