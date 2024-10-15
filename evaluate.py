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
from models.sign import Binarizer

# Load the saved model
encoder_path = 'checkpoints/00125/encoder.pth'
entropy_bottleneck_path = 'checkpoints/00125/entropy_bottleneck.pth'
decoder_path = 'checkpoints/00125/decoder.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Autoencoder().to(device)
model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
model.entropy_bottleneck.load_state_dict(torch.load(entropy_bottleneck_path, map_location=device))
model.decoder.load_state_dict(torch.load(decoder_path, map_location=device))

model.eval()

# Update the entropy bottleneck
model.entropy_bottleneck.update()

# Path to the test dataset
test_dir = 'data/test/'

# Create the test dataset and dataloader
test_dataset = ImageDataset(test_dir, transform=data_transforms['test'])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define loss functions
criterion = RateDistortionLoss(lmbda=0.01)
ms_ssim_loss = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)

test_loss = 0

psnr_values = []
ssim_values = []
bpp_values = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)  # Move inputs to the same device as the model
        latent = model.compress(inputs)
        outputs = model.decompress(latent["strings"], latent["shape"])


        for i, (input_img, output_img, compressed_string) in enumerate(zip(inputs, outputs, latent["strings"])):
            input_img = input_img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            output_img = output_img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format

            num_pixels = input_img.shape[0] * input_img.shape[1]
            bpp = sum(len(c) * 8 / num_pixels for c in compressed_string)

            ssim_val = ssim(input_img, output_img, multichannel=True, win_size=3, data_range=1.0)
            psnr_val = psnr(input_img, output_img, data_range=1.0)

            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            bpp_values.append(bpp/len(compressed_string))

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Save the output images
import os
import torchvision
from PIL import Image

output_dir = 'output'
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
                    
                    Image.fromarray((input_img * 255).astype('uint8')).show()
                    Image.fromarray((output_img * 255).astype('uint8')).show()

# Output: Test Loss: 0.0162
# The above code snippet evaluates the trained model on the test dataset and saves the input and output images in the 'output' directory.
# It also calculates the test loss using Mean Squared Error (MSE) and Multi-Scale Structural Similarity Index (MS-SSIM) loss. 
# The test loss is printed at the end of the evaluation process.

# Save all the values from the bpp, PSNR, and SSIM arrays in a text file
with open('evaluation_results.txt', 'w') as f:
    for bpp_val, psnr_val, ssim_val in zip(bpp_values, psnr_values, ssim_values):
        f.write(f'{bpp_val:.4f}, {psnr_val:.2f}, {ssim_val:.4f}\n')

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)
avg_bpp = np.mean(bpp_values)
print(f'Average PSNR: {avg_psnr:.2f}')
print(f'Average SSIM: {avg_ssim:.4f}')
print(f'Average bpp: {avg_bpp:.4f}')