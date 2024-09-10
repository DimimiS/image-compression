import torch
from torch.utils.data import DataLoader
from models.autoencoder import Autoencoder
from utils.dataset import ImageDataset
from utils.transforms import data_transforms
from pytorch_msssim import MS_SSIM
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Paths
test_dir = 'data/test'
model_path = 'checkpoints/model_epoch3.pth'

# Datasets and Dataloaders
test_dataset = ImageDataset(test_dir, transform=data_transforms['test'])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluation
mse_loss = torch.nn.MSELoss()
ms_ssim_loss = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)

test_loss = 0

psnr_values = []
ssim_values = []
bpp_values = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Ensure outputs is a tensor
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = mse_loss(outputs, inputs) + (1 - ms_ssim_loss(outputs, inputs))
        test_loss += loss.item()

        for i, (input_img, output_img) in enumerate(zip(inputs, outputs)):
            input_img = input_img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            output_img = output_img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            ssim_val = ssim(input_img, output_img, multichannel=True, win_size=3, data_range=1.0) 
            psnr_val = psnr(input_img, output_img, data_range=1.0)
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            print(f'SSIM: {ssim_val}')

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
    for i, inputs in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        # Extract the relevant tensor from the tuple
        outputs = outputs[0]
        
        for j in range(inputs.size(0)):
            torchvision.utils.save_image(inputs[j], os.path.join(output_dir, f'input_{i * inputs.size(0) + j}.png'))
            torchvision.utils.save_image(outputs[j], os.path.join(output_dir, f'output_{i * inputs.size(0) + j}.png'))
            if i == 0 and j == 0:
                # Ensure outputs[j] is a single image tensor
                input_img = inputs[j].permute(1, 2, 0).cpu().numpy()
                output_img = outputs[j].permute(1, 2, 0).cpu().numpy()  # Explicitly select the image tensor
                
                Image.fromarray((input_img * 255).astype('uint8')).show()
                Image.fromarray((output_img * 255).astype('uint8')).show()

                # Calculate bpp from compression ratio of saved images
                input_size = os.path.getsize(os.path.join(output_dir, f'input_{i * inputs.size(0) + j}.png'))
                output_size = os.path.getsize(os.path.join(output_dir, f'output_{i * inputs.size(0) + j}.png'))
                bpp = (output_size * 8) / (inputs[j].size(1) * inputs[j].size(2))
                bpp_values.append(bpp)

# Output: Test Loss: 0.0162
# The above code snippet evaluates the trained model on the test dataset and saves the input and output images in the 'output' directory.
# It also calculates the test loss using Mean Squared Error (MSE) and Multi-Scale Structural Similarity Index (MS-SSIM) loss. 
# The test loss is printed at the end of the evaluation process.

avg_psnr = np.mean(psnr_values)
avg_ssim = np.mean(ssim_values)
avg_bpp = np.mean(bpp_values)
print(f'Average PSNR: {avg_psnr:.2f}')
print(f'Average SSIM: {avg_ssim:.4f}')
print(f'Average bpp: {avg_bpp:.4f}')
