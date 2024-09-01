import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.autoencoder import Autoencoder
from models.vgg_perceptual_loss import VGGPerceptualLoss
from utils.dataset import ImageDataset
from utils.transforms import data_transforms
from pytorch_msssim import MS_SSIM
import time
from models.rd_loss import BppDistortionLoss

# Paths
train_dir = 'data/train/png'
val_dir = 'data/val/png'

# Datasets and Dataloaders
train_dataset = ImageDataset(train_dir, transform=data_transforms['train'])
val_dataset = ImageDataset(val_dir, transform=data_transforms['val'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Model, Loss, Optimizer
model = Autoencoder().to(device)
mse_loss = torch.nn.MSELoss()
perceptual_loss = VGGPerceptualLoss().to(device)
ms_ssim_loss = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)
bpp_distortion_loss = BppDistortionLoss(lambda_bpp=1.0, lambda_distortion=0.001).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(100):
    print('-' * 10)
    print(f'Epoch {epoch+1}')
    print('-' * 10)
    print('Training...')
    start_time = time.time()
    model.train()
    running_loss = 0.0
    for i, inputs in enumerate(train_loader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs, latent = model(inputs)
        loss, bpp, distortion, entropy = bpp_distortion_loss(outputs, inputs, latent)
        loss.backward()
        
        # Apply gradient clipping if needed
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 20 == 0:  # Print every 20 batches
            print(f'Batch {i+1}/{len(train_loader)}, Loss: {loss:.4f}, Bpp: {bpp:.4f}, MSE: {distortion:.4f}, Entropy: {entropy:.4f}')
    
    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')

    print('Validation loss calculation...')
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, inputs in enumerate(val_loader):
            inputs = inputs.to(device)
            outputs, latent = model(inputs)
            loss, bpp, distortion, entropy = bpp_distortion_loss(outputs, inputs, latent)
            val_loss += loss.item()
            if (i + 1) % 20 == 0:
                print(f'Validation Batch {i+1}/{len(val_loader)}, Loss: {loss:.4f}, Bpp: {bpp:.4f}, MSE: {distortion:.4f}, Entropy: {entropy:.4f}')
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch+1}, Average Validation Loss: {avg_val_loss:.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')

torch.save(model.state_dict(), 'model.pth')