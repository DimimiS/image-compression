import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.autoencoder import Autoencoder
from models.vgg_perceptual_loss import VGGPerceptualLoss
from utils.dataset import ImageDataset
from utils.transforms import data_transforms
from pytorch_msssim import MS_SSIM
import time
import os

# Paths
train_dir = 'data/train/'
val_dir = 'data/validation/'

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
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(500):
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

        # Forward pass
        outputs, y_hat, entropy = model(inputs)

        # Calculate loss
        mse = mse_loss(outputs, inputs)
        bpp = entropy

        loss = 0.001 * mse + bpp

        loss.backward()

        # Update weights
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 20 == 0:  # Print every 20 batches
            print(f'Batch {i+1}/{len(train_loader)}, Loss: {loss:.4f}, BPP: {bpp:.4f}, MSE: {mse:.4f}')
    
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
            outputs, y_hat, entropy = model(inputs)
            
            # Calculate loss
            mse = mse_loss(outputs, inputs)
            bpp = entropy

            loss = 0.001 * mse + bpp
            val_loss += loss.item()
            if (i + 1) % 20 == 0:
                print(f'Validation Batch {i+1}/{len(val_loader)}, Loss: {loss:.4f}, BPP: {bpp:.4f}, MSE: {mse:.4f}')
    
    avg_val_loss = val_loss / len(val_loader)

    # Save checkpoint of epoch
    save_path = f'checkpoints/model_epoch{epoch+1}.pth'
    # Print save complete path
    print(f'Saving model checkpoint to {os.path.abspath(save_path)}')
    torch.save(model.state_dict(), save_path)
    
    # Delete previous checkpoint
    prev_checkpoint = f'checkpoints/model_epoch{epoch}.pth'
    if os.path.exists(prev_checkpoint):
        os.remove(prev_checkpoint)
    print(f'Epoch {epoch+1}, Average Validation Loss: {avg_val_loss:.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')

torch.save(model.state_dict(), 'model.pth')