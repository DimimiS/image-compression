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
train_dir = 'data/train'
val_dir = 'data/val'

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
bpp_distortion_loss = BppDistortionLoss(lambda_bpp=0.01, lambda_distortion=1.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(100):
    print('-' * 10)
    print(f'Epoch {epoch+1}')
    print('-' * 10)
    print('Training...')
    start_time = time.time()
    model.train()
    for inputs in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = bpp_distortion_loss(outputs, inputs)[0]
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')

    print('Validation loss calculation...')
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = bpp_distortion_loss(outputs, inputs)[0]
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

torch.save(model.state_dict(), 'model.pth')