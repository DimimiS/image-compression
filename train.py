import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from compressai.losses import RateDistortionLoss
from utils.dataset import ImageDataset
from utils.transforms import data_transforms
import time
from models.autoencoder import Autoencoder
import os
import sys

cwd = os.getcwd()
checkpoint = sys.argv[1] if len(sys.argv) > 1 else "01"
model_path = os.path.join(cwd, f"checkpoints/{checkpoint}")


# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder().to(device)
parameters = set(p for n, p in model.named_parameters() if not n.endswith(".quantiles"))
aux_parameters = set(p for n, p in model.named_parameters() if n.endswith(".quantiles"))
optimizer = optim.Adam(parameters, lr=1e-4)
aux_optimizer = optim.Adam(aux_parameters, lr=1e-3)

criterion = RateDistortionLoss(lmbda=0.01 * int(checkpoint))

# Path to the training dataset
train_dir = 'data/train/'

# Create the training dataset and dataloader
train_dataset = ImageDataset(train_dir, transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_mse_loss = 0.0
    running_bpp_loss = 0.0
    start_time = time.time()
    
    for images in train_loader:
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)
        backward_loss = loss["loss"]
        backward_loss.backward()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        optimizer.step()
        
        running_loss += backward_loss.item()
        running_mse_loss += loss["mse_loss"].item()
        running_bpp_loss += loss["bpp_loss"].item()

    epoch_loss = running_loss / len(train_loader)
    epoch_mse_loss = running_mse_loss / len(train_loader)
    epoch_bpp_loss = running_bpp_loss / len(train_loader)
    epoch_time = time.time() - start_time
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Loss: {epoch_loss:.4f}, '
          f'MSE Loss: {epoch_mse_loss:.4f}, '
          f'BPP Loss: {epoch_bpp_loss:.4f}, '
          f'Time: {epoch_time:.2f}s')

    # Save the trained model
    torch.save(model.encoder.state_dict(), model_path + '/encoder.pth')
    torch.save(model.entropy_bottleneck.state_dict(), model_path + '/entropy_bottleneck.pth')
    torch.save(model.decoder.state_dict(), model_path + '/decoder.pth')