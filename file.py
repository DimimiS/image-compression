import time
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from models.autoencoder import Autoencoder
from utils.dataset import ImageDataset
from utils.transforms import data_transforms
from models.rd_loss import BppDistortionLoss
from torch.autograd import Variable

# Paths
train_dir = 'data/train/png'
val_dir = 'data/val/png'

# Datasets and Dataloaders
train_dataset = ImageDataset(train_dir, transform=data_transforms['train'])
val_dataset = ImageDataset(val_dir, transform=data_transforms['val'])

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', '-N', type=int, default=16, help='batch size')
parser.add_argument('--max-epochs', '-e', type=int, default=1, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
args = parser.parse_args()

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

print('total images: {}; total batches: {}'.format(len(train_dataset), len(train_loader)))

autoencoder = Autoencoder().cuda()

solver = optim.Adam(autoencoder.parameters(), lr=args.lr)

def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'
    autoencoder.load_state_dict(torch.load(f'checkpoint/autoencoder_{s}_{epoch:08d}.pth'))

def save(index, epoch=True):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    s = 'epoch' if epoch else 'iter'
    torch.save(autoencoder.state_dict(), f'checkpoint/autoencoder_{s}_{index:08d}.pth')

scheduler = optim.lr_scheduler.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)
last_epoch = 0
if args.checkpoint:
    resume(args.checkpoint)
    last_epoch = args.checkpoint
    scheduler.last_epoch = last_epoch - 1

bpp_distortion_loss = BppDistortionLoss()

for epoch in range(last_epoch + 1, args.max_epochs + 1):
    scheduler.step()
    running_loss = 0.0
    start_time = time.time()

    autoencoder.train()
    for batch, data in enumerate(train_loader):
        batch_t0 = time.time()
        patches = Variable(data.cuda())
        solver.zero_grad()
        losses = []
        res = patches - 0.5
        bp_t0 = time.time()

        for _ in range(args.iterations):
            latent = autoencoder.encoder(res)
            codes = autoencoder.binarizer(latent)
            output = autoencoder.decoder(codes)
            res = res - output
            losses.append(res.abs().mean())

        bp_t1 = time.time()
        loss = sum(losses) / args.iterations
        loss.backward()
        solver.step()
        batch_t1 = time.time()
        running_loss += loss.item()

        print(f'[TRAIN] Epoch[{epoch}]({batch + 1}/{len(train_loader)}); Loss: {loss.item():.6f}; Backpropagation: {bp_t1 - bp_t0:.4f} sec; Batch: {batch_t1 - batch_t0:.4f} sec')
        print(('{:.4f} ' * args.iterations + '\n').format(*[l.item() for l in losses]))

        index = (epoch - 1) * len(train_loader) + batch
        if index % 500 == 0:
            save(0, False)

    avg_train_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch}, Average Training Loss: {avg_train_loss:.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')

    # Validation
    print('Validation loss calculation...')
    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, inputs in enumerate(val_loader):
            inputs = inputs.cuda()
            res = inputs - 0.5
            for _ in range(args.iterations):
                latent = autoencoder.encoder(res)
                codes = autoencoder.binarizer(latent)
                output = autoencoder.decoder(codes)
                res = res - output

            loss, bpp, mse = bpp_distortion_loss(inputs, res, training=False)
            val_loss += loss.item()
            if (i + 1) % 20 == 0:
                print(f'Validation Batch {i+1}/{len(val_loader)}, Loss: {loss:.4f}, BPP: {bpp:.4f}, MSE: {mse:.4f}')
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch}, Average Validation Loss: {avg_val_loss:.4f}')
    print(f'Time taken: {time.time() - start_time:.2f}s')

    save(epoch)