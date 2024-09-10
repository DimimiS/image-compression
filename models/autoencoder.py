import torch.nn as nn
from .gdn import GDN
from .sign import Binarizer
from .residual_block import ResidualBlock
from compressai.entropy_models import EntropyBottleneck
import torch

def calculate_entropy(latent):
    """
    Function that calculates the entropy for each image in the batch using histograms
    """
    entropy = 0
    for i in range(latent.size(0)):
        hist = torch.histc(latent[i], bins=256, min=latent[i].min().item(), max=latent[i].max().item())
        hist = hist / hist.sum()
        entropy += -torch.sum(hist * torch.log2(hist + 1e-6))
    return entropy / latent.size(0)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            GDN(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            GDN(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            GDN(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(256, inverse=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(128, inverse=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(64, inverse=True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResidualBlock(3),
            nn.Sigmoid()
        )
        self.binarizer = Binarizer()

    def forward(self, x):
        latent = self.encoder(x)

        binarized = self.binarizer(latent)
        reconstructed = self.decoder(binarized)
        entropy = calculate_entropy(reconstructed)

        return reconstructed, latent, entropy
