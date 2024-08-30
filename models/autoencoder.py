import torch
import torch.nn as nn
import torch.nn.functional as F
from .residual_block import ResidualBlock
from .gdn import GDN
from .sign import Binarizer

# Example usage in Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            GDN(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            GDN(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(128, inverse=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            GDN(64, inverse=True),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        self.binarizer = Binarizer()

    def forward(self, x):
        x = self.encoder(x)
        # x = self.binarizer(x)
        x = self.decoder(x)
        return x