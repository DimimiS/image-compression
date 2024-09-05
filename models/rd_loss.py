import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from compressai.entropy_models import EntropyBottleneck

def calculate_bits_per_pixel(outputs, image_size):
    """
    Calculate bits per pixel (bpp) for the output tensor.
    
    Args:
    - outputs (torch.Tensor): The output tensor from the model with shape (batch_size, channels, height, width).
    - image_size (tuple): The original image size (height, width).

    Returns:
    - bpp (float): The calculated bits per pixel.
    """
    batch_size, channels, height, width = outputs.shape
    
    # Quantize outputs to 256 levels (8-bit per pixel)
    quantized_outputs = torch.round(outputs * 255.0).clamp(0, 255).long()
    
    # Flatten the tensor to calculate the histogram over all pixels
    quantized_outputs = quantized_outputs.view(batch_size, -1)
    
    # Calculate histogram across the batch
    hist = torch.histc(quantized_outputs.float(), bins=256, min=0, max=255)
    
    # Normalize the histogram to get probability distribution
    hist = hist / hist.sum()
    
    # Calculate entropy (bits per symbol)
    epsilon = 1e-12
    hist = torch.clamp(hist, epsilon, 1)
    entropy = -torch.sum(hist * torch.log2(hist))
    
    # Calculate the number of pixels per image
    num_pixels = height * width
    
    # Calculate bpp (bits per pixel)
    bpp = (entropy * channels) / num_pixels

    return bpp


def calculate_entropy(latent):
    # Flatten the latent space tensor
    latent_flat = latent.view(latent.size(0), -1)

    # Calculate the histogram
    hist = torch.histc(latent_flat.float(), bins=256, min=latent_flat.min().item(), max=latent_flat.max().item())
    hist = hist / hist.sum()  # Normalize to get a probability distribution

    # Calculate entropy
    epsilon = 1e-12  # To avoid log(0)
    hist = torch.clamp(hist, epsilon, 1)
    entropy = -torch.sum(hist * torch.log2(hist))

    return entropy / latent_flat.size(0)  # Normalize by batch size

class BppDistortionLoss(torch.nn.Module):
    def __init__(self, lmbda=0.01):
        super(BppDistortionLoss, self).__init__()
        self.lmbda = lmbda
        self.entropy_bottleneck = EntropyBottleneck(3)

    def forward(self, x, x_hat, y_hat, training=True):
        # Compute the bits using entropy bottleneck
        y_hat, likelihoods = self.entropy_bottleneck(x_hat)

        # Calculate the number of pixels
        num_pixels = x.size(0) * x.size(2) * x.size(3)

        # Calculate bits per pixel (bpp)
        bpp = sum(torch.sum(-torch.log(likelihood)) for likelihood in likelihoods) / num_pixels

        # Mean squared error (MSE)
        mse = F.mse_loss(x, x_hat, reduction='mean')

        # Calculate MSSSIM loss
        ms_ssim_val = 1 - ms_ssim(x, x_hat, data_range=1.0)

        # Rate-distortion loss
        loss = bpp + self.lmbda * mse

        return loss, bpp, mse
