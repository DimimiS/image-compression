import torch
import torch.nn.functional as F

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
    def __init__(self, lambda_bpp=0.01, lambda_distortion=1.0, lambda_entropy=1.0):
        super(BppDistortionLoss, self).__init__()
        self.lambda_bpp = lambda_bpp
        self.lambda_distortion = lambda_distortion
        self.lambda_entropy = lambda_entropy

    def forward(self, outputs, inputs, latent):
        # Calculate distortion (MSE)
        distortion = F.mse_loss(outputs, inputs, reduction='mean')

        # Calculate bits per pixel (bpp)
        bpp = calculate_bits_per_pixel(outputs, inputs.shape[-2:])

        # Calculate entropy of the latent space
        entropy = calculate_entropy(latent)

        # Combine bpp, distortion, and entropy
        loss = self.lambda_distortion * distortion + self.lambda_entropy * entropy
        return loss, bpp, distortion, entropy
