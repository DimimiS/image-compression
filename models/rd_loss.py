import torch
import torch.nn.functional as F

def calculate_bits_per_pixel(outputs):
    # Calculate bits per pixel (bpp)
    # We assume that the outputs are in the range [0, 1]
    # So, we quantize the values to 256 levels
    # Then, we calculate the entropy of the quantized values
    # Finally, we divide by the number of pixels to get the bits per pixel
    batch_size = outputs.size(0)

    # Calculate histogram
    hist = torch.histc(outputs.float(), bins=256, min=0, max=1)
    hist = hist / hist.sum()  # Normalize to get probability distribution

    # Calculate entropy
    epsilon = 1e-12
    hist = torch.clamp(hist, epsilon, 1)
    entropy = -torch.sum(hist * torch.log2(hist))

    # Calculate bits per pixel
    bpp = entropy/batch_size

    return bpp

class BppDistortionLoss(torch.nn.Module):
    def __init__(self, lambda_bpp=0.01, lambda_distortion=1.0):
        super(BppDistortionLoss, self).__init__()
        self.lambda_bpp = lambda_bpp
        self.lambda_distortion = lambda_distortion

    def forward(self, outputs, inputs):
        # Calculate distortion (MSE)
        distortion = F.mse_loss(outputs, inputs, reduction='mean')

        # Calculate bits per pixel (bpp)
        bpp = calculate_bits_per_pixel(outputs)

        # Combine bpp and distortion
        loss = bpp + self.lambda_distortion * distortion
        return loss, bpp, distortion
