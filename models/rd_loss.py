import torch
import torch.nn as nn
import torch.nn.functional as F

class BppDistortionLoss(nn.Module):
    def __init__(self, lambda_bpp=1.0, lambda_distortion=0.001):
        super(BppDistortionLoss, self).__init__()
        self.lambda_bpp = lambda_bpp
        self.lambda_distortion = lambda_distortion

    def forward(self, outputs, inputs):
        # Calculate distortion (MSE)
        distortion = F.mse_loss(outputs, inputs, reduction='mean')

        # Calculate bits per pixel (bpp)
        num_pixels = inputs.size(2) * inputs.size(3)

        # Calculate for each output image size in bits (as if they were stored) to calculate bpp
        compressed_size = outputs.numel() * 8
        bpp = compressed_size / (num_pixels*inputs.size(0))

        # Combine bpp and distortion
        loss = self.lambda_bpp * bpp + self.lambda_distortion * distortion
        return loss, bpp, distortion
