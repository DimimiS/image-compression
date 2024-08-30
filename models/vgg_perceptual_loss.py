import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg)[:9]).eval()  # Use up to relu_2_2
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x, target):
        x_features = self.layers(x)
        target_features = self.layers(target)
        return nn.functional.mse_loss(x_features, target_features)