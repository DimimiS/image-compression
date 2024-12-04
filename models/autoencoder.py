from compressai.models import CompressionModel
from compressai.layers import GDN
from compressai.entropy_models import EntropyBottleneck
import torch.nn as nn

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.gdn1 = GDN(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.gdn2 = GDN(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.gdn3 = GDN(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.igdn1 = GDN(256, inverse=True)
        self.tconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.igdn2 = GDN(128, inverse=True)
        self.tconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.igdn3 = GDN(64, inverse=True)
        self.tconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tconv1(x)
        x = self.igdn1(x)
        x = self.tconv2(x)
        x = self.igdn2(x)
        x = self.tconv3(x)
        x = self.igdn3(x)
        x = self.tconv4(x)
        x = self.sigmoid(x)
        return x

# Define the Autoencoder
class Autoencoder(CompressionModel):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.entropy_bottleneck = EntropyBottleneck(512)

    def forward(self, x):
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods}}

    def compress(self, x):
        y = self.encoder(x)
        y_quantized = self.entropy_bottleneck.quantize(y, mode='noise')
        y_strings = self.entropy_bottleneck.compress(y_quantized)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        y_hat_dequantized = self.entropy_bottleneck.dequantize(y_hat)
        x_hat = self.decoder(y_hat_dequantized).clamp_(0, 1)
        return x_hat