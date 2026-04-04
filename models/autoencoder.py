# Import
import torch
from torch import nn


# Encoder
class CnnEncoder(nn.Module):
    """_summary_"""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


# Decoder
class CnnDecoder(nn.Module):
    """_summary_"""

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


# Full Autoender
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = CnnEncoder()
        self.decoder = CnnDecoder()

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
