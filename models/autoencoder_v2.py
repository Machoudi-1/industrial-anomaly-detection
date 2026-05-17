"""
autoencoder_v2.py
-----------------
Improved convolutional autoencoder (V2) for unsupervised anomaly detection.

Architecture:
    Encoder : 3 blocks (Conv -> Conv -> MaxPool) + bottleneck (Conv -> Conv)
              (B, 3, 256, 256) -> (B, 256, 32, 32)
    Decoder : 3 upsampling blocks (Upsample -> Conv -> Conv) + final Conv
              (B, 256, 32, 32) -> (B, 3, 256, 256)

Improvements over V1 (autoencoder.py):
    - Deeper encoder with double convolutions per block
    - MaxPool downsampling instead of strided convolutions (better feature preservation)
    - Bilinear upsampling in the decoder instead of ConvTranspose2d (smoother reconstructions)
    - Richer bottleneck at 256 channels

Key finding:
    Despite better reconstruction quality (lower MSE), AUROC only reached 0.613 on MVTec AD.
    Root cause: a well-trained model reconstructs anomalies accurately too,
    which is the fundamental ceiling of all reconstruction-based approaches.
    This motivated the switch to PatchCore (patchcore.py).

Usage:
    model = AutoEncoderV2()
    output = model(image_tensor)  # shape: (B, 3, 256, 256)
"""

import torch
from torch import nn


# Encoder
class EncoderV2(nn.Module):
    """Convolutional encoder used in the imporved autoencoder.

    The decoder progressively reconstructs the image by increasing
    the spatial resolution with upsampling followed by convolutions.

    The encoder progressively extracts richer features while reducing
    the spatial resolution of the input image.

    Input shape:
        (B, 3, 256, 256)

    Output shape:
        (B, 256, 32, 32)
    """

    def __init__(self) -> None:
        """
        Initialize the encoder blocks.
        """
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, 256, 256).

        Returns:
            torch.Tensor: Encoded tensor of shape (B, 256, 32, 32).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.bottleneck(x)
        return x


# Decoder
class DecoderV2(nn.Module):
    """
    Convolutional decoder used in the improved autoencoder.

    The decoder progressively reconstructs the image by increasing
    the spatial resolution with upsampling followed by convolutions.

    Input shape:
        (B, 256, 32, 32)

    Output shape:
        (B, 3, 256, 256)
    """

    def __init__(self) -> None:
        """
        Initialize the decoder blocks.
        """
        super().__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Encoded tensor of shape (B, 256, 32, 32).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, 3, 256, 256).
        """
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.final(x)
        return x


# Full Autoencoder
class AutoEncoderV2(nn.Module):
    """
    Improved convolutional autoencoder for image reconstruction.

    This version uses:
    - a deeper encoder
    - a richer bottleneck
    - upsampling + convolution in the decoder
    """

    def __init__(self) -> None:
        """
        Initialize the improved autoencoder.
        """
        super().__init__()
        self.encoder = EncoderV2()
        self.decoder = DecoderV2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, 256, 256).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, 3, 256, 256).
        """
        z = self.encoder(x)
        out = self.decoder(z)
        return out
