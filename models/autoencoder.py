"""
autoencoder.py
--------------
Baseline convolutional autoencoder (V1) for unsupervised anomaly detection.

Architecture:
    Encoder: 3 convolutional layers (3 -> 32 -> 64 -> 128 channels), stride 2, ReLU
    Decoder: 3 transposed convolutional layers (128 -> 64 -> 32 -> 3 channels), stride 2, Sigmoid

Input:  RGB image tensor of shape (B, 3, H, W)
Output: Reconstructed image tensor of shape (B, 3, H, W)

Anomaly detection principle:
    The model is trained only on normal images. At inference, anomalous regions
    are poorly reconstructed, producing a high pixel-wise reconstruction error.

Note:
    This is the V1 baseline. Its fundamental limit is that a well-trained model
    also reconstructs anomalies correctly, capping AUROC around 0.61 on MVTec AD.
    See patchcore.py for the representation-based approach that overcomes this limit.

Usage:
    model = AutoEncoder()
    output = model(image_tensor)  # shape: (B, 3, H, W)
"""

# Import
import torch
from torch import nn


# Encoder
class CnnEncoder(nn.Module):
    """
    Convolutional encoder that compresses an RGB image into a latent feature map.

    Architecture (each Conv2d halves the spatial dimensions via stride=2):
        Conv2d(3  -> 32,  kernel=3, stride=2, padding=1) + ReLU
        Conv2d(32 -> 64,  kernel=3, stride=2, padding=1) + ReLU
        Conv2d(64 -> 128, kernel=3, stride=2, padding=1) + ReLU

    Args:
        x (torch.Tensor): Input tensor of shape (B, 3, H, W).

    Returns:
        torch.Tensor: Latent feature map of shape (B, 128, H/8, W/8).
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# Decoder
class CnnDecoder(nn.Module):
    """
    Convolutional decoder that reconstructs an RGB image from a latent feature map.

    Architecture (each ConvTranspose2d doubles the spatial dimensions via stride=2):
        ConvTranspose2d(128 -> 64, kernel=3, stride=2, padding=1) + ReLU
        ConvTranspose2d(64  -> 32, kernel=3, stride=2, padding=1) + ReLU
        ConvTranspose2d(32  -> 3,  kernel=3, stride=2, padding=1) + Sigmoid

    The final Sigmoid constrains output values to [0, 1], matching normalized inputs.

    Args:
        x (torch.Tensor): Latent feature map of shape (B, 128, H/8, W/8).

    Returns:
        torch.Tensor: Reconstructed image of shape (B, 3, H, W).
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# Full Autoender
class AutoEncoder(nn.Module):
    """
    Full convolutional autoencoder (Encoder + Decoder).

    Chains CnnEncoder and CnnDecoder into a single end-to-end model.
    The bottleneck (latent space) has shape (B, 128, H/8, W/8).

    Args:
        x (torch.Tensor): Input image tensor of shape (B, 3, H, W).

    Returns:
        torch.Tensor: Reconstructed image tensor of shape (B, 3, H, W).

    Example:
        >>> model = AutoEncoder()
        >>> out = model(torch.randn(4, 3, 256, 256))
        >>> out.shape
        torch.Size([4, 3, 256, 256])
    """

    def __init__(self):
        super().__init__()

        self.encoder = CnnEncoder()
        self.decoder = CnnDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)  # compress to latent space
        out = self.decoder(z)  # reconstruct from latent space
        return out
