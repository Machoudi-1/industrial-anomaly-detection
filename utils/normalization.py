"""
normalization.py
----------------
Image preprocessing utilities shared across all models and scripts.

Provides a single entry point (preprocess_image) to ensure consistent
preprocessing throughout the project: autoencoder training, PatchCore
inference, and the Streamlit app.

Pipeline: Resize → float32 → normalize to [0, 1]

Usage:
    from utils.normalization import preprocess_image

    tensor = preprocess_image(raw_tensor)  # shape: (C, 256, 256), values in [0, 1]
"""

import torch
from torchvision import transforms

# Define resize once
_resize = transforms.Resize((256, 256))


def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    """Preprocess an input for the autoencoder model.
    Steps:
        - Resize image to (256, 256)
        - Convert to float
        - Normalize pixel values to [0,1]

    Args:
        image (torch.Tensor): INput image tensor of shape(C, H, W)
        with values in [0, 255]


    Returns:
        torch.Tensor: Preprocessed image tensor of shape (C, 256, 256)
        with values in [0,1]
    """
    # Resize
    image = _resize(image)

    # Convert to float
    image = image.float()

    return image / 255.0
