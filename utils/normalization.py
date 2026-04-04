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
