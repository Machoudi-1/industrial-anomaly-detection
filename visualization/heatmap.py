import matplotlib.pyplot as plt
from typing import List
import torch
from datetime import datetime
from pathlib import Path

import logging

LOGGER = logging.getLogger(__name__)


def plot_loss(losses: List[float]) -> None:
    """
    Plot training loss curve.

    Args:
        losses (List[float]): List of loss values per epoch.
    """

    if not losses:
        LOGGER.warning("No losses to plot.")
        return
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker="o")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()

    save_path = output_dir / f"{now}_loss_curve.png"
    plt.savefig(save_path)
    plt.close()

    LOGGER.info("Loss curve saved to: %s", save_path)


def visualize_reconstructions(model, dataloader, device, num_images=5):
    """
    Visualize original vs reconstructed images.

    Args:
        model (nn.Module): Trained autoencoder
        dataloader (DataLoader): DataLoader for dataset
        device (torch.device): CPU or GPU
        num_images (int): Number of images to display
    """

    model.eval()

    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    with torch.no_grad():
        outputs = model(images)

    images = images.cpu()
    outputs = outputs.cpu()

    plt.figure(figsize=(10, 4))

    for i in range(num_images):
        # Original
        plt.subplot(2, num_images, i + 1)
        img = images[i].permute(1, 2, 0)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")

        # Reconstruction
        plt.subplot(2, num_images, i + 1 + num_images)
        recon = outputs[i].permute(1, 2, 0)
        plt.imshow(recon)
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Error map
def visualize_reconstruction_error(model, dataloader, device, num_images=5):
    """This function visualize original images, reconstructions, and
    reconstruction error maps.

    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Pytorch DataLoader
        device (torch.device): 'cpu' or 'cuda' or 'mps'
        num_images (int, optional):  Number of images to display.
        Defaults to 5.
    """
    model.eval()

    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    with torch.no_grad():
        outputs = model(images)

    # Move to CPU for visualization
    images = images.cpu()
    outputs = outputs.cpu()

    # Compute local pixel-wise absolute error
    error_maps = torch.abs(images - outputs)

    # Reduce channel dimension: (3, H, W) -> (H, W)
    error_maps = error_maps.mean(dim=1)

    plt.figure(figsize=(3 * num_images, 8))

    for i in range(num_images):
        # Original image
        plt.subplot(3, num_images, i + 1)
        img = images[i].permute(1, 2, 0)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")

        # Reconstructed image
        plt.subplot(3, num_images, i + 1 + num_images)
        recon = outputs[i].permute(1, 2, 0)
        plt.imshow(recon)
        plt.title("Reconstructed")
        plt.axis("off")

        # Erro map
        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        err = error_maps[i]
        plt.imshow(err, cmap="hot")
        plt.title("Error map")
        plt.axis("off")

    plt.tight_layout
    plt.show()
