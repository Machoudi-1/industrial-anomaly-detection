import logging
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from evaluation.metrics import (
    compute_patchcore_batch_scores,
    ReductionMethod,
)

LOGGER = logging.getLogger(__name__)


# Loss curve
def plot_loss(losses: List[float]) -> None:
    """
    Plot and save training loss curve.

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


# Autoencoder reconstruction
def visualize_reconstructions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_images: int = 5,
) -> None:
    """
    Visualize original vs reconstructed images.

    Args:
        model (nn.Module): Trained autoencoder
        dataloader (DataLoader): DataLoader
        device (torch.device): Device
        num_images (int): Number of images to display
    """
    model.eval()

    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    with torch.no_grad():
        outputs = model(images)

    images = images.cpu()
    outputs = outputs.cpu()

    num_images = min(num_images, images.size(0))

    plt.figure(figsize=(3 * num_images, 4))

    for i in range(num_images):
        # Original
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title("Original")
        plt.axis("off")

        # Reconstruction
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(outputs[i].permute(1, 2, 0))
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Autoencoder error maps
def visualize_reconstruction_error(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_images: int = 5,
) -> None:
    """
    Visualize original images, reconstructions, and error maps.

    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): DataLoader
        device (torch.device): Device
        num_images (int): Number of images
    """
    model.eval()

    batch = next(iter(dataloader))
    images = batch["image"].to(device)

    with torch.no_grad():
        outputs = model(images)

    images = images.cpu()
    outputs = outputs.cpu()

    error_maps = torch.abs(images - outputs).mean(dim=1)

    num_images = min(num_images, images.size(0))

    plt.figure(figsize=(3 * num_images, 8))

    for i in range(num_images):
        # Original
        plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title("Original")
        plt.axis("off")

        # Reconstruction
        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(outputs[i].permute(1, 2, 0))
        plt.title("Reconstructed")
        plt.axis("off")

        # Error map
        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(error_maps[i], cmap="hot")
        plt.title("Error map")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# PatchCore visualization
def visualize_patchcore_maps(
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    memory_bank: torch.Tensor,
    device: torch.device,
    num_images: int = 5,
    reduction: ReductionMethod = "max",
    k_ratio: float = 0.01,
) -> None:
    """
    Visualize PatchCore anomaly maps.

    Args:
        feature_extractor (nn.Module): Feature extractor
        dataloader (DataLoader): DataLoader
        memory_bank (torch.Tensor): Memory bank
        device (torch.device): Device
        num_images (int): Number of images
        reduction (ReductionMethod): Reduction strategy
        k_ratio (float): Top-k ratio
    """
    batch = next(iter(dataloader))
    images = batch["image"]

    anomaly_maps, image_scores = compute_patchcore_batch_scores(
        feature_extractor=feature_extractor,
        images=images,
        memory_bank=memory_bank,
        device=device,
        reduction=reduction,
        k_ratio=k_ratio,
    )

    images = images.cpu()
    anomaly_maps = anomaly_maps.cpu()
    image_scores = image_scores.cpu()

    num_images = min(num_images, images.size(0))

    plt.figure(figsize=(3 * num_images, 6))

    for i in range(num_images):
        # Input image
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.title("Input")
        plt.axis("off")

        # Anomaly map
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(anomaly_maps[i], cmap="hot")
        plt.title(f"Score={image_scores[i]:.3f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def upsample_anomaly_maps(
    anomaly_maps: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """
    Upsample anomaly maps to the target spatial resolution.

    Args:
        anomaly_maps (torch.Tensor): Tensor of shape (B, H, W).
        output_size (tuple[int, int]): Target size (H_out, W_out).

    Returns:
        torch.Tensor: Upsampled anomaly maps of shape (B, H_out, W_out).
    """
    anomaly_maps = anomaly_maps.unsqueeze(1)  # (B, 1, H, W)
    anomaly_maps = F.interpolate(
        anomaly_maps,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return anomaly_maps.squeeze(1)


def normalize_anomaly_maps(anomaly_maps: torch.Tensor) -> torch.Tensor:
    """
    Normalize each anomaly map independently to the range [0, 1].

    Args:
        anomaly_maps (torch.Tensor): Tensor of shape (B, H, W).

    Returns:
        torch.Tensor: Normalized anomaly maps of shape (B, H, W).
    """
    flat_maps = anomaly_maps.view(anomaly_maps.size(0), -1)
    min_vals = flat_maps.min(dim=1).values.view(-1, 1, 1)
    max_vals = flat_maps.max(dim=1).values.view(-1, 1, 1)

    normalized_maps = (anomaly_maps - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized_maps


def visualize_patchcore_overlay(
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    memory_bank: torch.Tensor,
    device: torch.device,
    num_images: int = 5,
    reduction: ReductionMethod = "max",
    k_ratio: float = 0.01,
    alpha: float = 0.5,
) -> None:
    """
    Visualize PatchCore anomaly maps as overlays on the input images.

    Args:
        feature_extractor (nn.Module): Feature extractor.
        dataloader (DataLoader): DataLoader.
        memory_bank (torch.Tensor): Memory bank.
        device (torch.device): Device.
        num_images (int, optional): Number of images to display.
        reduction (ReductionMethod, optional): Reduction strategy.
        k_ratio (float, optional): Top-k ratio.
        alpha (float, optional): Heatmap transparency.
    """
    batch = next(iter(dataloader))
    images = batch["image"]

    anomaly_maps, image_scores = compute_patchcore_batch_scores(
        feature_extractor=feature_extractor,
        images=images,
        memory_bank=memory_bank,
        device=device,
        reduction=reduction,
        k_ratio=k_ratio,
    )

    images = images.cpu()
    anomaly_maps = anomaly_maps.cpu()
    image_scores = image_scores.cpu()

    num_images = min(num_images, images.size(0))

    _, _, img_h, img_w = images.shape
    anomaly_maps = upsample_anomaly_maps(anomaly_maps, output_size=(img_h, img_w))
    anomaly_maps = normalize_anomaly_maps(anomaly_maps)

    plt.figure(figsize=(4 * num_images, 8))

    for i in range(num_images):
        img = images[i].permute(1, 2, 0)

        # Original
        plt.subplot(3, num_images, i + 1)
        plt.imshow(img)
        plt.title("Input")
        plt.axis("off")

        # Heatmap
        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(anomaly_maps[i], cmap="hot")
        plt.title(f"Map ({image_scores[i]:.3f})")
        plt.axis("off")

        # Overlay
        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(img)
        plt.imshow(anomaly_maps[i], cmap="jet", alpha=alpha)
        plt.title("Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
