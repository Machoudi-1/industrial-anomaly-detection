"""
train_autoencoder.py
--------------------
Training script for AutoEncoder V1 and V2 on MVTec AD.

Usage:
    # Train V2 Autoencoder on capsule with defaults
    python training/train_autoencoder.py --category capsule --version v2

    # Train V1 Autoencoder on bottle, 50 epochs
    python training/train_autoencoder.py --category bottle --version v1 --epochs 50

    # Full override
    python training/train_autoencoder.py \\
        --category screw --version v2 \\
        --epochs 30 --lr 1e-3 --batch-size 16
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.mvtec import MvtecAdDataset
from models.autoencoder import AutoEncoder
from models.autoencoder_v2 import AutoEncoderV2
from src.config import (
    ALL_CATEGORIES,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    EPOCHS,
    FIGURES_DIR,
    LEARNING_RATE,
    METRICS_DIR,
    SEED,
)
from utils.normalization import preprocess_image
from visualization.heatmap import plot_loss

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# Argument parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AutoEncoder V1 or V2 on a MVTec AD category."
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=ALL_CATEGORIES,
        help="MVTec category to train on.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Autoencoder version to train (default: v2).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE}).",
    )
    return parser.parse_args()


# Model factory


def build_model(version: str, device: torch.device) -> nn.Module:
    """Instantiate the autoencoder by version string."""
    if version == "v1":
        model = AutoEncoder()
    elif version == "v2":
        model = AutoEncoderV2()
    else:
        raise ValueError(f"Unknown version '{version}'. Choose 'v1' or 'v2'.")
    return model.to(device)


# Training loop


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Run one full training epoch.

    Returns:
        float: Mean loss over the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)

        reconstructed = model(images)
        loss = loss_fn(reconstructed, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            LOGGER.info(
                "  Batch %d/%d — loss: %.6f",
                batch_idx + 1,
                len(dataloader),
                loss.item(),
            )

    return running_loss / len(dataloader)


# Main


def main() -> None:
    args = parse_args()

    # Reproducibility
    torch.manual_seed(SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    # Run name — includes version for clear checkpoint naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.category}_autoencoder_{args.version}"
    LOGGER.info("Run: %s", run_name)

    # Output directories
    for d in [CHECKPOINT_DIR, FIGURES_DIR, METRICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_dataset = MvtecAdDataset(
        root_dir=str(DATA_DIR),
        category=args.category,
        split="train",
        transform=preprocess_image,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    LOGGER.info(
        "Category: %s | Version: %s | Train images: %d",
        args.category,
        args.version,
        len(train_dataset),
    )

    # Model
    model = build_model(args.version, device)
    LOGGER.info("Model: %s", model.__class__.__name__)

    # Optimizer & loss
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    loss_history: list[float] = []

    LOGGER.info("Starting training — %d epochs", args.epochs)
    for epoch in range(args.epochs):
        epoch_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        loss_history.append(epoch_loss)
        LOGGER.info("Epoch %d/%d — mean loss: %.6f", epoch + 1, args.epochs, epoch_loss)

    # Save checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    LOGGER.info("Checkpoint saved: %s", checkpoint_path)

    # Save loss curve
    plot_loss(loss_history)

    # Save training metadata
    metadata = {
        "run_name": run_name,
        "category": args.category,
        "version": args.version,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "final_loss": round(loss_history[-1], 6),
        "checkpoint": str(checkpoint_path),
    }
    meta_path = METRICS_DIR / f"{run_name}_train.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    LOGGER.info("Metadata saved: %s", meta_path)


if __name__ == "__main__":
    main()
