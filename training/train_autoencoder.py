# IMPORTS
import logging
import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.mvtec import MvtecAdDataset
from models.autoencoder import AutoEncoder
from src.config import DATA_DIR, CATEGORY, BATCH_SIZE, EPOCHS, LEARNING_RATE
from utils.normalization import preprocess_image
from visualization.heatmap import plot_loss

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s -%(message)s"
)


# DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info("Using %s", DEVICE)


# DATASET + DATALOADER
train_dataset = MvtecAdDataset(
    root_dir=DATA_DIR,
    category=CATEGORY,
    split="train",
    transform=preprocess_image,
)

LOGGER.info(
    "=== Training for: %s | Category:%s | number of image:%d ===",
    DATA_DIR,
    CATEGORY,
    len(train_dataset),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
# MODEL
model = AutoEncoder().to(DEVICE)

# OPTIMISATION
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# TRANING LOOP
def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Perform one epoch of traning for autoencoder

    Args:
        dataloader (DataLoader): Pytorch DataLoader providing traing batches
        model (nn.Module): Autoencoder model to train
        loss_fn (nn.Module): loss function
        optimizer (torch.optim.Optimizer): Optimizer used for training

    Returns:
        float: Average loss over the epoch
    """

    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Recover the image
        images = batch["image"]
        images = images.to(DEVICE)

        # Compute prediction and loss
        out = model(images)
        loss = loss_fn(images, out)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            LOGGER.info(
                "Batch %d/%d - loss: %.6f", batch_idx + 1, len(dataloader), loss.item()
            )

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def main() -> None:
    loss_history = []

    for epoch in range(EPOCHS):
        epoch_loss = train_loop(train_loader, model, loss_fn, optimizer)
        loss_history.append(epoch_loss)
        LOGGER.info("Epoch %d/%d - Mean Loss: %.6f", epoch + 1, EPOCHS, epoch_loss)

    os.makedirs("models/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "models/checkpoints/autoencoder.pth")
    LOGGER.info("Model saved to autoencoder.pth")

    plot_loss(loss_history)


# PLOTTING


if __name__ == "__main__":
    main()
