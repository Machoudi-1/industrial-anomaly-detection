from typing import Literal, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


ReductionMethod = Literal["mean", "max", "topk_mean"]


def reduce_error_map(
    errors: torch.Tensor,
    method: ReductionMethod = "mean",
    k_ratio: float = 0.01,
) -> torch.Tensor:
    """
    Reduce a batch of reconstruction error maps into one anomaly score per image.

    Args:
        errors (torch.Tensor): Reconstruction errors of shape (B, C, H, W).
        method (ReductionMethod, optional): Reduction strategy.
            Supported values are:
            - "mean": mean error over all pixels
            - "max": maximum error over all pixels
            - "topk_mean": mean of the top-k largest errors
            Defaults to "mean".
        k_ratio (float, optional): Fraction of pixels to keep when using "topk_mean".
            Must be in ]0, 1]. Defaults to 0.01.

    Returns:
        torch.Tensor: Anomaly scores of shape (B,).
    """
    if method not in {"mean", "max", "topk_mean"}:
        raise ValueError("method must be one of {'mean', 'max', 'topk_mean'}")

    if not (0 < k_ratio <= 1):
        raise ValueError("k_ratio must be in the interval ]0, 1].")

    batch_size = errors.size(0)
    flat_errors = errors.view(batch_size, -1)

    if method == "mean":
        scores = flat_errors.mean(dim=1)

    elif method == "max":
        scores = flat_errors.max(dim=1).values

    elif method == "topk_mean":
        k = max(1, int(k_ratio * flat_errors.size(1)))
        topk_values = torch.topk(flat_errors, k, dim=1).values
        scores = topk_values.mean(dim=1)

    return scores


def compute_anomaly_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    method: ReductionMethod = "mean",
    k_ratio: float = 0.01,
) -> Tuple[list[float], list[int]]:
    """
    Compute anomaly scores for each image in a dataloader.

    Args:
        model (nn.Module): Trained autoencoder model.
        dataloader (DataLoader): DataLoader providing image batches.
        device (torch.device): Device used for inference.
        method (ReductionMethod, optional): Reduction strategy used to compute
            one score per image. Defaults to "mean".
        k_ratio (float, optional): Fraction of pixels to keep when using
            "topk_mean". Defaults to 0.01.

    Returns:
        Tuple[list[float], list[int]]:
            - scores: anomaly scores for each image
            - labels: corresponding labels (0 = normal, 1 = anomaly)
    """
    model.eval()

    scores: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            batch_labels = batch["label"]

            outputs = model(images)

            # Pixel-wise absolute reconstruction error
            errors = torch.abs(images - outputs)

            # Reduce the error map into one score per image
            batch_scores = reduce_error_map(
                errors=errors,
                method=method,
                k_ratio=k_ratio,
            )

            scores.extend(batch_scores.cpu().tolist())
            labels.extend(batch_labels.tolist())

    return scores, labels
