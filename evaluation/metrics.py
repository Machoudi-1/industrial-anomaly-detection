"""
metrics.py
----------
Anomaly scoring utilities for autoencoder and PatchCore models.

Provides:
    - reduce_error_map       : reduce a spatial error map to one score per image
    - compute_anomaly_scores : autoencoder inference loop over a full dataloader
    - compute_patchcore_batch_scores : PatchCore scores for a single batch
    - compute_patchcore_scores       : PatchCore inference loop over a full dataloader

Supported reduction strategies: "mean", "max", "topk_mean".
"""

from typing import Literal, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.patchcore import (
    compute_patchwise_distances,
    extract_patch_embeddings,
    patch_distances_to_maps,
)

ReductionMethod = Literal["mean", "max", "topk_mean"]


def reduce_error_map(
    errors: torch.Tensor,
    method: ReductionMethod = "mean",
    k_ratio: float = 0.01,
) -> torch.Tensor:
    """
    Reduce a batch of error maps into one anomaly score per image.

    Args:
        errors (torch.Tensor): Error maps of shape (B, ...) where B is the batch size.
        method (ReductionMethod, optional): Reduction strategy.
            Supported values are:
            - "mean": mean error over all values
            - "max": maximum error over all values
            - "topk_mean": mean of the top-k largest values
            Defaults to "mean".
        k_ratio (float, optional): Fraction of values to keep when using "topk_mean".
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

    else:  # topk_mean
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
    Compute image-level anomaly scores for an autoencoder on a full dataloader.

    Args:
        model (nn.Module): Trained autoencoder model.
        dataloader (DataLoader): DataLoader providing image batches.
        device (torch.device): Device used for inference.
        method (ReductionMethod, optional): Reduction strategy used to compute
            one score per image. Defaults to "mean".
        k_ratio (float, optional): Fraction of values to keep when using
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


def compute_patchcore_batch_scores(
    feature_extractor: nn.Module,
    images: torch.Tensor,
    memory_bank: torch.Tensor,
    device: torch.device,
    reduction: ReductionMethod = "max",
    k_ratio: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute PatchCore anomaly maps and image-level scores for a batch of images.

    Args:
        feature_extractor (nn.Module): Pretrained feature extractor.
        images (torch.Tensor): Input batch of shape (B, 3, H, W).
        memory_bank (torch.Tensor): Memory bank of shape (N_train, C).
        device (torch.device): Computation device.
        reduction (ReductionMethod, optional): Reduction strategy for image score.
            Defaults to "max".
        k_ratio (float, optional): Fraction used for top-k mean. Defaults to 0.01.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - anomaly_maps: Tensor of shape (B, Hf, Wf)
            - image_scores: Tensor of shape (B,)
    """
    feature_extractor.eval()

    with torch.no_grad():
        images = images.to(device)
        feature_map = feature_extractor(images)

    batch_size, _, height, width = feature_map.shape

    test_patches = extract_patch_embeddings(feature_map)
    patch_distances = compute_patchwise_distances(
        test_patches=test_patches,
        memory_bank=memory_bank.to(test_patches.device),
    )

    anomaly_maps = patch_distances_to_maps(
        patch_distances=patch_distances,
        batch_size=batch_size,
        height=height,
        width=width,
    )

    image_scores = reduce_error_map(
        errors=anomaly_maps,
        method=reduction,
        k_ratio=k_ratio,
    )

    return anomaly_maps, image_scores


def compute_patchcore_scores(
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    memory_bank: torch.Tensor,
    device: torch.device,
    reduction: ReductionMethod = "max",
    k_ratio: float = 0.01,
) -> Tuple[list[float], list[int]]:
    """
    Compute PatchCore image-level anomaly scores for all images in a dataloader.

    Args:
        feature_extractor (nn.Module): Pretrained feature extractor.
        dataloader (DataLoader): DataLoader providing image batches.
        memory_bank (torch.Tensor): Memory bank of normal patch embeddings.
        device (torch.device): Device used for inference.
        reduction (ReductionMethod, optional): Reduction strategy for image score.
            Defaults to "max".
        k_ratio (float, optional): Fraction used for top-k mean. Defaults to 0.01.

    Returns:
        Tuple[list[float], list[int]]:
            - scores: image-level anomaly scores
            - labels: corresponding labels (0 = normal, 1 = anomaly)
    """
    feature_extractor.eval()

    scores: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"]
            batch_labels = batch["label"]

            _, image_scores = compute_patchcore_batch_scores(
                feature_extractor=feature_extractor,
                images=images,
                memory_bank=memory_bank,
                device=device,
                reduction=reduction,
                k_ratio=k_ratio,
            )

            scores.extend(image_scores.cpu().tolist())
            labels.extend(batch_labels.tolist())

    return scores, labels
