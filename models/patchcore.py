"""
patchcore.py
------------
PatchCore anomaly detection pipeline.

Components:
    - FeatureExtractor         : single-scale ResNet18 features (layer3)
    - MultiScaleFeatureExtractor: fused layer2 + layer3 features
    - extract_patch_embeddings : feature map -> patch matrix
    - build_memory_bank        : build normal patch reference from train set
    - compute_patchwise_distances: nearest-neighbor distances (chunked, OOM-safe)
    - patch_distances_to_maps  : flat distances -> spatial anomaly maps
    - random_coreset_sampling  : random subset of the memory bank
    - greedy_coreset_sampling  : k-center coreset (better coverage)
"""

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

import torch.nn.functional as F


# Feature extractors
class FeatureExtractor(nn.Module):
    """
    Feature extractor based on a pretrained ResNet18 backbone.

    This model is used to extract intermediate convolutional
    feature maps for PatchCore anomaly detection.

    Input shape:
        (B, 3, H, W)
    Output shape:
        (B, C, H', W')
    """

    def __init__(self) -> None:
        """Initialize the pretrained ResNet18 backbone and keep
        only the layers required for feature extraction.
        """
        super().__init__()

        # 1. Load pretrained ResNet18
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 2. Keep only the convolutional feature extractor part
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # 3. Freeze backbone parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate feature maps from the input image.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Extract feature map of shape (B, C, H', W')
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extractor based on a pretrained ResNet18 backbone.

    This module combines intermediate features from layer2 and layer3
    to improve local anomaly detection.

    Input shape:
        (B, 3, H, W)

    Output shape:
        (B, C_combined, H2, W2)
    """

    def __init__(self) -> None:
        """Initialize the pretrained ResNet18 backbone and keep
        only the layers required for feature extraction.
        """
        super().__init__()

        # 1. Load pretrained ResNet18
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 2. Keep only the convolutional feature extractor part
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # 3. Freeze backbone parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate feature maps from the input image.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: Multi-scale feature map of shape (B, C_combined, H2, W2).
        """
        x = self.stem(x)
        x = self.layer1(x)
        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)
        feat3_up = F.interpolate(
            feat3, size=feat2.shape[-2:], mode="bilinear", align_corners=False
        )

        # Concatenate
        embedding = torch.cat([feat2, feat3_up], dim=1)

        return embedding


# Extract pactch embeddings
def extract_patch_embeddings(feature_map: torch.Tensor) -> torch.Tensor:
    """
    Convert a feature map of shape (B, C, H, W) into a matrix of patch embeddings
    of shape (B * H * W, C).

    Args:
        feature_map (torch.Tensor): Feature map tensor of shape (B, C, H, W).

    Returns:
        torch.Tensor: Patch embedding matrix of shape (B * H * W, C).
    """
    B, C, H, W = feature_map.shape
    patches = feature_map.permute(0, 2, 3, 1).reshape(B * H * W, C)
    return patches


# Memory bank


def build_memory_bank(
    feature_extractor: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a memory bank of normal patch embeddings from the training set.

    Each training image is passed through the feature extractor, then its
    feature map is converted into a set of patch embeddings. All patch
    embeddings are concatenated into a single memory bank.

    Args:
        feature_extractor (nn.Module): Pretrained feature extractor.
        dataloader (DataLoader): DataLoader containing normal training images.
        device (torch.device): Device used for feature extraction.

    Returns:
        torch.Tensor: Memory bank of shape (N_patches, C), where:
            - N_patches is the total number of extracted patches
            - C is the embedding dimension
    """
    feature_extractor.eval()

    all_patches: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)

            feature_map = feature_extractor(images)
            patch_embeddings = extract_patch_embeddings(feature_map)

            all_patches.append(patch_embeddings.cpu())

    memory_bank = torch.cat(all_patches, dim=0)
    return memory_bank


# Nearest-neighbor distances (chunked)
def compute_patchwise_distances(
    test_patches: torch.Tensor,
    memory_bank: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Compute the nearest-neighbor distance from each test patch
    to the memory bank, using chunked processing to avoid OOM.

    Why chunked?
        A naive torch.cdist(test, memory_bank) allocates a matrix of
        shape (N_test, N_train). With N_test = N_train = 200k patches,
        this requires ~26 GB of VRAM — far beyond a T4 GPU (15 GB).

        Instead, we split test_patches into chunks of `chunk_size` rows,
        compute distances for each chunk independently, take the minimum
        distance to the memory bank per patch, then free the intermediate
        tensor. Peak VRAM usage is proportional to chunk_size × N_train,
        not N_test × N_train.

    Args:
        test_patches (torch.Tensor): Shape (N_test, C). Can be on CPU or GPU.
        memory_bank (torch.Tensor): Shape (N_train, C). Moved to GPU internally.
        chunk_size (int): Number of test patches processed per chunk.
            Lower values use less VRAM. Defaults to 1024.

    Returns:
        torch.Tensor: Minimum distances of shape (N_test,), on CPU.
    """
    # Move memory bank to GPU once — stays there for all chunks
    memory_bank_gpu = memory_bank.cuda() if torch.cuda.is_available() else memory_bank

    min_distances: list[torch.Tensor] = []

    for start in range(0, test_patches.shape[0], chunk_size):
        chunk = test_patches[start : start + chunk_size]

        # Move chunk to same device as memory bank
        chunk = chunk.to(memory_bank_gpu.device)

        # (chunk_size, N_train) distance matrix
        dists = torch.cdist(chunk, memory_bank_gpu, p=2)

        # Keep only the minimum distance per test patch
        min_dists = dists.min(dim=1).values

        min_distances.append(min_dists.cpu())

        # Free intermediate tensors immediately
        del chunk, dists

    return torch.cat(min_distances, dim=0)


#  Anomaly maps


def patch_distances_to_maps(
    patch_distances: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Reshape a flat vector of patch distances into anomaly maps.

    Args:
        patch_distances (torch.Tensor): Tensor of shape (B * H * W,)
        batch_size (int): Batch size B
        height (int): Spatial height H of the feature map
        width (int): Spatial width W of the feature map

    Returns:
        torch.Tensor: Anomaly maps of shape (B, H, W)
    """
    return patch_distances.view(batch_size, height, width)


#  Coreset sampling
def random_coreset_sampling(
    memory_bank: torch.Tensor,
    sampling_ratio: float = 0.1,
    seed: int = 42,
) -> torch.Tensor:
    """
    Randomly sample a subset of patch embeddings from the memory bank.

    Args:
        memory_bank (torch.Tensor): Full memory bank of shape (N, C).
        sampling_ratio (float, optional): Fraction of embeddings to keep.
            Must be in ]0, 1]. Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        torch.Tensor: Sampled memory bank of shape (N_sampled, C).
    """
    if not (0 < sampling_ratio <= 1):
        raise ValueError("sampling_ratio must be in the interval ]0, 1].")

    num_embeddings = memory_bank.size(0)
    num_sampled = max(1, int(num_embeddings * sampling_ratio))

    generator = torch.Generator(device=memory_bank.device)
    generator.manual_seed(seed)

    indices = torch.randperm(num_embeddings, generator=generator)[:num_sampled]
    return memory_bank[indices]


def greedy_coreset_sampling(
    memory_bank: torch.Tensor,
    sampling_ratio: float = 0.1,
    pre_sample_ratio: float = 0.2,
    seed: int = 42,
) -> torch.Tensor:
    """
    Greedy coreset sampling using a k-center approximation.

    Produces a more representative subset than random sampling by
    iteratively selecting the patch that is farthest from the
    already-selected set (max-min criterion).

    A pre-sampling step first draws a random subset to make the
    greedy search tractable on large memory banks.

    Args:
        memory_bank (torch.Tensor): Full memory bank of shape (N, C).
        sampling_ratio (float): Final fraction to keep, in ]0, 1].
        pre_sample_ratio (float): Fraction for the initial random pre-sample.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: Coreset of shape (N_sampled, C).
    """
    if not (0 < sampling_ratio <= 1):
        raise ValueError("sampling_ratio must be in ]0, 1].")
    if not (0 < pre_sample_ratio <= 1):
        raise ValueError("pre_sample_ratio must be in ]0, 1].")

    N = memory_bank.size(0)

    generator = torch.Generator(device=memory_bank.device)
    generator.manual_seed(seed)

    # Step 1 — random pre-sample to reduce search space
    pre_size = int(N * pre_sample_ratio)
    indices = torch.randperm(N, generator=generator)[:pre_size]
    subset = memory_bank[indices]

    # Step 2 — greedy k-center selection
    n_select = int(N * sampling_ratio)
    selected: list[int] = []

    first_idx = torch.randint(0, subset.size(0), (1,), generator=generator).item()
    selected.append(first_idx)

    # Distance from each point to the nearest selected point
    distances = torch.cdist(subset, subset[first_idx].unsqueeze(0)).squeeze(1)

    for _ in range(n_select - 1):
        idx = torch.argmax(distances).item()
        selected.append(idx)

        new_dists = torch.cdist(subset, subset[idx].unsqueeze(0)).squeeze(1)
        distances = torch.minimum(distances, new_dists)

    return subset[selected]
