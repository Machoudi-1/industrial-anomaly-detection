import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

import torch.nn.functional as F


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


# Similarity search
def compute_patchwise_distances(
    test_patches: torch.Tensor,
    memory_bank: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the anomaly score for each test patch by finding its nearest neighbor
    in the memory bank.

    Args:
        test_patches (torch.Tensor): Tensor of shape (N_test, C)
        memory_bank (torch.Tensor): Tensor of shape (N_train, C)

    Returns:
        torch.Tensor: Minimum distances of shape (N_test,)
    """

    # Compute pairwise distances
    distances = torch.cdist(test_patches, memory_bank, p=2)

    # For each test patch, take the minimum distance
    min_distances = distances.min(dim=1).values

    return min_distances


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

    Args:
        memory_bank (torch.Tensor): Full memory bank (N, C)
        sampling_ratio (float): Final fraction to keep
        pre_sample_ratio (float): Fraction for initial random subset
        seed (int): Random seed

    Returns:
        torch.Tensor: Coreset (N_sampled, C)
    """
    if not (0 < sampling_ratio <= 1):
        raise ValueError("sampling_ratio must be in ]0,1]")
    if not (0 < pre_sample_ratio <= 1):
        raise ValueError("pre_sample_ratio must be in ]0,1]")

    N = memory_bank.size(0)

    # Step 1 — pre-sampling (to reduce cost)
    pre_sample_size = int(N * pre_sample_ratio)

    generator = torch.Generator(device=memory_bank.device)
    generator.manual_seed(seed)

    indices = torch.randperm(N, generator=generator)[:pre_sample_size]
    subset = memory_bank[indices]

    # Step 2 — greedy selection
    num_select = int(N * sampling_ratio)

    # Initialize
    selected_indices = []

    # pick random first point
    first_idx = torch.randint(0, subset.size(0), (1,), generator=generator).item()
    selected_indices.append(first_idx)

    selected = subset[first_idx].unsqueeze(0)

    # distances to selected set
    distances = torch.cdist(subset, selected).squeeze(1)

    for _ in range(num_select - 1):
        idx = torch.argmax(distances).item()
        selected_indices.append(idx)

        new_point = subset[idx].unsqueeze(0)

        # update distances (min distance to selected set)
        new_dist = torch.cdist(subset, new_point).squeeze(1)
        distances = torch.minimum(distances, new_dist)

    coreset = subset[selected_indices]
    return coreset
