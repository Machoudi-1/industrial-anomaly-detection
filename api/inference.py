import numpy as np
import torch
import torch.nn.functional as F
import time

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


from models.patchcore import (
    MultiScaleFeatureExtractor,
    extract_patch_embeddings,
    compute_patchwise_distances,
    patch_distances_to_maps,
)
from utils.normalization import preprocess_image
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Constants
READY_DIR = ROOT / "outputs" / "memory_banks" / "ready"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOPK_RATIO = 0.01

OPTIMAL_THRESHOLDS = {
    "bottle": 0.351,
    "cable": 0.402,
    "capsule": 0.296,
    "carpet": 0.347,
    "grid": 0.353,
    "hazelnut": 0.419,
    "leather": 0.338,
    "metal_nut": 0.385,
    "pill": 0.368,
    "screw": 0.349,
    "tile": 0.385,
    "toothbrush": 0.367,
    "transistor": 0.397,
    "wood": 0.383,
    "zipper": 0.327,
}


# Inference
def preprocess_pil(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    arr = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return preprocess_image(tensor).unsqueeze(0)


def run_inference(image_tensor, extractor, memory_bank):
    t0 = time.time()
    extractor.eval()
    with torch.no_grad():
        feature_map = extractor(image_tensor.to(DEVICE))
    B, C, H, W = feature_map.shape
    test_patches = extract_patch_embeddings(feature_map)
    test_patches = F.normalize(test_patches, dim=1)
    patch_distances = compute_patchwise_distances(
        test_patches=test_patches,
        memory_bank=memory_bank,
        chunk_size=1024,
    )
    anomaly_map = patch_distances_to_maps(patch_distances, B, H, W)
    flat = anomaly_map.view(1, -1)
    k = max(1, int(TOPK_RATIO * flat.shape[1]))
    raw_score = float(torch.topk(flat, k, dim=1).values.mean())
    score_norm = float(np.clip(raw_score / (raw_score + 1.0), 0, 1))
    anomaly_map_up = F.interpolate(
        anomaly_map.unsqueeze(1),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    a_min, a_max = anomaly_map_up.min(), anomaly_map_up.max()
    heatmap = ((anomaly_map_up - a_min) / (a_max - a_min + 1e-8)).cpu().numpy()
    return score_norm, heatmap, time.time() - t0


# Cached resources
def load_extractor() -> MultiScaleFeatureExtractor:
    extractor = MultiScaleFeatureExtractor().to(DEVICE)
    extractor.eval()
    return extractor


def load_memory_bank(category: str) -> tuple:
    ready_path = READY_DIR / f"{category}_ready.pt"
    if not ready_path.exists():
        return None, {}
    saved = torch.load(ready_path, map_location="cpu")
    memory_bank = saved["memory_bank"]
    meta = {
        "full_size": saved.get("full_size", "?"),
        "coreset_ratio": saved.get("coreset_ratio", "?"),
        "coreset_size": memory_bank.shape[0],
        "embedding_dim": memory_bank.shape[1],
        "coreset_method": saved.get("coreset_method", "greedy"),
    }
    return memory_bank, meta
