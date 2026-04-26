"""
app.py
------
Streamlit application for PatchCore industrial anomaly detection.

Improvements over v1:
- Greedy coreset sampling (better memory bank coverage than random)
- L2 normalization of embeddings (improves distance discrimination)
- Per-category optimal thresholds calibrated from benchmark scores
- Multi-image upload support
- Cleaner UI with inference timing

Usage:
    poetry run streamlit run app/app.py
"""

import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from models.patchcore import (
    MultiScaleFeatureExtractor,
    extract_patch_embeddings,
    compute_patchwise_distances,
    patch_distances_to_maps,
    greedy_coreset_sampling,
)
from utils.normalization import preprocess_image

# ── Constants ─────────────────────────────────────────────────
MEMORY_BANK_DIR = ROOT / "outputs" / "memory_banks"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Benchmark AUROC (topk_mean) from 04_multicategory_benchmark
AUROC_REFERENCE = {
    "bottle":     0.9992,
    "cable":      0.8564,
    "capsule":    0.8783,
    "carpet":     0.8941,
    "grid":       0.5773,
    "hazelnut":   0.9996,
    "leather":    0.9253,
    "metal_nut":  0.9462,
    "pill":       0.7621,
    "screw":      0.8965,
    "tile":       0.8413,
    "toothbrush": 0.9611,
    "transistor": 0.8592,
    "wood":       0.9833,
    "zipper":     0.9309,
}

# Per-category thresholds calibrated from benchmark observations.
# Higher AUROC -> tighter threshold (model more reliable).
# Lower AUROC  -> looser threshold (avoid false positives).
OPTIMAL_THRESHOLDS = {
    "bottle":     0.68,
    "cable":      0.62,
    "capsule":    0.63,
    "carpet":     0.62,
    "grid":       0.58,
    "hazelnut":   0.70,
    "leather":    0.64,
    "metal_nut":  0.65,
    "pill":       0.60,
    "screw":      0.63,
    "tile":       0.62,
    "toothbrush": 0.67,
    "transistor": 0.62,
    "wood":       0.68,
    "zipper":     0.65,
}

CORESET_RATIO = 0.1
OVERLAY_ALPHA = 0.45
TOPK_RATIO    = 0.01


# ── Cached resources ──────────────────────────────────────────

@st.cache_resource(show_spinner="Loading feature extractor...")
def load_extractor() -> MultiScaleFeatureExtractor:
    """Load and freeze the ResNet18 multiscale feature extractor."""
    extractor = MultiScaleFeatureExtractor().to(DEVICE)
    extractor.eval()
    return extractor


@st.cache_resource(show_spinner="Building memory bank...")
def load_memory_bank(category: str) -> tuple[torch.Tensor | None, int | None]:
    """
    Load, normalize and apply greedy coreset to the memory bank.

    Why normalize?
        L2 normalization maps embeddings onto the unit hypersphere.
        This makes distances scale-invariant and improves discrimination
        between normal and anomalous patches.

    Why greedy coreset?
        Greedy k-center sampling maximizes coverage of the embedding space.
        Every original patch is within distance d of some selected patch,
        where d is minimized. This reduces false positives compared to
        random sampling which may miss rare normal patches.
    """
    pattern = f"*_{category}_multiscale.pt"
    matches = list(MEMORY_BANK_DIR.glob(pattern))
    if not matches:
        return None, None

    pt_path = max(matches, key=lambda p: p.stat().st_mtime)
    saved = torch.load(pt_path, map_location="cpu")
    memory_bank = saved["memory_bank"]
    full_size = memory_bank.shape[0]

    # L2 normalize
    memory_bank = F.normalize(memory_bank, dim=1)

    # Greedy coreset
    if memory_bank.shape[0] > 10_000:
        memory_bank = greedy_coreset_sampling(
            memory_bank,
            sampling_ratio=CORESET_RATIO,
            pre_sample_ratio=0.2,
        )

    return memory_bank, full_size


# ── Inference ─────────────────────────────────────────────────

def preprocess_pil(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to preprocessed tensor (1, C, 256, 256)."""
    image = image.convert("RGB")
    arr = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    processed = preprocess_image(tensor)
    return processed.unsqueeze(0)


def run_inference(
    image_tensor: torch.Tensor,
    extractor: MultiScaleFeatureExtractor,
    memory_bank: torch.Tensor,
) -> tuple[float, np.ndarray, float]:
    """
    Run PatchCore inference on a single image.

    Returns:
        score_normalized: anomaly score in [0, 1]
        heatmap: spatial anomaly map (256, 256) in [0, 1]
        elapsed: inference time in seconds
    """
    t0 = time.time()

    extractor.eval()
    with torch.no_grad():
        feature_map = extractor(image_tensor.to(DEVICE))

    B, C, H, W = feature_map.shape
    test_patches = extract_patch_embeddings(feature_map)

    # L2 normalize test patches — same space as memory bank
    test_patches = F.normalize(test_patches, dim=1)

    patch_distances = compute_patchwise_distances(
        test_patches=test_patches,
        memory_bank=memory_bank,
        chunk_size=1024,
    )

    anomaly_map = patch_distances_to_maps(patch_distances, B, H, W)

    # Image-level score: top-k mean
    flat = anomaly_map.view(1, -1)
    k = max(1, int(TOPK_RATIO * flat.shape[1]))
    raw_score = float(torch.topk(flat, k, dim=1).values.mean())

    # Soft normalization to [0, 1]
    score_normalized = float(np.clip(raw_score / (raw_score + 1.0), 0, 1))

    # Upsample to 256x256
    anomaly_map_up = F.interpolate(
        anomaly_map.unsqueeze(1),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Per-image min-max normalization for visualization
    a_min, a_max = anomaly_map_up.min(), anomaly_map_up.max()
    heatmap = ((anomaly_map_up - a_min) / (a_max - a_min + 1e-8)).cpu().numpy()

    elapsed = time.time() - t0
    return score_normalized, heatmap, elapsed


def make_overlay(original: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    """Blend jet heatmap over original image."""
    orig = np.array(original.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    colored = plt.colormaps["jet"](heatmap)[..., :3]
    blended = (1 - OVERLAY_ALPHA) * orig + OVERLAY_ALPHA * colored
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def render_result(
    original: Image.Image,
    score: float,
    heatmap: np.ndarray,
    elapsed: float,
    threshold: float,
    category: str,
    filename: str,
) -> None:
    """Render results for one uploaded image."""
    is_anomaly = score >= threshold

    st.markdown(f"#### `{filename}`")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Original")
        st.image(original, use_container_width=True)
    with col2:
        st.caption("Anomaly heatmap")
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with col3:
        st.caption("Overlay")
        st.image(make_overlay(original, heatmap), use_container_width=True)

    score_col, verdict_col = st.columns([1, 2])
    with score_col:
        st.metric(
            label="Anomaly score",
            value=f"{score:.3f}",
            delta=f"threshold {threshold:.2f}",
            delta_color="off",
        )
        st.progress(float(score))
        st.caption(f"Inference time: {elapsed:.2f}s")

    with verdict_col:
        if is_anomaly:
            st.error(
                f"### ❌ Defective\n"
                f"Score **{score:.3f}** exceeds threshold **{threshold:.2f}**. "
                f"Check the heatmap for defect localization."
            )
        else:
            st.success(
                f"### ✅ Normal\n"
                f"Score **{score:.3f}** is below threshold **{threshold:.2f}**. "
                f"No significant anomaly detected."
            )

    with st.expander("Technical details"):
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Category | `{category}` |
| Extractor | MultiScale ResNet18 (layer2 + layer3) |
| Embedding dim | 384 (128 layer2 + 256 layer3) |
| Normalization | L2 on memory bank and test patches |
| Coreset method | Greedy k-center ({CORESET_RATIO*100:.0f}%) |
| Scoring | Top-{TOPK_RATIO*100:.0f}% patch distances (mean) |
| Anomaly score | {score:.4f} (normalized) |
| Threshold | {threshold:.2f} (calibrated for {category}) |
| Device | `{DEVICE}` |
| Benchmark AUROC | {AUROC_REFERENCE[category]:.3f} |
| Inference time | {elapsed:.2f}s |
        """)

    st.divider()


# ── Main ──────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Industrial Anomaly Detection",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Industrial Anomaly Detection")
    st.markdown("**PatchCore** — ResNet18 multiscale | MVTec AD benchmark")
    st.divider()

    # ── Sidebar ──
    with st.sidebar:
        st.header("Configuration")

        category = st.selectbox(
            "Category",
            options=ALL_CATEGORIES,
            index=2,
            help="Select the industrial part category.",
        )

        st.caption(f"Benchmark AUROC: **{AUROC_REFERENCE[category]:.3f}**")

        default_threshold = OPTIMAL_THRESHOLDS[category]
        threshold = st.slider(
            "Detection threshold",
            min_value=0.0,
            max_value=1.0,
            value=default_threshold,
            step=0.01,
            help=(
                "Score above this value is classified as Defective. "
                "Default is calibrated per category. "
                "Lower = more sensitive. Higher = more conservative."
            ),
        )

        st.divider()
        st.markdown("**How it works**")
        st.caption(
            "PatchCore extracts local patch features via a pretrained ResNet18. "
            "Each patch is compared to a memory bank of normal patches — "
            "high distance signals an anomaly. "
            "The heatmap shows where the defect is localized."
        )
        st.caption(
            "Embeddings are L2-normalized. Memory bank uses greedy k-center "
            "coreset for better coverage of the normal patch space."
        )
        st.caption(f"Device: `{DEVICE}`")

    # ── Load models ──
    extractor = load_extractor()
    memory_bank, full_size = load_memory_bank(category)

    if memory_bank is None:
        st.error(
            f"No memory bank found for **{category}**. "
            "Run the benchmark notebook with `SAVE_MEMORY_BANKS = True` first."
        )
        st.stop()

    st.success(
        f"Ready — {memory_bank.shape[0]:,} patches "
        f"(greedy coreset {CORESET_RATIO*100:.0f}% of {full_size:,}) "
        f"× {memory_bank.shape[1]} dims"
    )

    # ── Multi-image upload ──
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="You can upload multiple images at once.",
    )

    if not uploaded_files:
        st.info("Upload one or more images to start detection.")
        return

    st.divider()
    st.markdown(f"### Results — {len(uploaded_files)} image(s)")

    for uploaded in uploaded_files:
        original = Image.open(uploaded)
        image_tensor = preprocess_pil(original)

        with st.spinner(f"Analyzing `{uploaded.name}`..."):
            score, heatmap, elapsed = run_inference(
                image_tensor, extractor, memory_bank
            )

        render_result(
            original=original,
            score=score,
            heatmap=heatmap,
            elapsed=elapsed,
            threshold=threshold,
            category=category,
            filename=uploaded.name,
        )


if __name__ == "__main__":
    main()