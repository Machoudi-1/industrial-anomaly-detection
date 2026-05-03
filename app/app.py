"""
app.py
------
Streamlit application for PatchCore industrial anomaly detection.

Prerequisites:
    Run prepare_memory_banks.py first to build the ready memory banks:
        poetry run python scripts/prepare_memory_banks.py

Usage:
    poetry run streamlit run app/app.py
"""

import sys
import time
from pathlib import Path

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
)
from utils.normalization import preprocess_image

# Constants
READY_DIR = ROOT / "outputs" / "memory_banks" / "ready"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

AUROC_REFERENCE = {
    "bottle": 0.9992,
    "cable": 0.8564,
    "capsule": 0.8783,
    "carpet": 0.8941,
    "grid": 0.5773,
    "hazelnut": 0.9996,
    "leather": 0.9253,
    "metal_nut": 0.9462,
    "pill": 0.7621,
    "screw": 0.8965,
    "tile": 0.8413,
    "toothbrush": 0.9611,
    "transistor": 0.8592,
    "wood": 0.9833,
    "zipper": 0.9309,
}

# Per-category thresholds calibrated from benchmark results
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

OVERLAY_ALPHA = 0.45
TOPK_RATIO = 0.01


# Cached resources


@st.cache_resource(show_spinner="Loading feature extractor...")
def load_extractor() -> MultiScaleFeatureExtractor:
    extractor = MultiScaleFeatureExtractor().to(DEVICE)
    extractor.eval()
    return extractor


@st.cache_resource(show_spinner="Loading memory bank...")
def load_memory_bank(category: str) -> tuple[torch.Tensor | None, dict]:
    """
    Load the precomputed ready memory bank for a category.

    The ready memory banks are produced by scripts/prepare_memory_banks.py.
    They are already L2-normalized and greedy-coreset sampled - loading
    is instantaneous, no computation needed at startup.
    """
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
        "normalized": saved.get("normalized", True),
        "coreset_method": saved.get("coreset_method", "greedy"),
    }
    return memory_bank, meta


# Inference


def preprocess_pil(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to preprocessed tensor (1, C, 256, 256)."""
    image = image.convert("RGB")
    arr = np.array(image, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return preprocess_image(tensor).unsqueeze(0)


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

    # L2 normalize test patches - same space as memory bank
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
    score_normalized = float(np.clip(raw_score / (raw_score + 1.0), 0, 1))

    # Upsample to 256×256
    anomaly_map_up = F.interpolate(
        anomaly_map.unsqueeze(1),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    a_min, a_max = anomaly_map_up.min(), anomaly_map_up.max()
    heatmap = ((anomaly_map_up - a_min) / (a_max - a_min + 1e-8)).cpu().numpy()

    return score_normalized, heatmap, time.time() - t0


def make_overlay(original: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    orig = (
        np.array(original.convert("RGB").resize((256, 256)), dtype=np.float32) / 255.0
    )
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
    meta: dict,
) -> None:
    """Render detection results for one image."""
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
        st.caption(f"Inference: {elapsed:.2f}s")

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
| Embedding dim | {meta.get('embedding_dim', 384)} |
| Normalization | L2 (memory bank + test patches) |
| Coreset method | {meta.get('coreset_method', 'greedy')} k-center |
| Coreset size | {meta.get('coreset_size', '?'):,} patches ({meta.get('coreset_ratio', '?') and f"{meta['coreset_ratio']*100:.0f}%" } of {meta.get('full_size', '?'):,}) |
| Anomaly score | {score:.4f} |
| Threshold | {threshold:.2f} (calibrated for `{category}`) |
| Device | `{DEVICE}` |
| Benchmark AUROC | {AUROC_REFERENCE[category]:.3f} |
| Inference time | {elapsed:.2f}s |
        """)

    st.divider()


# Main


def main():
    st.set_page_config(
        page_title="Industrial Anomaly Detection",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Industrial Anomaly Detection")
    st.markdown("**PatchCore** - ResNet18 multiscale | MVTec AD benchmark")
    st.divider()

    #  Sidebar
    with st.sidebar:
        st.header("Configuration")

        category = st.selectbox(
            "Category",
            options=ALL_CATEGORIES,
            index=2,
            help="Select the industrial part category.",
        )

        st.caption(f"Benchmark AUROC: **{AUROC_REFERENCE[category]:.3f}**")

        threshold = st.slider(
            "Detection threshold",
            min_value=0.0,
            max_value=1.0,
            value=OPTIMAL_THRESHOLDS[category],
            step=0.01,
            help=(
                "Score above this value → Defective. "
                "Default is calibrated per category. "
                "Lower = more sensitive. Higher = more conservative."
            ),
        )

        st.divider()
        st.markdown("**How it works**")
        st.caption(
            "PatchCore extracts local patch features via a pretrained ResNet18. "
            "Each patch is compared to a memory bank of normal patches - "
            "high distance signals an anomaly. "
            "The heatmap shows where the defect is localized."
        )
        st.caption(
            "Embeddings are L2-normalized. Memory bank uses greedy k-center "
            "coreset for better coverage of the normal patch space."
        )
        st.caption(f"Device: `{DEVICE}`")

    # Load models
    extractor = load_extractor()
    memory_bank, meta = load_memory_bank(category)

    if memory_bank is None:
        st.error(f"No ready memory bank found for **{category}**.")
        st.info(
            "Run the preparation script first:\n"
            "```\npoetry run python scripts/prepare_memory_banks.py\n```"
        )
        st.stop()

    st.success(
        f"Ready - {meta['coreset_size']:,} patches "
        f"(greedy coreset {meta['coreset_ratio']*100:.0f}% of {meta['full_size']:,}) "
        f"× {meta['embedding_dim']} dims"
    )

    # Multi-image upload
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Upload images of the selected industrial part. Multiple files supported.",
    )

    if not uploaded_files:
        st.info("Upload one or more images to start detection.")
        return

    st.divider()
    st.markdown(f"### Results - {len(uploaded_files)} image(s)")

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
            meta=meta,
        )


if __name__ == "__main__":
    main()
