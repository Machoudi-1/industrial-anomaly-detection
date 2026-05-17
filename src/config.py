"""
config.py
---------
Central configuration for the industrial anomaly detection project.

Defines all shared constants: paths, image size, training hyperparameters,
and the list of MVTec AD categories. Import from here instead of hardcoding
values across scripts.

Usage:
    from config import DATA_DIR, ALL_CATEGORIES, BATCH_SIZE
"""

from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = BASE_DIR / "data" / "mvtec_ad"

# All images are resized to IMAGE_SIZE × IMAGE_SIZE before processing
IMAGE_SIZE = 256

#  Training defaults
# These are defaults — train_autoencoder.py accepts CLI overrides
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42

# MVTec AD categories
# All 15 product categories of the MVTec Anomaly Detection benchmark
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

# Output paths
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"  # plots and heatmaps
METRICS_DIR = OUTPUT_DIR / "metrics"  # AUROC scores, CSV results
MEMORY_DIR = OUTPUT_DIR / "memory_banks"  # PatchCore coreset files (.pt)
TABLES_DIR = OUTPUT_DIR / "tables"  # summary tables
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"  # saved model weights
