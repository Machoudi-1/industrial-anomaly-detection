from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = BASE_DIR / "data" / "mvtec_ad"
IMAGE_SIZE = 256

#  Training defaults
# These are defaults — train_autoencoder.py accepts CLI overrides
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42

# All 15 MVTec categories
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
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MEMORY_DIR = OUTPUT_DIR / "memory_banks"
TABLES_DIR = OUTPUT_DIR / "tables"
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"
