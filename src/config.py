from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data path
DATA_DIR = BASE_DIR / "data" / "mvtec_ad"

CATEGORY = "capsule"
IMAGE_SIZE = 256
BATCH_SIZE = 14
EPOCHS = 14
LEARNING_RATE = 1e-3
SEED = 42
