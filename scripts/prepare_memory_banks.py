"""
prepare_memory_banks.py
-----------------------
Precompute normalized + greedy coreset memory banks for all categories.

Run this script ONCE after the benchmark to prepare the memory banks
for the Streamlit app. The app then loads directly from outputs/memory_banks/ready/
with no waiting time.

Usage:
    poetry run python scripts/prepare_memory_banks.py

    # Custom coreset ratio
    poetry run python scripts/prepare_memory_banks.py --coreset-ratio 0.15

    # Single category
    poetry run python scripts/prepare_memory_banks.py --category capsule
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.patchcore import greedy_coreset_sampling

MEMORY_BANK_DIR = ROOT / "outputs" / "memory_banks"
READY_DIR = ROOT / "outputs" / "memory_banks" / "ready"

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


def prepare_category(
    category: str,
    coreset_ratio: float,
    pre_sample_ratio: float,
) -> bool:
    """
    Load, normalize and apply greedy coreset for one category.

    Steps:
        1. Find the most recent memory bank .pt file for this category
        2. L2 normalize all embeddings
        3. Apply greedy k-center coreset sampling
        4. Save the result to outputs/memory_banks/ready/

    Returns True if successful, False if memory bank not found.
    """
    pattern = f"*_{category}_multiscale.pt"
    matches = list(MEMORY_BANK_DIR.glob(pattern))

    # Skip already-prepared files in ready/
    matches = [m for m in matches if "ready" not in str(m)]

    if not matches:
        print(f"  [{category}] WARNING — no memory bank found, skipping.")
        return False

    pt_path = max(matches, key=lambda p: p.stat().st_mtime)
    saved = torch.load(pt_path, map_location="cpu")
    memory_bank = saved["memory_bank"]
    full_size = memory_bank.shape[0]

    print(
        f"  [{category}] Loaded — {full_size:,} patches × {memory_bank.shape[1]} dims"
    )

    # Step 1 — L2 normalize
    memory_bank = F.normalize(memory_bank, dim=1)
    print(f"  [{category}] L2 normalized.")

    # Step 2 — Greedy coreset
    t0 = time.time()
    coreset = greedy_coreset_sampling(
        memory_bank,
        sampling_ratio=coreset_ratio,
        pre_sample_ratio=pre_sample_ratio,
    )
    elapsed = time.time() - t0
    print(
        f"  [{category}] Greedy coreset done — "
        f"{coreset.shape[0]:,} patches ({coreset_ratio*100:.0f}% of {full_size:,}) "
        f"in {elapsed:.1f}s"
    )

    # Step 3 — Save
    output_path = READY_DIR / f"{category}_ready.pt"
    torch.save(
        {
            "category": category,
            "extractor_type": "multiscale",
            "full_size": full_size,
            "coreset_ratio": coreset_ratio,
            "normalized": True,
            "coreset_method": "greedy",
            "memory_bank": coreset,
        },
        output_path,
    )
    print(f"  [{category}] Saved to {output_path.relative_to(ROOT)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute normalized greedy coreset memory banks for the Streamlit app."
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.1,
        help="Fraction of patches to keep (default: 0.1 = 10%%).",
    )
    parser.add_argument(
        "--pre-sample-ratio",
        type=float,
        default=0.2,
        help="Pre-sampling ratio for greedy coreset (default: 0.2).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=ALL_CATEGORIES,
        help="Process a single category (default: all categories).",
    )
    args = parser.parse_args()

    READY_DIR.mkdir(parents=True, exist_ok=True)

    categories = [args.category] if args.category else ALL_CATEGORIES

    print(f"\nPreparing memory banks — coreset {args.coreset_ratio*100:.0f}%")
    print(f"Output dir: {READY_DIR.relative_to(ROOT)}")
    print(f"Categories: {categories}\n")

    success, skipped = 0, 0

    for category in tqdm(categories, desc="Preparing", unit="category"):
        print()
        ok = prepare_category(
            category=category,
            coreset_ratio=args.coreset_ratio,
            pre_sample_ratio=args.pre_sample_ratio,
        )
        if ok:
            success += 1
        else:
            skipped += 1

    print(f"\nDone — {success} prepared, {skipped} skipped.")
    print(f"Ready banks saved in: {READY_DIR.relative_to(ROOT)}")
    print("\nYou can now launch the app:")
    print("  poetry run streamlit run app/app.py")


if __name__ == "__main__":
    main()
