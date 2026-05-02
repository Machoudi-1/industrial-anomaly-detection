"""
prepare_memory_banks.py
-----------------------
Precompute normalized + greedy coreset memory banks for all categories.

Run this script ONCE to prepare memory banks for the Streamlit app.
The app loads directly from the ready/ directory with no waiting time.

Usage (local):
    poetry run python scripts/prepare_memory_banks.py

Usage (Colab — custom paths):
    python scripts/prepare_memory_banks.py \\
        --input-dir /content/drive/MyDrive/anomaly_outputs/memory_banks \\
        --output-dir /content/drive/MyDrive/anomaly_outputs/memory_banks/ready

Options:
    --input-dir       Directory containing raw memory bank .pt files
    --output-dir      Directory where ready banks will be saved
    --coreset-ratio   Fraction of patches to keep (default: 0.1)
    --pre-sample-ratio Pre-sampling ratio for greedy search (default: 0.1)
    --category        Process a single category only
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
    input_dir: Path,
    output_dir: Path,
    coreset_ratio: float,
    pre_sample_ratio: float,
) -> bool:
    """
    Load, normalize and apply greedy coreset for one category.

    Steps:
        1. Find the most recent memory bank .pt file for this category
        2. L2 normalize all embeddings
        3. Apply greedy k-center coreset sampling
        4. Save the result to output_dir/

    Returns True if successful, False if memory bank not found.
    """
    pattern = f"*_{category}_multiscale.pt"
    matches = [p for p in input_dir.glob(pattern) if "ready" not in str(p)]

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
        f"  [{category}] Greedy coreset — "
        f"{coreset.shape[0]:,} patches "
        f"({coreset_ratio*100:.0f}% of {full_size:,}) "
        f"in {elapsed:.1f}s"
    )

    # Step 3 — Save
    output_path = output_dir / f"{category}_ready.pt"
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
    print(f"  [{category}] Saved → {output_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute normalized greedy coreset memory banks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing raw memory bank .pt files. "
            "Default: <repo_root>/outputs/memory_banks"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where ready banks will be saved. "
            "Default: <repo_root>/outputs/memory_banks/ready"
        ),
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
        default=0.1,
        help="Pre-sampling ratio for greedy coreset (default: 0.1).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=ALL_CATEGORIES,
        help="Process a single category only (default: all).",
    )
    args = parser.parse_args()

    # Resolve paths
    input_dir = args.input_dir or (ROOT / "outputs" / "memory_banks")
    output_dir = args.output_dir or (ROOT / "outputs" / "memory_banks" / "ready")

    output_dir.mkdir(parents=True, exist_ok=True)

    categories = [args.category] if args.category else ALL_CATEGORIES

    print(f"\nPreparing memory banks")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(
        f"  Coreset: {args.coreset_ratio*100:.0f}% (pre-sample {args.pre_sample_ratio*100:.0f}%)"
    )
    print(f"  Categories: {categories}\n")

    success, skipped = 0, 0
    total_t0 = time.time()

    for category in tqdm(categories, desc="Preparing", unit="category"):
        print()
        ok = prepare_category(
            category=category,
            input_dir=input_dir,
            output_dir=output_dir,
            coreset_ratio=args.coreset_ratio,
            pre_sample_ratio=args.pre_sample_ratio,
        )
        if ok:
            success += 1
        else:
            skipped += 1

    total_elapsed = time.time() - total_t0
    print(f"\nDone in {total_elapsed:.1f}s — {success} prepared, {skipped} skipped.")
    print(f"Ready banks: {output_dir}")
    print("\nLaunch the app:")
    print("  poetry run streamlit run app/app.py")


if __name__ == "__main__":
    main()
