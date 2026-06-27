"""
Build train/valid manifest .npy files from one or more info directories produced
by generate_patch.py.  Supports merging multiple datasets into a single manifest.

Usage (single dataset):
    python make_data/build_manifest.py

Usage (multiple datasets):
    python make_data/build_manifest.py \
        --info_dirs data_affine/FVC2002_DB1A/results/info \
                    data_affine/FVC2002_DB3A/results/info \
        --out_dir data
"""

import argparse
import os
import random
from glob import glob

import numpy as np


def build_manifest(info_dirs, out_dir, train_ratio=0.8, seed=42):
    """
    Collect *.txt pair-info files from one or more directories, split into
    train/valid, and save manifests as data/train.npy and data/valid.npy.

    Args:
        info_dirs   : str or list[str] — directories containing .txt pair files
        out_dir     : directory where train.npy and valid.npy will be saved
        train_ratio : fraction used for training (rest → validation)
        seed        : random seed for reproducible split
    """
    if isinstance(info_dirs, str):
        info_dirs = [info_dirs]

    txt_files = []
    for d in info_dirs:
        found = sorted(glob(os.path.join(d, "*.txt")))
        if not found:
            print(f"  WARNING: no .txt files found in {d!r} — skipping")
        else:
            print(f"  {d}: {len(found)} pairs")
        txt_files.extend(found)

    if not txt_files:
        raise FileNotFoundError(
            "No .txt pair files found in any of the provided directories."
        )

    print(f"Total pairs across all datasets: {len(txt_files)}")

    random.seed(seed)
    random.shuffle(txt_files)

    n_train = int(len(txt_files) * train_ratio)
    train_files = txt_files[:n_train]
    valid_files = txt_files[n_train:]

    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.npy")
    valid_path = os.path.join(out_dir, "valid.npy")

    np.save(train_path, {"info_lst": train_files})
    np.save(valid_path, {"info_lst": valid_files})

    print(f"Train : {len(train_files):>6d} pairs  →  {train_path}")
    print(f"Valid : {len(valid_files):>6d} pairs  →  {valid_path}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build train/valid manifests from generated pair-info directories."
    )
    parser.add_argument(
        "--info_dirs",
        nargs="+",
        default=["./data_affine/results/info"],
        help=(
            "One or more directories containing .txt pair-info files. "
            "Example: --info_dirs data_affine/DB_A/results/info data_affine/DB_B/results/info"
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="./data",
        help="Output directory for train.npy and valid.npy (default: ./data).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of pairs used for training (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/valid split (default: 42).",
    )
    args = parser.parse_args()

    build_manifest(
        info_dirs=args.info_dirs,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
