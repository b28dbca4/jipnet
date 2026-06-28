"""
Create a small synthetic dataset for smoke-testing the training pipeline on
environments (e.g. Kaggle) where real fingerprint data is not yet available.

Generates:
  <out_dir>/img/         — synthetic 160×160 grayscale patch images
  <out_dir>/info/        — matching .txt pair-info files
  data/train.npy         — manifest pointing to the above (95% train)
  data/valid.npy         — manifest (5% valid)

Usage:
    python make_data/create_dummy_data.py
    python make_data/create_dummy_data.py --out_dir /kaggle/working/dummy --n_fingers 20
"""

import argparse
import os
import random

import cv2
import numpy as np


def make_dummy_patch(size=160, seed=None):
    """Synthetic grayscale fingerprint-like image (ridge pattern + noise)."""
    rng = np.random.default_rng(seed)
    # Sinusoidal ridge pattern
    x = np.linspace(0, 4 * np.pi, size)
    ridge = (np.sin(x[None, :] + rng.uniform(0, 2 * np.pi)) * 60 + 180).astype(np.uint8)
    ridge = np.tile(ridge, (size, 1))
    noise = rng.integers(0, 30, (size, size), dtype=np.uint8)
    img = np.clip(ridge.astype(int) + noise - 15, 0, 255).astype(np.uint8)
    # Circular mask to simulate partial fingerprint
    cy, cx = size // 2, size // 2
    r = size // 2 - 4
    mask = (
        (np.arange(size)[:, None] - cy) ** 2 + (np.arange(size)[None, :] - cx) ** 2
    ) > r**2
    img[mask] = 255
    return img


def write_info(path, img1_path, img2_path, info1, info2, gt):
    with open(path, "w") as f:
        f.write(
            "File for patch_info. from top to left: img_path1/2, info1/2, gt, "
            "info from left to right: row col theta.\n"
        )
        f.write(img1_path + "\n")
        f.write(img2_path + "\n")
        f.write(" ".join(f"{v:.4f}" for v in info1) + "\n")
        f.write(" ".join(f"{v:.4f}" for v in info2) + "\n")
        f.write(f"{gt}\n")


def create_dummy_dataset(out_dir, n_fingers=10, n_impressions=4, patch_size=160):
    img_dir = os.path.join(out_dir, "img")
    info_dir = os.path.join(out_dir, "info")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    rng = random.Random(42)
    pair_idx = 0

    for finger in range(1, n_fingers + 1):
        # Save impressions
        img_paths = []
        for imp in range(1, n_impressions + 1):
            img = make_dummy_patch(patch_size, seed=finger * 100 + imp)
            p = os.path.join(img_dir, f"{finger}_{imp}.png")
            cv2.imwrite(p, img)
            img_paths.append(p)

        # Genuine pairs (same finger, different impressions)
        for i in range(n_impressions):
            for j in range(n_impressions):
                if i == j:
                    continue
                info1 = [patch_size / 2, patch_size / 2, rng.uniform(-180, 180)]
                info2 = [patch_size / 2, patch_size / 2, rng.uniform(-180, 180)]
                write_info(
                    os.path.join(info_dir, f"{pair_idx}.txt"),
                    img_paths[i], img_paths[j], info1, info2, gt=1,
                )
                pair_idx += 1

        # Impostor pair (this finger vs finger+1, wrap around)
        other_finger = (finger % n_fingers) + 1
        other_imp = 1
        other_path = os.path.join(img_dir, f"{other_finger}_{other_imp}.png")
        if os.path.exists(other_path):
            info1 = [patch_size / 2, patch_size / 2, rng.uniform(-180, 180)]
            info2 = [patch_size / 2, patch_size / 2, rng.uniform(-180, 180)]
            write_info(
                os.path.join(info_dir, f"{pair_idx}.txt"),
                img_paths[0], other_path, info1, info2, gt=0,
            )
            pair_idx += 1

    print(f"Generated {pair_idx} pairs  →  {out_dir}")
    return info_dir


def build_manifest(info_dir, data_dir, train_ratio=0.95):
    from glob import glob

    txt_files = sorted(glob(os.path.join(info_dir, "*.txt")))
    random.Random(42).shuffle(txt_files)

    n_train = max(1, int(len(txt_files) * train_ratio))
    train_files = txt_files[:n_train]
    valid_files = txt_files[n_train:] or txt_files[:1]  # at least 1 for valid

    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "train.npy"), {"info_lst": train_files})
    np.save(os.path.join(data_dir, "valid.npy"), {"info_lst": valid_files})

    print(f"Train : {len(train_files):>4d} pairs  →  {data_dir}/train.npy")
    print(f"Valid : {len(valid_files):>4d} pairs  →  {data_dir}/valid.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic dummy data for smoke-testing.")
    parser.add_argument("--out_dir", default="./dummy_data",
                        help="Directory to write synthetic images and info files.")
    parser.add_argument("--data_dir", default="./data",
                        help="Directory to write train.npy / valid.npy manifests.")
    parser.add_argument("--n_fingers", type=int, default=10,
                        help="Number of synthetic 'fingers' (identities).")
    parser.add_argument("--n_impressions", type=int, default=4,
                        help="Number of impressions per finger.")
    parser.add_argument("--patch_size", type=int, default=160,
                        help="Image size in pixels (must match configs/JIPNet.yaml input_size).")
    args = parser.parse_args()

    print("Creating synthetic dataset …")
    info_dir = create_dummy_dataset(
        args.out_dir,
        n_fingers=args.n_fingers,
        n_impressions=args.n_impressions,
        patch_size=args.patch_size,
    )
    print("Building manifests …")
    build_manifest(info_dir, args.data_dir)
    print("Done. Update configs/JIPNet.yaml:")
    print(f"  train_info_path: {args.data_dir}/train.npy")
    print(f"  valid_info_path: {args.data_dir}/valid.npy")
