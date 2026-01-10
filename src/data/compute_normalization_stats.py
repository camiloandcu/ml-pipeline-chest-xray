from pathlib import Path
import argparse
import json

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image_grayscale(path: Path) -> np.ndarray:
    """
    Load image as float32 grayscale array in [0, 1].
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compute_mean_std(image_paths):
    """
    Compute global mean and std over a list of image paths.
    """
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    num_pixels = 0

    for path in tqdm(image_paths, desc="Computing normalization stats"):
        img = load_image_grayscale(path)
        pixel_sum += img.sum()
        pixel_sq_sum += np.square(img).sum()
        num_pixels += img.size

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean ** 2)

    return float(mean), float(std)


def main(
    splits_csv: Path,
    output_path: Path,
    seed: int,
):
    df = pd.read_csv(splits_csv)

    train_df = df[df["split"] == "train"]

    if train_df.empty:
        raise ValueError("No training samples found in split file.")

    image_paths = [Path(p) for p in train_df["file_path"]]

    mean, std = compute_mean_std(image_paths)

    stats = {
        "mean": mean,
        "std": std,
        "num_images": len(image_paths),
        "computed_on": "train",
        "seed": seed,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved normalization stats to {output_path}")
    print(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute normalization statistics on training split only"
    )
    parser.add_argument(
        "--splits-csv",
        type=Path,
        default=Path("data/splits/splits_v1.csv"),
        help="Path to split CSV produced by build_splits.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stats/stats_v1.json"),
        help="Output path for normalization statistics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (for traceability)",
    )

    args = parser.parse_args()

    main(
        splits_csv=args.splits_csv,
        output_path=args.output,
        seed=args.seed,
    )
