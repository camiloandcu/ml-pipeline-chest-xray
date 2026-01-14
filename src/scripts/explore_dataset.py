import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LABEL_COLUMNS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "effusion",
]


def load_dataset(path: Path, split: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)

    if split is not None:
        if "split" not in df.columns:
            raise ValueError("Dataset does not contain a 'split' column.")
        df = df[df["split"] == split]

    return df.copy()


def compute_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute label distribution with:
    - positive (1)
    - negative (0)
    - uncertain (-1)
    - empty (NaN)

    Percentages are computed over non-empty samples.
    """
    rows = []

    for label in LABEL_COLUMNS:
        series = df[label]

        positive = (series == 1).sum()
        negative = (series == 0).sum()
        uncertain = (series == -1).sum()
        empty = series.isna().sum()

        non_empty = positive + negative + uncertain

        rows.append({
            "label": label,
            "positive": int(positive),
            "negative": int(negative),
            "uncertain": int(uncertain),
            "empty": int(empty),
            "positive_pct": positive / non_empty if non_empty > 0 else 0.0,
            "uncertain_pct": uncertain / non_empty if non_empty > 0 else 0.0,
        })

    return pd.DataFrame(rows).set_index("label")


def compute_sex_distribution(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    counts = df["sex"].value_counts(dropna=False)

    rows = []
    for value, count in counts.items():
        label = value if pd.notna(value) else "missing"
        rows.append({
            "sex": label,
            "count": int(count),
            "percentage": count / total,
        })

    return pd.DataFrame(rows).set_index("sex")


def compute_view_position_distribution(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    counts = df["view_position"].value_counts(dropna=False)

    rows = []
    for value, count in counts.items():
        label = value if pd.notna(value) else "missing"
        rows.append({
            "view_position": label,
            "count": int(count),
            "percentage": count / total,
        })

    return pd.DataFrame(rows).set_index("view_position")


def plot_age_distribution(df: pd.DataFrame, out_path: Path):
    """
    Plot age distribution following medical imaging reporting best practices.
    """
    ages = df["age"].dropna()

    # Restrict to plausible adult range
    ages = ages[(ages >= 0) & (ages <= 100)]

    plt.figure(figsize=(10, 4))

    # 5-year bins are standard in clinical reporting
    bins = range(0, 100, 5)

    plt.hist(
        ages,
        bins=bins,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.xlabel("Age [years]")
    plt.ylabel("Number of images")
    plt.title("Patient Age Distribution")

    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)
    plt.close()


def main(dataset_path: Path, out_dir: Path, split: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path, split)

    label_table = compute_label_distribution(df)
    sex_table = compute_sex_distribution(df)
    view_table = compute_view_position_distribution(df)

    label_table.to_csv(out_dir / "label_distribution.csv")
    sex_table.to_csv(out_dir / "sex_distribution.csv")
    view_table.to_csv(out_dir / "view_position_distribution.csv")

    plot_age_distribution(df, out_dir / "age_distribution.png")

    print("Dataset description generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore label and demographic distributions in split/manifest CSV")

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/manifests/master_manifest.csv"),
        help="Path to the dataset CSV file",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/reports"),
        help="Directory to save plots",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default=None,
        help="Optional split to analyze",
    )

    args = parser.parse_args()

    main(args.dataset, args.output_dir, args.split)
