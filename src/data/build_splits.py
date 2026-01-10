from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def apply_label_policies(df, label_cols, uncertain_policy, blank_policy):
    for col in label_cols:
        # Handle blank labels (NaN)
        if blank_policy == "negative":
            df[col] = df[col].fillna(0)
        elif blank_policy == "uncertain":
            df[col] = df[col].fillna(-1)

        # Handle uncertainty (-1)
        if uncertain_policy == "negative":
            df[col] = df[col].replace(-1, 0)
        elif uncertain_policy == "positive":
            df[col] = df[col].replace(-1, 1)
        elif uncertain_policy == "ignore":
            pass  # keep -1, to be masked later

    return df


def main(args):
    manifest_path = Path(args.manifest)
    out_path = Path(args.output)

    df = pd.read_csv(manifest_path)

    label_cols = [
        "atelectasis",
        "cardiomegaly",
        "consolidation",
        "edema",
        "effusion",
    ]

    # View filtering
    if args.view_policy == "frontal_only":
        df = df[df["view_type"] == "Frontal"].reset_index(drop=True)

    # Apply label policies
    df = apply_label_policies(
        df,
        label_cols,
        uncertain_policy=args.uncertain_policy,
        blank_policy=args.blank_policy,
    )

    # Patient-level split
    rng = np.random.default_rng(args.seed)
    patients = df["patient_id"].unique()
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])

    def assign_split(pid):
        if pid in train_patients:
            return "train"
        elif pid in val_patients:
            return "val"
        else:
            return "test"

    df["split"] = df["patient_id"].apply(assign_split)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("Split summary:")
    print(df["split"].value_counts())
    print(f"Wrote split file to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build patient-level train/val/test splits")

    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/master_manifest.csv"), help="Path to the manifest CSV")
    parser.add_argument("--output", type=Path, default=Path("data/splits/splits_v1.csv"), help="Output CSV path for the splits")

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)

    parser.add_argument(
        "--uncertain-policy",
        choices=["negative", "positive", "ignore"],
        default="negative",
        help="Policy for handling uncertain labels (-1)",
    )

    parser.add_argument(
        "--blank-policy",
        choices=["negative", "uncertain"],
        default="negative",
        help="Policy for handling blank labels (NaN)",
    )

    parser.add_argument(
        "--view-policy",
        choices=["frontal_only", "all"],
        default="frontal_only",
        help="Policy for including view types",
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed (for traceability)",    
    )

    args = parser.parse_args()
    main(args)
