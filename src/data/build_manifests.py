import json
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm
import argparse


DATA_ROOT = Path("data/raw")


def parse_chexpert_path(raw_path: str) -> Tuple[str, str, str]:
    """
    Parse and normalize a CheXpert image path.

    :param raw_path: Original path from CSV, e.g. 'CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg'
    :returns patient_id_raw: 'patient00001'
    :returns study_id_raw: 'study1'
    :returns relative_path: 'train/patient00001/study1/view1_frontal.jpg'
    """
    prefix = "CheXpert-v1.0-small/"

    if not raw_path.startswith(prefix):
        raise ValueError(f"Unexpected CheXpert path format: {raw_path}")

    relative_path = raw_path.replace(prefix, "", 1)

    p = Path(relative_path)

    if len(p.parts) < 4:
        raise ValueError(f"Incomplete CheXpert path: {relative_path}")

    patient_id_raw = p.parts[-3]
    study_id_raw = p.parts[-2]

    if not patient_id_raw.startswith("patient"):
        raise ValueError(f"Invalid patient identifier: {patient_id_raw}")

    if not study_id_raw.startswith("study"):
        raise ValueError(f"Invalid study identifier: {study_id_raw}")

    return patient_id_raw, study_id_raw, relative_path


def load_nih() -> List[Dict]:
    records = []

    labels_csv = DATA_ROOT / "nih" / "Data_Entry_2017_v2020.csv"
    labels = pd.read_csv(labels_csv)

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="NIH"):
        img_path = DATA_ROOT / "nih" / "images" / row["Image Index"]
        nih_labels = set(row["Finding Labels"].split("|"))

        records.append({
            "dataset": "nih",
            "file_path": str(img_path),
            "patient_id": f"nih_{row['Patient ID']}",
            "sex": row["Patient Sex"],
            "age": row["Patient Age"],
            "study_id": None,  # NIH has no true study ID
            "view_type": "Frontal", # NIH images are all frontal
            "view_position": row["View Position"],
            "label_source_file": str(labels_csv),
            "atelectasis": int("Atelectasis" in nih_labels),
            "cardiomegaly": int("Cardiomegaly" in nih_labels),
            "consolidation": int("Consolidation" in nih_labels),
            "edema": int("Edema" in nih_labels),
            "effusion": int("Effusion" in nih_labels),
        })

    return records


def load_chexpert_train() -> List[Dict]:
    records = []

    labels_csv = DATA_ROOT / "chexpert" / "train.csv"
    labels = pd.read_csv(labels_csv)

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="CheXpert (train)"):
        patient_id_raw, study_id_raw, relative_path = parse_chexpert_path(row["Path"])
        img_path = DATA_ROOT / "chexpert" / relative_path

        records.append({
            "dataset": "chexpert-train",
            "file_path": str(img_path),
            "patient_id": f"chexpert_{patient_id_raw}",
            "sex": 'M' if row["Sex"] == "Male" else 'F',
            "age": row["Age"],
            "study_id": f"chexpert_{study_id_raw}",
            "view_type": row["Frontal/Lateral"],
            "view_position": row["AP/PA"],
            "label_source_file": str(labels_csv),
            "atelectasis": row["Atelectasis"],
            "cardiomegaly": row["Cardiomegaly"],
            "consolidation": row["Consolidation"],
            "edema": row["Edema"],
            "effusion": row["Pleural Effusion"],
        })

    return records


def load_chexpert_valid() -> List[Dict]:
    records = []

    labels_csv = DATA_ROOT / "chexpert" / "valid.csv"
    labels = pd.read_csv(labels_csv)

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="CheXpert (valid)"):
        patient_id_raw, study_id_raw, relative_path = parse_chexpert_path(row["Path"])
        img_path = DATA_ROOT / "chexpert" / relative_path

        records.append({
            "dataset": "chexpert-valid",
            "file_path": str(img_path),
            "patient_id": f"chexpert_{patient_id_raw}",
            "sex": 'M' if row["Sex"] == "Male" else 'F',
            "age": row["Age"],
            "study_id": f"chexpert_{study_id_raw}",
            "view_type": row["Frontal/Lateral"],
            "view_position": row["AP/PA"],
            "label_source_file": str(labels_csv),
            "atelectasis": row["Atelectasis"],
            "cardiomegaly": row["Cardiomegaly"],
            "consolidation": row["Consolidation"],
            "edema": row["Edema"],
            "effusion": row["Pleural Effusion"],
        })

    return records


def main(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    loaders = [
        load_nih,
        load_chexpert_train,
        load_chexpert_valid,
    ]

    all_records: List[Dict] = []

    for loader in loaders:
        records = loader()
        all_records.extend(records)

    df = pd.DataFrame(all_records)

    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build master manifest for CXR datasets")
    parser.add_argument(
        "--output", 
        type=Path,
        default=Path("data/manifests/master_manifest.csv"), 
        help="Output CSV path for the manifest"
    )

    args = parser.parse_args()

    main(args.output)