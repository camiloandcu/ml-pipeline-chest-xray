
## Data Management & Governance

### Datasets

This project uses two large-scale chest X-ray datasets:

- NIH ChestX-ray14
- CheXpert

Raw data is stored under `data/raw/` and tracked using DVC. Dataset source information are documented in `data/provenance/`.

### Manifest 

A unified master manifest is constructed to index all images and metadata across datasets. The manifest:

- Standardizes schemas across NIH and CheXpert
- Harmonizes a subset of clinically common labels (atelectasis, cardiomegaly, consolidation, edema and effusion)
- Defers task-specific label resolution and splitting decisions to downstream stages.

It is generated via: `python src/data/build_manifests.py --filename master_manifest`, and its output is found in `data/manifests/master_manifest.csv`