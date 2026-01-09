
## Data Management & Governance

### Datasets

This project uses two large-scale chest X-ray datasets:

- NIH ChestX-ray14
- CheXpert

Raw data is stored under `data/raw/` and tracked using DVC. Dataset source information are documented in `data/provenance/`. Other artifacts like manifests and splits are also tracked and can be found in `data/manifests/` and `data/splits/`.

### Manifest 

A unified master manifest is constructed to index all images and metadata across datasets. The manifest:

- Standardizes schemas across NIH and CheXpert
- Harmonizes a subset of clinically common labels (atelectasis, cardiomegaly, consolidation, edema and effusion)
- Defers task-specific label resolution and splitting decisions to downstream stages

It is generated via: `python src/data/build_manifests.py`, and its output is found in `data/manifests/master_manifest.csv`.

### Splits

Dataset splits are generated via: `python src/data/build_splits.py`, which assigns each image to one of three partitions: train, validation, or test. Splits are performed at the patient level to prevent data leakage across partitions, following standard practice in medical imaging research [1,2]. The default proportions are: Train 70%, Validation 20%, Test 10%.

The split process operates on the master manifest and produces a single CSV file that preserves all metadata, adding only a `split` column. No rows are modified or removed during this step.

The script exposes configurable parameters to control dataset composition without altering the underlying manifest, including:

- Handling of uncertain labels (e.g. CheXpert -1 values)
- Handling of missing label values
- Inclusion or exclusion of lateral views
- Random seed for reproducibility


## References

[1] Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, et al. (2018) Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. PLOS Medicine 15(11): e1002683. https://doi.org/10.1371/journal.pmed.1002683

[2]  Irvin, e al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 590-597. https://doi.org/10.1609/aaai.v33i01.3301590