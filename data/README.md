# SentinelNet — Dataset Documentation

> **IMPORTANT:** Raw data files are **never committed** to this repository.  
> This file documents dataset provenance, acquisition instructions, and data organization.

---

## Overview

This directory organizes all data used in SentinelNet research experiments. It is divided into three tiers:

| Directory | Contents | Committed? |
|-----------|----------|------------|
| `raw/` | Original, unmodified datasets | **NO** — gitignored |
| `interim/` | Partially processed data | **NO** — gitignored |
| `processed/` | Final ML-ready feature matrices | **NO** — gitignored |

All directories are gitignored to prevent accidental data commits. Only this `README.md` is tracked.

---

## Datasets

> **STATUS: No datasets have been acquired yet.**

Datasets will be documented here as they are approved and downloaded by the research lead.

For each dataset, record:

```
### Dataset Name

- **Source:** [URL or citation]
- **Version/Date:** 
- **License:**
- **Size:** 
- **Description:** 
- **Features:** 
- **Classes/Labels:** 
- **Acquisition:** [Steps to obtain the data]
- **Placement:** data/raw/<subdirectory>/
- **SHA256 checksum:** [Verify integrity after download]
- **Citation:**
```

---

## Candidate Datasets (Under Consideration)

> **RESEARCH DECISION NEEDED:** The research lead must approve which datasets to use before acquisition.

The following datasets are commonly used in network intrusion detection research and may be considered:

- **KDD Cup 1999** — Classic benchmark; known class imbalance issues
- **NSL-KDD** — Improved version of KDD Cup 1999
- **UNSW-NB15** — Modern synthetic dataset with 9 attack categories
- **CIC-IDS-2017** — Canadian Institute for Cybersecurity; realistic traffic
- **CIC-IDS-2018** — Extended version of CIC-IDS-2017
- **CAIDA** — Real-world anonymized internet traffic

No dataset has been selected. No data has been downloaded.

---

## Data Organization Convention

Once a dataset is approved and downloaded, organize as follows:

```
data/
├── raw/
│   └── <dataset-name>/          # Unmodified source files
│       ├── <original-files>
│       └── CHECKSUMS.sha256
│
├── interim/
│   └── <dataset-name>/          # After cleaning, before feature engineering
│       ├── train.parquet
│       ├── val.parquet
│       └── test.parquet
│
└── processed/
    └── <dataset-name>/          # Final ML-ready feature matrices
        ├── X_train.parquet
        ├── X_val.parquet
        ├── X_test.parquet
        ├── y_train.parquet
        ├── y_val.parquet
        └── y_test.parquet
```

---

## Reproducibility Note

- Always verify file integrity with checksums after download.
- Document the exact download date and source URL.
- Any preprocessing applied to raw data must be scripted in `src/preprocessing/` — never manual.
- The split strategy (train/val/test ratios, stratification) must be recorded in the relevant experiment config.

---

*Last updated: 2026-09-30 | SentinelNet Research Team*
