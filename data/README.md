# SentinelNet — Dataset Documentation

> **IMPORTANT:** Raw dataset files are **never committed** to this repository.  
> This directory organizes data used in SentinelNet research experiments.

---

## 1. Overview

The primary benchmark evaluated in SentinelNet is **NSL-KDD**, an established benchmark derived from the KDD Cup 1999 dataset. Due to file size and licensing constraints, raw dataset files are **not included** in this repository.

Researchers wishing to inspect or reproduce the experiments must obtain the dataset files independently and place them in the local directory structure as described below.

---

## 2. Dataset Provenance

| Field | Description |
|---|---|
| **Dataset Name** | NSL-KDD |
| **Originating Institution** | Canadian Institute for Cybersecurity, University of New Brunswick (UNB) |
| **Documented URL** | https://www.unb.ca/cic/datasets/nsl.html |
| **Citation** | Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). *A detailed analysis of the KDD CUP 99 data set*. IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA). |

---

## 3. Required Local Files

The experimental pipeline in SentinelNet (EXP-001 through EXP-005) requires the following two raw dataset files:

| File Name | Expected Location | Description | Expected Rows | SHA-256 Checksum |
|---|---|---|---:|---|
| `KDDTrain+.txt` | `data/raw/KDDTrain+.txt` | Primary training dataset | 125,973 | `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95` |
| `KDDTest+.txt` | `data/raw/KDDTest+.txt` | Official test benchmark | 22,544 | `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84` |

---

## 4. Directory Organization and Immutability

The `data/` directory is organized into three tiers:

```
data/
├── README.md               # This documentation file (tracked in Git)
├── raw/                    # Raw source files (READ-ONLY, never committed)
│   ├── KDDTrain+.txt       # (User-supplied, gitignored)
│   └── KDDTest+.txt        # (User-supplied, gitignored)
├── interim/                # Partially processed or intermediate data (gitignored)
└── processed/              # ML-ready features (gitignored)
```

### Immutability Rule
- **`data/raw/` is strictly READ-ONLY.**
- Raw data files must **never be modified, overwritten, or edited**.
- All cleaning, transformation, and feature extraction steps must be performed programmatically via reproducible scripts without altering raw input files.
- `.gitignore` is configured to exclude all files in `data/raw/`, `data/interim/`, and `data/processed/` except directory `.gitkeep` markers and this documentation file.

---

## 5. Verification

Before running experiments, verify the integrity of the downloaded files using SHA-256:

### Linux / macOS
```bash
sha256sum data/raw/KDDTrain+.txt data/raw/KDDTest+.txt
```

### Windows (PowerShell)
```powershell
Get-FileHash data/raw/KDDTrain+.txt, data/raw/KDDTest+.txt -Algorithm SHA256
```

The resulting hashes should match the checksums listed in Section 3 above. Once verified, the experiment pipeline and test suite can be run directly.
