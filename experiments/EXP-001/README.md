# EXP-001 — NSL-KDD Dataset Acquisition and Leakage Audit

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-001 |
| **Phase** | Data Audit |
| **Status** | PENDING (run `python experiments/EXP-001/run.py` to execute) |
| **Research Question** | What are the statistical properties of the NSL-KDD train/test sets, and are there data-leakage risks that would invalidate downstream evaluations? |

---

## Purpose

This experiment performs a rigorous, reproducible audit of the raw NSL-KDD
dataset files **before any preprocessing or model training**. It establishes a
ground-truth baseline for all subsequent experiments.

No models are trained. No data is normalized, encoded, scaled, or resampled.
All reported statistics are computed directly from the raw files.

---

## Inputs

| File | Path |
|------|------|
| Training set | `data/raw/KDDTrain+.txt` |
| Test set | `data/raw/KDDTest+.txt` |

> `data/raw/` is **read-only** per the Research Constitution.

---

## What the Audit Covers

1. **Feature schema** — names, types, and roles of all 43 columns (41 features + label + difficulty)
2. **Data quality** — missing values, duplicate rows, constant features, data-type mismatches
3. **Class distribution** — label counts and percentages for both splits
4. **Attack category mapping** — grouping fine-grained labels into the 5 NSL-KDD super-categories
5. **Train/test overlap** — exact-row and label-set comparisons
6. **Leakage risk assessment** — difficulty column, label distribution shift, feature cardinality anomalies
7. **Numerical summaries** — per-feature descriptive statistics

---

## Outputs

All outputs are saved to `results/`:

```
results/
├── figures/
│   ├── train_class_distribution.png
│   ├── test_class_distribution.png
│   ├── train_test_label_comparison.png
│   ├── difficulty_distribution.png
│   └── feature_types_overview.png
├── metrics/
│   ├── summary_statistics.json
│   ├── class_distribution_train.json
│   ├── class_distribution_test.json
│   ├── data_quality_report.json
│   └── leakage_risks.json
├── tables/
│   ├── feature_schema.csv
│   ├── descriptive_stats_train.csv
│   └── descriptive_stats_test.csv
└── audit_report.md
```

---

## How to Reproduce

```bash
# From the repository root:
python experiments/EXP-001/run.py

# Optionally specify custom data paths:
python experiments/EXP-001/run.py \
    --train data/raw/KDDTrain+.txt \
    --test  data/raw/KDDTest+.txt \
    --seed  42
```

---

## Dependencies

```
pandas >= 2.0
numpy  >= 1.24
matplotlib >= 3.7
scipy >= 1.11
```

Install: `pip install -r requirements.txt`

---

## Configuration

See [`experiment.yaml`](experiment.yaml) for the full reproducibility config.

---

## Reproducibility Notes

- Random seed: `42` (used only if any sampling operation is introduced)
- Raw data files are **never modified**
- Results are deterministic given the same input files
- SHA-256 checksums of input files are recorded in `results/metrics/summary_statistics.json`

---

*SentinelNet Research Team — EXP-001*
