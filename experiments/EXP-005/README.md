# EXP-005 — Threshold / Operating-Point Analysis

| Field | Value |
|---|---|
| **Experiment ID** | EXP-005 |
| **Phase** | Operating-Point Analysis |
| **Status** | COMPLETE |
| **Depends On** | EXP-002, EXP-003, EXP-004 |
| **Research Question** | How does the classification threshold affect the trade-off between attack detection rate and false-alarm rate for the SentinelNet binary classifiers? |

---

## Purpose

EXP-005 investigates the operating-point trade-off between attack Detection Rate (Recall) and False Alarm Rate (FAR) for the three binary classifiers trained in EXP-002 (Logistic Regression, Decision Tree, Random Forest) on the NSL-KDD benchmark.

The experiment:
- Sweeps continuous attack probability scores over a 0.00–1.00 threshold grid (step 0.01; 101 candidate thresholds).
- Selects an operating-point threshold for each model using an internal validation partition only (highest threshold achieving validation DR ≥ 90%).
- Keeps `KDDTest+` quarantined until thresholds are frozen.
- Evaluates the frozen thresholds on `KDDTest+` and compares them against the default θ = 0.50 baseline.
- Documents the validation-to-test generalization gap observationally.

---

## Inputs

| File | Path | Role |
|---|---|---|
| Training set | `data/raw/KDDTrain+.txt` | 80% train partition (fitting), 20% validation partition (threshold selection) |
| Test set | `data/raw/KDDTest+.txt` | Held-out benchmark evaluation (quarantined during selection) |

---

## Outputs

```
results/
├── figures/
│   ├── threshold_vs_detection_far.png
│   ├── roc_curve_thresholds.png
│   └── default_vs_selected_test_metrics.png
├── metrics/
│   └── exp005_metrics.json
├── tables/
│   ├── validation_threshold_sweep.csv
│   ├── selected_thresholds.csv
│   └── threshold_operating_point_results.csv
└── experiment_report.md
```

---

## How to Reproduce

```bash
python experiments/EXP-005/run.py
```

---

## Configuration

See [`experiment.yaml`](experiment.yaml) for the full configuration.
