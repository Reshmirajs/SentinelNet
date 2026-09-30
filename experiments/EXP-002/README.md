# EXP-002 — Supervised Baseline Binary Intrusion Detection and Generalization Audit

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-002 |
| **Phase** | Supervised Baseline |
| **Status** | PENDING (run `python experiments/EXP-002/run.py` to execute) |
| **Depends On** | EXP-001 |
| **Research Question** | What baseline binary classification performance do Logistic Regression, Decision Tree, and Random Forest achieve on NSL-KDD? Do novel attack types and cross-split duplicates affect reported performance? |

---

## Purpose

EXP-002 trains and evaluates three canonical supervised machine-learning models for binary
network intrusion detection (`Normal` vs. `Attack`) on the NSL-KDD benchmark.

The experiment:
- Establishes a rigorous, reproducible performance baseline for a college conference poster.
- Evaluates the full `KDDTest+` as the primary benchmark (consistent with published literature).
- Reports disaggregated results for the 610 exact duplicate samples and the 21,934 non-overlapping samples (from EXP-001 LEAK-003).
- Reports disaggregated detection rates for known vs. novel (test-only) attack types (from EXP-001 LEAK-002).

No model selection uses the test set. No data is fabricated.

---

## Inputs

| File | Path | Role |
|------|------|------|
| Training set | `data/raw/KDDTrain+.txt` | Training & internal validation |
| Test set | `data/raw/KDDTest+.txt` | Primary evaluation (quarantined until evaluation) |

> `data/raw/` is **read-only** per the Research Constitution.

---

## Target Definition

| Value | Meaning |
|-------|---------|
| `0` | Normal — `label == "normal"` |
| `1` | Attack — any other label |

---

## Models

| Model | Class |
|-------|-------|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` |
| Decision Tree | `sklearn.tree.DecisionTreeClassifier` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` |

---

## Preprocessing Pipeline (Train-Partition Fitted Only)

1. Drop `difficulty` — metadata, never a feature (LEAK-001).
2. Drop `num_outbound_cmds` — zero-variance constant (LEAK-004).
3. One-Hot Encode `protocol_type`, `service`, `flag`.
4. StandardScale continuous/count features.
5. Pass binary features (0/1 integers) through unchanged.

---

## Evaluation Cohorts

| Cohort | Description |
|--------|-------------|
| Full `KDDTest+` | Official benchmark (contains overlap and novel samples) |
| Non-overlapping | Test rows with no exact feature+label match in train |
| Overlapping | Test rows that exactly duplicate a training record |
| Known attacks | Test attacks whose label exists in the training label set |
| Novel attacks | Test attacks whose label does NOT exist in the training label set |

---

## Outputs

```
results/
├── figures/
│   ├── poster_model_comparison_bar.png
│   ├── known_vs_novel_detection_rate.png
│   ├── overlap_inflation_comparison.png
│   └── roc_curves_comparison.png
├── metrics/
│   ├── summary_results.json
│   └── cohort_breakdowns.json
├── tables/
│   ├── poster_main_results_table.csv
│   ├── overlap_breakdown_table.csv
│   └── novel_attacks_by_category.csv
└── experiment_report.md
```

---

## How to Reproduce

```bash
python experiments/EXP-002/run.py
```

---

## Dependencies

```
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
scikit-learn >= 1.3
```

Install: `pip install -r requirements.txt`

---

## Configuration

See [`experiment.yaml`](experiment.yaml) for the full reproducibility config.

---

## Reproducibility Notes

- Random seed: `42` throughout (data splitting, all model estimators).
- Preprocessing is fitted strictly on the training partition only.
- `KDDTest+` is never accessed during preprocessing fitting, model training, or model selection.
- SHA-256 checksums of input files are logged in results.

---

*SentinelNet Research Team — EXP-002*
