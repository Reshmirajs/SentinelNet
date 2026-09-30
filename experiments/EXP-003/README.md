# EXP-003 — Minority Attack Category Sensitivity

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-003 |
| **Title** | Minority Attack Category Sensitivity |
| **Phase** | Category Sensitivity & Subgroup Analysis |
| **Status** | PENDING (run `python experiments/EXP-003/run.py` to execute) |
| **Depends On** | EXP-001 (Taxonomy & Distribution), EXP-002 (Supervised Binary Baseline) |
| **Research Question** | How does binary intrusion-detection performance vary across DoS, Probe, R2L, and U2R attack categories? |
| **Hypothesis** | Detection performance will be lower for the less represented attack categories, particularly R2L and U2R, than for the more represented DoS and Probe categories. |

---

## Purpose

EXP-003 conducts a post-hoc subgroup sensitivity audit of the binary Normal-vs-Attack models
developed in EXP-002 (Logistic Regression, Decision Tree, and Random Forest).

The objective is to determine whether aggregate binary performance metrics (Accuracy, Detection Rate, F1)
mask severe vulnerabilities on attack categories with minimal representation in the training data
(specifically R2L and U2R).

---

## Key Methodological Decisions

1. **Strictly Binary Model:**  
   The models are trained exclusively on the binary target (`Normal = 0`, `Attack = 1`).  
   This is **not** a multiclass experiment. Attack categories are evaluated as post-hoc ground-truth subgroups.

2. **No Artificial Rebalancing:**  
   EXP-003 measures the sensitivity of the binary detector as-is.  
   We strictly prohibit:
   - SMOTE / ADASYN
   - Oversampling / undersampling
   - `class_weight="balanced"`
   - Decision threshold adjustments
   - Model selection or hyperparameter tuning based on KDDTest+

3. **Subgroups Analyzed:**  
   - **DoS** (Denial of Service)
   - **Probe** (Surveillance and probing)
   - **R2L** (Remote to Local unauthorized access)
   - **U2R** (User to Root local privilege escalation)  
   *Normal traffic is not an attack category and is excluded from attack-category detection rate calculations.*

---

## Inputs

| File | Path | Role |
|------|------|------|
| Training set | `data/raw/KDDTrain+.txt` | Model training & representation measurement |
| Test set | `data/raw/KDDTest+.txt` | Post-hoc category evaluation (quarantined) |

---

## Outputs

```
results/
├── category_distribution.csv
├── category_detection_results.csv
├── experiment_report.md
├── figures/
│   ├── attack_category_detection_rate.png
│   └── training_representation_vs_detection.png
└── metrics/
    ├── category_distribution.json
    └── category_metrics.json
```

---

## Reproducibility

To reproduce:
```bash
python experiments/EXP-003/run.py
```

*SentinelNet Research Project — EXP-003*
