# EXP-004 — R2L Within-Category Label-Shift Analysis

| Field | Value |
|-------|-------|
| **Experiment ID** | EXP-004 |
| **Title** | R2L Within-Category Label-Shift Analysis |
| **Phase** | Within-Category Composition & Label Shift Analysis |
| **Status** | PENDING (run `python experiments/EXP-004/run.py` to execute) |
| **Depends On** | EXP-001 (Taxonomy & Audit), EXP-002 (Binary Baseline), EXP-003 (Category Sensitivity) |
| **Research Question** | RQ4: How does binary intrusion-detection performance vary across R2L attack labels according to their representation in the training data? |
| **Hypothesis** | H4: R2L attack labels with greater representation in the training data will generally show higher detection rates than R2L labels with limited or no training representation. |

---

## Motivation

In EXP-003, Remote-to-Local (R2L) attacks exhibited severe detection degradation across all baseline models (Logistic Regression: 1.18%, Random Forest: 5.55%, Decision Tree: 21.49%).

Subsequent data audit revealed an extreme within-category composition shift between training and test sets:
- **Training R2L (995 records):** 89.45% composed of `warezclient` (890 records), which has **0 records** in the test set.
- **Testing R2L (2,885 records):** Dominated by `guess_passwd` (1,231 records / 42.67%) and `warezmaster` (944 records / 32.72%), which had only 53 and 20 training records, alongside 7 test-only attack types (686 records / 23.78%).

EXP-004 disaggregates the R2L category result by individual attack label to evaluate how detection rate relates to training representation.

---

## Key Methodological Decisions

1. **Follow-Up Analysis:** EXP-004 is a granular subgroup audit of the binary detectors from EXP-002/EXP-003, not an independent new modeling benchmark.
2. **Three Representation Groups:**
   - **Group A (Shared):** Present in both KDDTrain+ and KDDTest+ (`ftp_write`, `guess_passwd`, `imap`, `multihop`, `phf`, `warezmaster`).
   - **Group B (Test-Only):** Present in KDDTest+ but absent from KDDTrain+ (`httptunnel`, `named`, `sendmail`, `snmpgetattack`, `snmpguess`, `xlock`, `xsnoop`).
   - **Group C (Train-Only):** Present in KDDTrain+ but absent from KDDTest+ (`warezclient`, `spy`). No test records exist; reported for composition documentation only.
3. **No Causal Claims:** Lower training representation is evaluated for observational association; causality cannot be claimed due to confounding attack-specific feature signatures.
4. **Primary Metric:** Detection Rate (Recall) = $TP / N$ for each R2L label.

---

## Outputs

```
results/
├── r2l_label_detection_results.csv
├── r2l_representation_groups.csv
├── experiment_report.md
├── figures/
│   ├── r2l_label_detection_rate.png
│   └── r2l_train_vs_test_composition.png
└── metrics/
    └── r2l_metrics.json
```

---

## Reproducibility

To run the experiment:
```bash
python experiments/EXP-004/run.py
```

*SentinelNet Research Project — EXP-004*
