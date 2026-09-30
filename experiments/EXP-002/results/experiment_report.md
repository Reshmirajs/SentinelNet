# EXP-002 — Supervised Baseline Binary Intrusion Detection

*Generated: 2026-09-30T13:30:57Z UTC*  
*Random seed: 42*

> **STATUS:** COMPLETE — All metrics computed from actual model outputs.


## Research Question


What baseline binary classification performance (Normal vs. Attack) do Logistic Regression, Decision Tree, and Random Forest achieve under train-only preprocessing and model-fitting isolation on NSL-KDD (while acknowledging that the KDDTest+ benchmark contains 610 exact cross-split duplicate records)? How does detection rate degrade on novel attack types? Do exact train/test duplicate samples inflate reported performance?


## Dataset

| File | Rows | SHA-256 |
| :--- | ---: | :--- |
| KDDTrain+.txt | 125,973 | 1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95 |
| KDDTest+.txt | 22,544 | fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84 |


### Training Partition & Quarantine
- **KDDTrain+ Total Records:** 125,973
- **Internal Stratified Training Partition (80%):** 100,778 rows (used to fit preprocessing and models)
- **Internal Stratified Validation Partition (20%):** 25,195 rows (used for internal validation sanity check)
- **Model Fitting:** Baseline models were fitted strictly on the 80% partition (100,778 rows); models were not refitted on 100% of KDDTrain+.
- **Test Quarantine:** KDDTest+ (22,544 rows) remained completely held out for final evaluation.


## Evaluation Cohort Sizes

| Cohort | N | Description |
| :--- | ---: | :--- |
| full_test | 22,544 | Official primary benchmark (all KDDTest+ samples) |
| non_overlapping | 21,934 | Test records with no exact full-vector duplicate in KDDTrain+ |
| overlapping | 610 | Exact duplicates of training records (LEAK-003) |
| known_attacks | 9,083 | Test attacks with label seen in training |
| novel_attacks | 3,750 | Test attacks with label NOT seen in training |


> **Note:** The full KDDTest+ result is reported as the primary benchmark for literature comparison, while the non-overlapping cohort isolates performance after removing exact feature-and-label duplicates (the 610 exact train/test duplicate samples).


## Primary Results — Full KDDTest+

| Model | Accuracy | Precision | Detection Rate | FAR | F1 | ROC-AUC | Train Time (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 75.36% | 91.70% | 62.36% | 7.46% | 74.24% | 0.7973 | 4.87 |
| Decision Tree | 78.73% | 91.27% | 69.26% | 8.75% | 78.76% | 0.8028 | 3.04 |
| Random Forest | 77.22% | 96.79% | 62.04% | 2.72% | 75.61% | 0.9535 | 8.78 |


## Overlap Cohort Analysis

| Model | Full KDDTest+ Acc. | Non-Overlapping Acc. | Overlapping Acc. | Δ (Overlap − Non-Overlap) |
| :--- | ---: | ---: | ---: | ---: |
| Logistic Regression | 75.36% | 74.77% | 96.72% | +21.95pp |
| Decision Tree | 78.73% | 78.16% | 99.34% | +21.19pp |
| Random Forest | 77.22% | 76.59% | 99.67% | +23.08pp |


## Known vs. Novel Attack Detection Rates

| Model | Known Attack DR | Novel Attack DR | Δ (Known − Novel) |
| :--- | ---: | ---: | ---: |
| Logistic Regression | 71.25% | 40.83% | +30.43pp |
| Decision Tree | 74.89% | 55.63% | +19.26pp |
| Random Forest | 76.79% | 26.29% | +50.50pp |


## Novel Attack Detection by Category


### Logistic Regression

| Category | N Samples | Detection Rate | Attack Labels |
| :--- | ---: | ---: | :--- |
| DOS | 1,719 | 42.58% | apache2, mailbomb, processtable, udpstorm, worm |
| PROBE | 1,315 | 59.85% | mscan, saint |
| R2L | 686 | 1.31% | httptunnel, named, sendmail, snmpgetattack, snmpguess, xlock, xsnoop |
| U2R | 30 | 10.00% | ps, sqlattack, xterm |


### Decision Tree

| Category | N Samples | Detection Rate | Attack Labels |
| :--- | ---: | ---: | :--- |
| DOS | 1,719 | 42.58% | apache2, mailbomb, processtable, udpstorm, worm |
| PROBE | 1,315 | 68.52% | mscan, saint |
| R2L | 686 | 64.58% | httptunnel, named, sendmail, snmpgetattack, snmpguess, xlock, xsnoop |
| U2R | 30 | 33.33% | ps, sqlattack, xterm |


### Random Forest

| Category | N Samples | Detection Rate | Attack Labels |
| :--- | ---: | ---: | :--- |
| DOS | 1,719 | 15.07% | apache2, mailbomb, processtable, udpstorm, worm |
| PROBE | 1,315 | 53.69% | mscan, saint |
| R2L | 686 | 2.33% | httptunnel, named, sendmail, snmpgetattack, snmpguess, xlock, xsnoop |
| U2R | 30 | 16.67% | ps, sqlattack, xterm |


## Reproducibility

| Item | Value |
| :--- | :--- |
| Experiment ID | EXP-002 |
| Random seed | 42 |
| Execution timestamp | 2026-09-30T13:30:57Z |
| Python | 3.14.2 |
| Platform | Windows-11-10.0.26200-SP0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.0 |
| numpy | 2.4.2 |


## Limitations


1. Binary classification groups high-severity attacks (U2R/R2L) with high-volume attacks (DoS) — per-category detection rates provide finer resolution.

2. `num_outbound_cmds` dropped (zero-variance per EXP-001); 40 features used for modelling.

3. Results evaluate historical DARPA/KDD benchmark data (1998). Generalisation to modern network traffic is not implied.
