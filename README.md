# SentinelNet

SentinelNet is an academic research project investigating supervised machine-learning-based network intrusion detection on the NSL-KDD benchmark, with a specific focus on benchmark dataset structure, subgroup sensitivity, cross-split generalization, and operating-point analysis.

**Status:** Research prototype / completed experimental evaluation

---

# Research Overview

Network Intrusion Detection Systems (NIDS) are critical components of contemporary network defense, and supervised machine learning classifiers are frequently proposed to identify malicious connections. However, in benchmark-based evaluations, classifiers are predominantly compared using aggregate summary statistics such as overall accuracy, macro-F1, or ROC-AUC.

While aggregate metrics provide a convenient high-level summary, they can obscure critical performance characteristics:
- High overall accuracy can coexist with near-zero detection on entire attack classes.
- Standard train/test benchmark splits often contain structural artifacts, including duplicate records and test-only attack labels.
- Category-level class imbalance and within-category label distributions can shift substantially between training and evaluation splits.
- Default decision thresholds (e.g., $\theta = 0.50$) represent arbitrary operating points that may not generalize across partitions.

SentinelNet investigates how three standard supervised classifiers—Logistic Regression, Decision Tree, and Random Forest—behave across the NSL-KDD benchmark when evaluated beyond aggregate summary metrics. The project is an empirical academic evaluation focused on benchmark validity and diagnostic analysis rather than a claim of operational deployment readiness.

The central research message is:

> **A single aggregate accuracy or F1 value does not fully characterize supervised NIDS behavior on NSL-KDD.**

---

# Research Questions

The research is organized around five empirical questions addressed sequentially by experiments EXP-001 through EXP-005:

1. **Benchmark Audit (EXP-001):** What are the structural and statistical properties of the NSL-KDD training and test splits, and what data-integrity risks (such as cross-split record duplication, feature leakage, or distribution shift) exist that affect evaluation validity?
2. **Supervised Baseline (EXP-002):** What binary detection performance do standard classifiers achieve under strict training-only preprocessing isolation on `KDDTest+`, and how does detection performance degrade on test-only (novel) attack labels compared to known attack labels?
3. **Attack-Category Sensitivity (EXP-003):** How does binary detection rate vary across the four individual attack categories—Denial of Service (DoS), Probe, Remote-to-Local (R2L), and User-to-Root (U2R)?
4. **Within-Category Label Shift (EXP-004):** Within the R2L category, does the number of training examples per attack label explain the observed detection performance on test instances of that label?
5. **Operating-Point Analysis (EXP-005):** How does adjusting the classification decision threshold affect the trade-off between Detection Rate and False Alarm Rate, and how well do validation-selected operating points generalize to `KDDTest+`?

---

# Dataset

The project evaluates the **NSL-KDD** benchmark (Canadian Institute for Cybersecurity, University of New Brunswick; Tavallaee et al., 2009):

- **`KDDTrain+.txt`:** 125,973 connection records
- **`KDDTest+.txt`:** 22,544 connection records

Each connection record contains 41 traffic features (3 categorical, 6 binary indicators, 32 numerical continuous/count features), 1 ground-truth attack label, and 1 difficulty score metadata value.

Raw dataset files are **not included** in this repository. Researchers wishing to inspect or reproduce the experiments must obtain the files independently and place them locally at:

```
data/raw/KDDTrain+.txt
data/raw/KDDTest+.txt
```

Refer to [`data/README.md`](data/README.md) for full dataset provenance, expected SHA-256 checksums, and placement guidelines.

---

# Dataset Audit

A comprehensive pre-modeling audit of the raw benchmark files (**EXP-001**) identified several structural properties that directly affect how experimental results must be interpreted:

- **Cross-Split Duplicates:** 610 test records (2.71% of `KDDTest+`) are exact feature-and-label duplicates of records in `KDDTrain+`. The overlap cohort exhibits substantially different classification performance from the non-overlapping cohort.
- **Novel Test Labels:** 17 attack labels appear exclusively in `KDDTest+` and are completely absent from `KDDTrain+`. Classifiers cannot have learned signatures for these labels from training data.
- **Feature-Identical Conflicting Records:** 58 records in `KDDTest+` share identical 41-dimensional feature vectors with records in `KDDTrain+` but carry conflicting ground-truth labels. 51 of these represent binary Normal-vs-Attack conflicts, establishing an inherent Bayes error floor for any deterministic classifier on those specific instances.
- **Category Distribution Shift:** The category proportions change markedly between splits. Most notably, Remote-to-Local (R2L) traffic constitutes only 0.79% (995 records) of `KDDTrain+`, but expands to 12.80% (2,885 records) of `KDDTest+` (+12.01 percentage points).
- **Metadata Exclusion:** The `difficulty` column reflects the number of classifiers in an earlier challenge that correctly classified the connection. Because it is directly derived from the label, it represents label leakage and is excluded from all feature matrices.
- **Constant Feature Exclusion:** The feature `num_outbound_cmds` has a constant value of zero across all training and test records. It carries zero variance and is removed during preprocessing.

These observations demonstrate that benchmark design characteristics—rather than model capacity alone—play a substantial role in observed test performance.

---

# Experimental Design

All experiments adhere to a common, reproducible evaluation protocol:

- **Target Formulation:** Strict binary classification distinguishing `Normal` (0) from `Attack` (1).
- **Classifiers:** Three standard model families:
  - *Logistic Regression:* `solver='lbfgs'`, `C=1.0`, `max_iter=1000`, `random_state=42`
  - *Decision Tree:* `criterion='gini'`, `max_depth=20`, `random_state=42`
  - *Random Forest:* `n_estimators=100`, `random_state=42`, `n_jobs=-1`
- **Data Partitioning:** `KDDTrain+` is split into an 80% training partition (100,778 records) and a 20% validation partition (25,195 records) using stratified random sampling with fixed seed 42.
- **Preprocessing Isolation:** All preprocessing transformations are fitted strictly on the 80% training partition only:
  - `difficulty` and `num_outbound_cmds` are dropped.
  - Categorical features (`protocol_type`, `service`, `flag`) are encoded using `OneHotEncoder(handle_unknown='ignore')`.
  - Numerical features are standardized using `StandardScaler`.
  - Binary indicator features are passed through unchanged.
- **Benchmark Quarantine:** `KDDTest+` (22,544 records) is quarantined as the held-out evaluation benchmark and is never exposed during preprocessing fitting, hyperparameter selection, or threshold tuning.
- **No Resampling:** No synthetic oversampling (such as SMOTE), undersampling, or artificial reweighting is applied in the frozen experimental pipeline.

---

# Experiments

## EXP-001 — Dataset Audit
- **Focus:** Data hygiene, leakage analysis, cross-split duplication, and distribution shifts.
- **Key Outcome:** Quantified 610 duplicate records, 17 test-only labels, 51 binary conflicting-label records, excluded `difficulty` and constant feature `num_outbound_cmds`, and established the dataset baseline for all downstream experiments.

## EXP-002 — Supervised Binary Baseline
- **Focus:** Evaluates standard binary classifiers on the full `KDDTest+` benchmark alongside disaggregated cohort evaluations (overlapping vs. non-overlapping records, known vs. novel attack labels).
- **Primary `KDDTest+` Results:**

| Model | Accuracy | Precision | Detection Rate | FAR | F1 | ROC-AUC | Train Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 75.36% | 91.70% | 62.36% | 7.46% | 74.24% | 0.7973 | 4.87 |
| Decision Tree | 78.73% | 91.27% | 69.26% | 8.75% | 78.76% | 0.8028 | 3.04 |
| Random Forest | 77.22% | 96.79% | 62.04% | 2.72% | 75.61% | 0.9535 | 8.78 |

These metrics illustrate distinct operating characteristics: Decision Tree achieves the highest default Detection Rate (69.26%) and F1 (78.76%), while Random Forest exhibits the lowest False Alarm Rate (2.72%) and highest ROC-AUC (0.9535). No model is designated as universally superior. Furthermore, detection rates drop substantially on novel attack labels compared to known attack labels (e.g., Random Forest detects 76.79% of known attacks but only 26.29% of novel attacks).

## EXP-003 — Attack-Category Sensitivity
- **Focus:** Post-hoc disaggregation of binary detection performance across the four NSL-KDD attack categories.
- **Category Detection Rates:**

| Attack Category | Test Samples (*N*) | Logistic Regression | Decision Tree | Random Forest |
|---|---:|---:|---:|---:|
| **DoS** | 7,460 | 81.92% | 83.70% | 80.16% |
| **Probe** | 2,421 | 76.17% | 82.86% | 74.85% |
| **R2L** | 2,885 | 1.18% | 21.49% | 5.55% |
| **U2R** | 67 | 20.90% | 26.87% | 13.43% |

Category-level evaluation reveals extreme performance divergence that is masked by overall accuracy and F1. While DoS and Probe attacks are detected at rates between 74% and 84%, R2L detection drops to between 1.18% and 21.49%. For U2R, the sample size is only *N* = 67 records, and 95% Wilson score confidence intervals confirm wide estimation uncertainty (e.g., [7.23%, 23.60%] for Random Forest and [17.72%, 38.52%] for Decision Tree); point estimates for U2R must be interpreted with caution.

## EXP-004 — R2L Within-Category Label Analysis
- **Focus:** Analyzes whether low R2L detection rates can be explained simply by training sample counts for individual attack labels.
- **R2L Label Observations:**
  - `guess_passwd`: 53 training examples, 1,231 test examples (dominant test attack)
  - `warezmaster`: 20 training examples, 944 test examples
  - `snmpguess`: 0 training examples, 331 test examples (novel, test-only)
- **Observed Decision Tree Detection Rates:**
  - `guess_passwd`: **0.00% detection** (0 / 1,231)
  - `snmpguess`: **99.09% detection** (328 / 331)

These observations indicate that training representation count alone does not explain the observed per-label detection differences. A classifier can fail completely on a label seen 53 times in training while successfully detecting over 99% of a label absent from the training set. Feature-space alignment with learned decision boundaries is a plausible contributing factor, though EXP-004 does not isolate causal mechanisms.

## EXP-005 — Operating-Point / Threshold Analysis
- **Focus:** Studies how adjusting the binary classification threshold $\theta$ across a 0.00–1.00 grid (step 0.01) alters the trade-off between Detection Rate and False Alarm Rate. Thresholds were selected using the validation partition only (highest threshold achieving validation DR $\ge$ 90%).
- **Operating-Point Comparison on `KDDTest+`:**

| Model | Selected $\theta$ | Default DR ($\theta=0.50$) | Default FAR | Selected DR | Selected FAR |
|---|:---:|---:|---:|---:|---:|
| Logistic Regression | 0.94 | 62.36% | 7.46% | 55.29% | 6.29% |
| Decision Tree | 1.00 | 69.26% | 8.75% | 69.25% | 8.75% |
| Random Forest | 1.00 | 62.04% | 2.72% | 44.42% | 0.79% |

The experiment demonstrates that threshold selection reconfigures the detection-rate / false-alarm trade-off, but operating points calibrated on the validation partition did not preserve the targeted detection rate when transferred to `KDDTest+`. For Logistic Regression, a threshold achieving 90.51% validation DR yielded only 55.29% on `KDDTest+`. This divergence highlights an operating-point generalization gap that should be considered when selecting operating thresholds for NIDS benchmarks.

---

# Key Findings

1. **Aggregate Metrics Conceal Subgroup Blind Spots:** Standard accuracy and F1 scores mask severe disparities across attack types. Classifiers achieving 75%–79% aggregate accuracy concurrently detect as few as 1%–21% of R2L attacks.
2. **Benchmark Duplication Influences Evaluation:** The 610 exact cross-split duplicate records represent an overlap cohort whose classification accuracy differs substantially from the non-overlapping portion of `KDDTest+`.
3. **Novel Attack Labels Challenge Generalization:** Detection rates degrade significantly on attack labels present in the test benchmark but absent from the training data across all evaluated classifiers.
4. **Training Sample Count Does Not Predict Detection:** Disaggregation of R2L demonstrates that training example volume alone does not determine detection rate: models can completely miss heavily represented shared attacks while detecting unseen test-only attacks.
5. **Operating Points Do Not Generalize Uniformly:** Tuning classification thresholds to meet a detection rate target on internal validation data produces different operating points on the held-out benchmark, illustrating the challenge of threshold selection under distribution shift.

---

# Reproducibility

### 1. Clone Repository
```bash
git clone https://github.com/Reshmirajs/SentinelNet.git
cd SentinelNet
```

### 2. Set Up Environment
```bash
python -m venv .venv

# Linux / macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Place Dataset Files
Obtain `KDDTrain+.txt` and `KDDTest+.txt` (see [`data/README.md`](data/README.md)) and place them in:
```
data/raw/KDDTrain+.txt
data/raw/KDDTest+.txt
```

### 5. Run Verification Test Suite
```bash
python -m pytest
```
All 181 automated tests verifying data schemas, baseline metrics, category breakdowns, R2L disaggregation, and threshold selections should pass.

### 6. Run Individual Experiments
Each experiment can be independently re-executed from repository root:
```bash
python experiments/EXP-001/run.py
python experiments/EXP-002/run.py
python experiments/EXP-003/run.py
python experiments/EXP-004/run.py
python experiments/EXP-005/run.py
```

---

# Repository Layout

```
sentinelnet/
├── README.md                      # Public research documentation (this file)
├── requirements.txt               # Pinned environment dependencies
├── .gitignore                     # Git exclusion rules
│
├── data/
│   ├── README.md                  # Provenance, hashes, and setup guide
│   └── raw/                       # Place KDDTrain+.txt and KDDTest+.txt here (gitignored)
│
├── experiments/
│   ├── registry.csv               # Master experiment tracking log
│   ├── EXP-001/                   # Dataset audit & leakage analysis
│   ├── EXP-002/                   # Baseline binary classifiers & cohort analysis
│   ├── EXP-003/                   # Attack-category sensitivity & Wilson CIs
│   ├── EXP-004/                   # Within-category R2L disaggregation
│   └── EXP-005/                   # Threshold sweep & operating-point analysis
│
├── paper/
│   └── sentinelnet_research_report.md  # Comprehensive academic research report
│
├── src/                           # Modular utility scaffolding
└── tests/                         # Full automated test suite (181 tests)
```

---

# License

This project is licensed under the MIT License. See `LICENSE` for details.
