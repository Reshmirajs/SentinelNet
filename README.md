# SentinelNet: Subgroup Sensitivity and Operating-Point Analysis in Supervised Network Intrusion Detection

[![Status](https://img.shields.io/badge/status-experiments--completed-success)](experiments/registry.csv)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

---

## 1. Project Overview

**SentinelNet** is an academic research repository investigating the behavior of supervised machine learning classifiers for Network Intrusion Detection Systems (NIDS) on the NSL-KDD benchmark.

In published NIDS literature, classifiers are frequently evaluated and compared using single aggregate metrics such as overall accuracy, F1 score, or ROC-AUC. The central objective of SentinelNet is to systematically investigate whether and how these aggregate metrics conceal critical performance phenomena:
- Severe subgroup heterogeneity across distinct attack categories (DoS, Probe, R2L, U2R).
- The impact of novel attack labels present in test data but absent during training.
- Within-category composition shift and the relationship between training representation and per-label detection.
- Decision threshold sensitivity and the generalization of operating points from validation to test partitions.

> **Research Scope & Non-Claims:**  
> SentinelNet is an academic research evaluation, **not** a production security tool. This work does **not** claim real-time detection performance, production readiness, zero-day threat detection, or state-of-the-art classifier superiority. No single model is designated as "best"; observed trade-offs and performance differences are reported objectively.

---

## 2. Research Status

All five planned experiments (**EXP-001 through EXP-005**) have been executed, independently audited, verified against a 181-item test suite, and frozen in [`experiments/registry.csv`](experiments/registry.csv).

The complete research report synthesized from these frozen results is available in [`paper/sentinelnet_research_report.md`](paper/sentinelnet_research_report.md).

| Experiment ID | Title | Status | Primary Focus |
|---|---|---|---|
| **EXP-001** | NSL-KDD Dataset Acquisition & Leakage Audit | COMPLETE | Dataset statistics, data leakage, duplicate analysis, distribution shift |
| **EXP-002** | Supervised Baseline Binary Intrusion Detection | COMPLETE | LR, DT, and RF baselines; cohort isolation (duplicates, known vs. novel attacks) |
| **EXP-003** | Minority Attack Category Sensitivity | COMPLETE | Post-hoc category disaggregation (DoS, Probe, R2L, U2R); Wilson CIs for U2R |
| **EXP-004** | R2L Within-Category Label Shift Analysis | COMPLETE | Evaluation of per-label representation vs. detection rate within R2L |
| **EXP-005** | Threshold / Operating-Point Analysis | COMPLETE | Continuous probability sweeps, validation-based threshold selection, operating-point trade-offs |

---

## 3. Dataset & Partitioning

The experiments evaluate the **NSL-KDD** benchmark (University of New Brunswick — Canadian Institute for Cybersecurity; Tavallaee et al., 2009).

### Dataset Setup & Hygiene
Raw dataset files are **not committed** to this repository. See [`data/README.md`](data/README.md) for full acquisition details, expected checksums, and placement guidelines.

- **`KDDTrain+.txt`** (125,973 records, SHA-256: `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95`)
- **`KDDTest+.txt`** (22,544 records, SHA-256: `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84`)

### Partitioning & Training Isolation
- **Training partition (80% of KDDTrain+):** 100,778 records. All preprocessing transformers (`OneHotEncoder`, `StandardScaler`) and model parameters are fitted exclusively on this partition.
- **Validation partition (20% of KDDTrain+):** 25,195 records. Used strictly for internal validation and threshold selection in EXP-005.
- **Held-out test benchmark:** `KDDTest+` (22,544 records). Kept completely quarantined from model fitting and threshold selection, used only for final evaluation.

---

## 4. Key Experimental Findings

### EXP-001: Dataset Audit
- Identified **610 exact cross-split duplicate records** (2.71% of test set), which inflate test accuracy by ~0.6 pp if unaddressed.
- Identified **17 test-only attack labels** absent from training data.
- Identified **58 feature-identical records with conflicting ground-truth labels** (51 binary Normal-vs-Attack conflicts representing an inherent Bayes error limit).
- Excluded label-derived metadata (`difficulty`) to prevent label leakage, and removed 1 constant feature (`num_outbound_cmds`).
- Documented massive category distribution shift in R2L: 0.79% of training data vs. 12.80% of test data (+12.01 pp).

### EXP-002: Baseline Binary Classification
- At standard default thresholds (θ = 0.50), classifiers achieve aggregate accuracy between 75.36% and 78.73%, and F1 between 74.24% and 78.76%.
- Detection rates degrade sharply on novel attack labels compared to known attack labels:
  - Logistic Regression: 71.25% (known) vs. 40.83% (novel) [Δ = −30.43 pp]
  - Decision Tree: 74.89% (known) vs. 55.63% (novel) [Δ = −19.26 pp]
  - Random Forest: 76.79% (known) vs. 26.29% (novel) [Δ = −50.50 pp]

### EXP-003: Attack Category Sensitivity
- Detection capability is highly uneven across categories:
  - **DoS:** 80.16%–83.70% detection across models.
  - **Probe:** 74.85%–82.86% detection across models.
  - **R2L:** 1.18%–21.49% detection across models.
  - **U2R:** 13.43%–26.87% detection across models (*N* = 67; 95% Wilson CIs: e.g., Decision Tree [17.72%, 38.52%], Random Forest [7.23%, 23.60%]).
- High aggregate F1 scores conceal near-zero detection on R2L.

### EXP-004: R2L Disaggregation
- Evaluated whether per-label training counts explain per-label detection within R2L.
- The hypothesis that higher training sample count produces higher detection rate is **not supported**:
  - `snmpguess` (0 training samples, test-only): Decision Tree achieves **99.09% detection** (328/331).
  - `guess_passwd` (53 training samples, shared): Decision Tree and Random Forest achieve **0.00% detection** (0/1,231).
- Training sample volume alone does not determine detection capability; feature-space overlap with learned boundaries is a plausible contributing factor.

### EXP-005: Threshold / Operating-Point Analysis
- Operating points selected on the validation partition (highest θ achieving validation DR ≥ 90%) selected θ = 0.94 for Logistic Regression, θ = 1.00 for Decision Tree, and θ = 1.00 for Random Forest.
- A substantial validation-to-test generalization gap was observed: Logistic Regression achieved 90.51% DR on validation but 55.29% on KDDTest+ at the same threshold.
- Distribution shift and test-only attack labels are plausible contributing factors that EXP-005 does not causally isolate.

---

## 5. Installation & Reproduction

### Prerequisites
- Python 3.10 or higher (experiments developed and verified on Python 3.14.2)
- Virtual environment tool (`venv` or `conda`)

### Step 1: Clone Repository
```bash
git clone https://github.com/Reshmirajs/SentinelNet.git
cd SentinelNet
```

### Step 2: Set Up Virtual Environment
```bash
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Place Raw Datasets
Download `KDDTrain+.txt` and `KDDTest+.txt` from the Canadian Institute for Cybersecurity and place them in `data/raw/`:
```
data/raw/KDDTrain+.txt
data/raw/KDDTest+.txt
```
Verify hashes match [`data/README.md`](data/README.md).

### Step 5: Run the Verification Test Suite
```bash
python -m pytest
```
All 181 unit and reproducibility tests should pass.

### Step 6: Inspect or Reproduce Individual Experiments
Each experiment is self-contained with its own configuration, execution script, and outputs:
```bash
python experiments/EXP-001/run.py
python experiments/EXP-002/run.py
python experiments/EXP-003/run.py
python experiments/EXP-004/run.py
python experiments/EXP-005/run.py
```

---

## 6. Repository Layout

```
sentinelnet/
├── README.md                      # Academic project documentation
├── requirements.txt               # Pinned Python package dependencies
├── .gitignore                     # Excludes datasets, checkpoints, and caches
│
├── configs/                       # Experiment configuration templates
├── data/
│   ├── README.md                  # Provenance, SHA-256 hashes, placement instructions
│   ├── raw/                       # Place KDDTrain+.txt and KDDTest+.txt here (gitignored)
│   ├── interim/                   # Regenerable intermediate artifacts (gitignored)
│   └── processed/                 # Regenerable processed feature data (gitignored)
│
├── experiments/
│   ├── registry.csv               # Master status and audit tracking log
│   ├── EXP-001/                   # Dataset audit script, schema, and report
│   ├── EXP-002/                   # Baseline binary classifiers & cohort analysis
│   ├── EXP-003/                   # Attack-category sensitivity & Wilson CIs
│   ├── EXP-004/                   # R2L label-level shift & disaggregation
│   └── EXP-005/                   # Threshold sweep & operating-point analysis
│
├── paper/
│   └── sentinelnet_research_report.md  # Comprehensive research manuscript
│
├── results/                       # Top-level publication summaries (gitkeep)
├── src/                           # Shared modular package scaffolding
└── tests/                         # Full automated test suite (181 tests)
```

---

## 7. Important Research Limitations

1. **Benchmark Age and Simulation:** NSL-KDD derives from 1998 simulated network traffic (DARPA/KDD Cup 99). It does not reflect modern network topologies, protocols, or threat environments.
2. **Exact Duplicates:** 610 cross-split duplicates remain in the benchmark, introducing minor optimistic bias in standard evaluations.
3. **Bayes Error Bound:** 51 binary conflicting-label records prevent any deterministic classifier from achieving 100% test accuracy on those specific instances.
4. **Small Sample Uncertainty:** The U2R test cohort contains only *N* = 67 records. U2R detection rates have broad confidence intervals and must not be interpreted with spurious precision.
5. **Observational Scope:** EXP-003 and EXP-004 are post-hoc analyses that identify patterns but do not causally prove underlying mechanisms.
6. **No Real-Time or Deployment Inference:** All experiments evaluate static batch connection records under controlled conditions.

---

## 8. License

This project is licensed under the MIT License.
