# SentinelNet: Subgroup Sensitivity and Operating-Point Analysis in Supervised Network Intrusion Detection on NSL-KDD

**Version:** 1.0 — Based on frozen experiments EXP-001 through EXP-005  
**Date:** 2026-09-30  
**Status:** Draft for review (not published)

---

## Abstract

Supervised binary classifiers for Network Intrusion Detection Systems (NIDS) are commonly evaluated using aggregate metrics such as accuracy, F1 score, or Area Under the Receiver Operating Characteristic Curve (ROC-AUC). This paper argues—and provides experimental evidence—that such aggregate measures are insufficient to characterize the behavior of these classifiers on NSL-KDD, a widely used intrusion detection benchmark. We present five sequential experiments (EXP-001 through EXP-005) on NSL-KDD using three standard supervised classifiers (Logistic Regression, Decision Tree, and Random Forest), trained on 80% of KDDTrain+ (100,778 samples) and evaluated on the held-out KDDTest+ benchmark (22,544 samples).

Our dataset audit (EXP-001) identifies 610 exact cross-split duplicate records, 17 attack labels present in KDDTest+ but not in KDDTrain+, 58 feature-identical records with conflicting ground-truth labels, and substantial category-level distribution shift—most severely a +12.01 percentage-point increase in R2L representation from training to test. Our baseline evaluation (EXP-002) shows that overall accuracy (75.36%–78.73%) and F1 (74.24%–78.76%) conceal severe subgroup variation. Attack-category analysis (EXP-003) reveals near-zero detection rates for R2L (1.18%–21.49%) and U2R (13.43%–26.87%, *N*=67), while DoS detection ranges from 80.16% to 83.70%. R2L disaggregation (EXP-004) demonstrates that training representation count alone does not explain per-label detection differences: the Decision Tree detects 99.09% of *snmpguess* test instances (0 training examples) while detecting 0.00% of *guess_passwd* (53 training examples), contradicting a simple more-training-data-equals-better-detection hypothesis. Threshold analysis (EXP-005) demonstrates that the trade-off between detection rate and false-alarm rate is sensitive to the decision threshold, with notable divergence between validation-partition-selected operating points and KDDTest+ outcomes—a gap that may reflect dataset characteristics documented in EXP-001, though EXP-005 does not causally isolate any contributing factor.

Our central finding is: **a single aggregate accuracy or F1 value does not adequately characterize supervised NIDS classifier behavior on NSL-KDD**. Subgroup evaluation, distribution shift documentation, and operating-point analysis are necessary complements to aggregate reporting.

**Keywords:** Network intrusion detection, NSL-KDD, binary classification, subgroup evaluation, threshold analysis, attack category sensitivity, benchmark evaluation, supervised learning.

---

## 1. Introduction

### 1.1 Background

A Network Intrusion Detection System (NIDS) monitors network traffic to identify potentially malicious activity. Supervised machine learning has been widely applied to NIDS, where the task is typically formulated as classifying each network connection record as either *Normal* or *Attack*. The NSL-KDD dataset, introduced by Tavallaee et al. [REFERENCE NEEDED: Tavallaee et al. 2009] as an improvement over the original KDD Cup 1999 benchmark [REFERENCE NEEDED: KDD Cup 99 dataset paper], has become a standard evaluation benchmark for this task.

Three performance characteristics receive particular attention in the NIDS literature:

- **Detection Rate (DR)**, also known as Recall or True Positive Rate: the fraction of actual attack records correctly classified as attacks.
- **False Alarm Rate (FAR)**, also known as the False Positive Rate: the fraction of normal records incorrectly classified as attacks.
- **Overall Accuracy and F1**: aggregate measures combining detection and false-alarm behavior across all classes.

In operational settings, DR and FAR are often more directly meaningful than aggregate accuracy, because a missed attack (false negative) or a flood of false alarms have distinct operational costs. The appropriate balance between DR and FAR depends on the deployment context.

### 1.2 Problem Statement

Most published evaluations on NSL-KDD report a single aggregate performance figure—typically accuracy, macro-F1, or ROC-AUC—for a model. This conceals several important phenomena:

1. **Subgroup heterogeneity:** NSL-KDD contains four attack categories (Denial of Service [DoS], Probe, Remote-to-Local [R2L], and User-to-Root [U2R]) with vastly different representation in the training and test sets. A classifier that achieves 75%+ accuracy may simultaneously achieve near-zero detection on an entire attack category.

2. **Novel attack labels:** KDDTest+ contains 17 attack labels that do not appear in KDDTrain+. The detection performance on these labels cannot be attributed to learned signatures.

3. **Distribution shift:** The proportional representation of categories differs substantially between KDDTrain+ and KDDTest+—most critically, R2L increases from 0.79% of training data to 12.80% of test data.

4. **Operating-point sensitivity:** Most evaluations use a fixed decision threshold of 0.50 applied to classifier output probabilities. Different thresholds yield different DR/FAR trade-offs, and the default threshold may not be meaningful for all classifiers or contexts.

### 1.3 Research Gap

Despite these known issues, many NSL-KDD evaluations in the literature [REFERENCE NEEDED: Survey of ML-based NIDS papers] focus on aggregate accuracy or F1 comparisons, without reporting per-category detection rates, addressing distribution shift, or examining threshold sensitivity. The internal composition of attack categories—and particularly R2L—has received less systematic attention. The gap between what aggregate metrics report and what per-subgroup analysis reveals is not fully characterized in existing literature.

### 1.4 Research Questions

This paper addresses five research questions, each aligned with one experiment:

- **RQ1:** What are the statistical properties and data-leakage risks of the NSL-KDD benchmark, and what constraints do they impose on evaluation validity?
- **RQ2:** What binary intrusion detection performance do standard supervised classifiers achieve on the full KDDTest+ benchmark, and how does detection degrade on novel attacks?
- **RQ3:** How does binary detection performance vary across the four NSL-KDD attack categories?
- **RQ4:** Within the R2L category, do per-label detection rates correlate with per-label training representation?
- **RQ5:** How does the binary classification threshold affect the trade-off between Detection Rate and False Alarm Rate?

### 1.5 Contributions

This paper makes the following contributions:

1. A reproducible, documented dataset audit of NSL-KDD quantifying exact cross-split overlap, novel test-only attack labels, feature-identical conflicting-label records, and category-level distribution shift.
2. A binary baseline evaluation across three standard classifiers, explicitly reporting per-cohort results including known-attack vs. novel-attack separation and overlap-cohort isolation.
3. A post-hoc subgroup analysis of per-category detection rates with 95% Wilson confidence intervals on the small U2R subgroup.
4. A within-category disaggregation of the R2L category that challenges a simple training-representation explanation for detection rate differences.
5. A threshold operating-point analysis with explicit documentation of a methodological correction made during the experiment, and observational reporting of the validation-to-test generalization gap.

---

## 2. Related Work

*Note: This section identifies the relevant literature areas. Verified bibliographic citations have not been independently confirmed for all referenced works and are marked accordingly.*

### 2.1 Network Intrusion Detection Systems

Network Intrusion Detection Systems classify network connections or packets as benign or malicious. Rule-based or signature-based systems require manually maintained rule databases and cannot detect novel attack patterns. Anomaly-based systems detect deviations from a baseline of normal behavior. Hybrid approaches combine both paradigms. This paper focuses exclusively on supervised binary classification in the style of anomaly-based detection, evaluated on a benchmark dataset.

[REFERENCE NEEDED: General NIDS survey paper]

### 2.2 Machine Learning for NIDS

Supervised ML approaches to NIDS have included decision trees, random forests, support vector machines, and more recently deep learning architectures. A survey of this literature would typically include [REFERENCE NEEDED: ML-based NIDS survey, e.g., Buczak & Guven 2016 or similar].

Key issues noted across the literature include: high variance of reported results across benchmark datasets; sensitivity to preprocessing choices; limited evaluation on real traffic; and the difficulty of reporting metrics that are meaningful for operational deployment.

### 2.3 NSL-KDD and Benchmark Limitations

NSL-KDD was proposed by Tavallaee et al. [REFERENCE NEEDED: Tavallaee et al. 2009, "A Detailed Analysis of the KDD CUP 99 Data Set"] as an improvement over KDD Cup 1999, which suffered from severe record redundancy that introduced measurement bias. NSL-KDD addressed this by removing exact duplicates within each split.

However, NSL-KDD retains several known limitations: (1) the data was captured in a 1998 simulated network environment and does not reflect modern traffic; (2) the training and test sets have different attack-category compositions; (3) 17 attack labels appear exclusively in the test set; (4) exact cross-split feature-and-label duplicates remain. Several authors have noted these limitations [REFERENCE NEEDED: papers critiquing NSL-KDD]. This paper documents them systematically via a dataset audit and reports their observed impact on evaluation.

### 2.4 Evaluation of Imbalanced Intrusion Detection Datasets

The NSL-KDD training set is imbalanced: DoS represents 36.46%, while U2R represents only 0.04% of training records. Class imbalance can cause classifiers to exhibit high aggregate accuracy while failing to detect minority-class examples. Standard remedies include SMOTE [REFERENCE NEEDED: Chawla et al. 2002] and cost-sensitive learning, but their use introduces additional methodological decisions. This paper does not apply any balancing technique; instead, it characterizes the impact of class imbalance through subgroup evaluation.

### 2.5 Operating-Point and Threshold Evaluation

Binary classifiers parameterized by a score threshold offer a continuous trade-off between detection rate and false-alarm rate, captured by the Receiver Operating Characteristic (ROC) curve. A single aggregate metric such as ROC-AUC summarizes this curve but does not characterize performance at any specific threshold. Operating-point analysis—evaluating a classifier at a specified or selected threshold—is relevant to operational NIDS deployment where the cost of false alarms and missed detections can differ substantially. [REFERENCE NEEDED: General paper on operating-point selection for classification]

---

## 3. Dataset and Experimental Setup

### 3.1 NSL-KDD

The NSL-KDD dataset was obtained from the Canadian Institute for Cybersecurity, University of New Brunswick. It consists of two files: KDDTrain+.txt and KDDTest+.txt, representing the training and evaluation benchmarks respectively.

| File | Rows | SHA-256 (first 16 hex) |
|---|---:|---|
| KDDTrain+.txt | 125,973 | `1b86d2f957b33082...` |
| KDDTest+.txt | 22,544 | `fa46b0935342616a...` |

### 3.2 KDDTrain+ / KDDTest+

Each record in NSL-KDD represents a single network connection and is described by 41 features plus a target label column and one metadata column (difficulty). The 41 features fall into three types: 3 categorical nominal features (`protocol_type`, `service`, `flag`), 6 binary integer features (`land`, `logged_in`, `root_shell`, `su_attempted`, `is_host_login`, `is_guest_login`), and 32 numerical features.

The target label is a multi-class attack type string (e.g., `neptune`, `guess_passwd`, `normal`). For all experiments in this paper, the task is binary classification: `label = "normal"` is encoded as 0 (Normal), and all other labels are encoded as 1 (Attack).

The class distribution differs substantially between splits:

| Category | KDDTrain+ Count | KDDTrain+ % | KDDTest+ Count | KDDTest+ % | Δ (pp) |
|---|---:|---:|---:|---:|---:|
| Normal | 67,343 | 53.46% | 9,711 | 43.08% | −10.38 |
| DoS | 45,927 | 36.46% | 7,460 | 33.09% | −3.37 |
| Probe | 11,656 | 9.25% | 2,421 | 10.74% | +1.49 |
| R2L | 995 | 0.79% | 2,885 | 12.80% | +12.01 |
| U2R | 52 | 0.04% | 67 | 0.30% | +0.26 |
| **Total** | **125,973** | | **22,544** | | |

### 3.3 Dataset Audit

A pre-training dataset audit (EXP-001) identified five data-integrity and evaluation-validity concerns:

- **Difficulty column leakage (LEAK-001, HIGH):** The `difficulty` column encodes the number of classifiers (out of 21) that correctly classified each record in the original KDD Cup 99 challenge. Because it is derived from the ground-truth label, including it as a feature would constitute label leakage. It is excluded from all feature matrices.
- **Novel test labels (LEAK-002, MEDIUM):** 17 attack labels appear in KDDTest+ but not in KDDTrain+. Models cannot have learned these specific attack signatures during training.
- **Cross-split exact duplicates (LEAK-003, LOW):** 610 test records (2.71% of KDDTest+) are exact feature-and-label duplicates of training records. These are reported but not excluded from primary evaluation.
- **Constant feature (LEAK-004, LOW):** `num_outbound_cmds` is constant across all training and test records (always zero) and carries no discriminative information. It is excluded.
- **Category distribution shift (LEAK-005, MEDIUM):** R2L increases from 0.79% of training to 12.80% of test. This represents the largest proportional shift and is the focus of EXP-004.

Additionally, 58 records in KDDTest+ share identical feature vectors with training records but carry different labels. Of these, 51 constitute binary Normal-vs-Attack conflicts, and 7 are multiclass Attack-vs-Attack conflicts that do not affect the binary task. The 51 binary conflicts represent an inherent irreducible error for any deterministic binary classifier on those specific inputs.

### 3.4 Preprocessing

For all modeling experiments (EXP-002 through EXP-005), preprocessing applies:

- **Feature removal:** `difficulty` and `num_outbound_cmds` are excluded from all feature matrices.
- **Categorical encoding:** `protocol_type`, `service`, `flag` are one-hot encoded using `OneHotEncoder` with `handle_unknown='ignore'`, fitted only on the training partition.
- **Numerical scaling:** All numerical features are standardized using `StandardScaler` (zero mean, unit variance), fitted only on the training partition.

This leaves **40 modeling features** (after removing the 2 excluded columns and before categorical expansion).

### 3.5 Train/Validation/Test Separation

KDDTrain+ (125,973 rows) is split into an **80% training partition** (100,778 rows) and a **20% validation partition** (25,195 rows) using stratified random sampling with random seed 42. All preprocessing transformers and model parameters are fitted exclusively on the 80% training partition. The validation partition is used for internal inspection and for threshold selection in EXP-005. KDDTest+ (22,544 rows) serves as the held-out test benchmark and is not used in any fitting or selection step prior to final evaluation.

### 3.6 Models

Three standard supervised classifiers are used throughout:

| Model | Configuration |
|---|---|
| Logistic Regression (LR) | `solver='lbfgs'`, `C=1.0`, `max_iter=1000`, `random_state=42` |
| Decision Tree (DT) | `criterion='gini'`, `max_depth=20`, `random_state=42` |
| Random Forest (RF) | `n_estimators=100`, `random_state=42`, `n_jobs=-1` |

These are identical across EXP-002, EXP-003, EXP-004, and EXP-005. No hyperparameter tuning, class weighting, or resampling is applied.

### 3.7 Metrics

**Detection Rate (DR):** TP / (TP + FN). The proportion of actual attacks correctly classified. Also known as Recall for the positive class.

**False Alarm Rate (FAR):** FP / (FP + TN). The proportion of normal records incorrectly classified as attacks. Also known as the False Positive Rate.

**Precision:** TP / (TP + FP). The proportion of records predicted as attacks that are true attacks.

**F1 Score:** Harmonic mean of Precision and Detection Rate.

**ROC-AUC:** Area under the Receiver Operating Characteristic curve. Computed from continuous probability scores and is therefore threshold-independent.

**Accuracy:** (TP + TN) / (TP + TN + FP + FN).

For subgroup analysis (EXP-003), subgroup Precision and F1 are computed on the cohort {Normal} ∪ {category} and are not directly equivalent to the overall binary Precision/F1 from EXP-002.

**Wilson Score Confidence Intervals:** 95% Wilson score intervals are reported for small-sample subgroups (specifically U2R, *N*=67).

### 3.8 Reproducibility

All experiments use Python 3.14.2 on Windows 11, scikit-learn 1.8.0, pandas 3.0.0, numpy 2.4.2, matplotlib 3.10.8. All random seeds are fixed at 42. Code and configuration files are versioned in the SentinelNet repository. Experiments were registered in `experiments/registry.csv` and assigned unique IDs (EXP-001 through EXP-005), all with status COMPLETE.

---

## 4. Experimental Design

### 4.1 EXP-001: Dataset Audit

**Research Question:** What are the statistical properties of the NSL-KDD training and test sets, and are there data-leakage risks that would invalidate downstream model evaluations?

**Hypothesis:** NSL-KDD contains measurable class imbalance, non-trivial train/test category overlap differences, and features that, if misused, would constitute label leakage.

**Method:** Programmatic analysis of raw data files: count statistics, duplicate detection, schema validation, feature-level constant detection, cross-split duplicate identification, novel attack label enumeration, and category-level distribution comparison.

**Evaluation Protocol:** No model training. Pure data-level analysis with all statistics computed programmatically from the raw files (SHA-256 verified).

**Results:**
- 610 exact feature-and-label cross-split duplicates (2.71% of KDDTest+).
- 17 attack labels in KDDTest+ with no corresponding training labels.
- 58 feature-identical records with conflicting labels (51 binary, 7 multiclass).
- 1 constant feature (`num_outbound_cmds`), excluded downstream.
- R2L category distribution shift: +12.01 percentage points from training to test.
- `difficulty` column identified as label-derived and excluded from all feature sets.

**Interpretation:** NSL-KDD contains measurable evaluation-validity concerns that should accompany any reported results. The 610 duplicates modestly inflate reported test accuracy; the 17 novel attack labels create structural difficulty for trained classifiers; the 51 binary conflicting-label records represent irreducible benchmark label noise; and the R2L distribution shift implies that R2L test performance measures a substantially different label distribution than what was seen during training.

**Limitations:** Cross-split near-duplicate records (feature similarity without exact match) are not characterized. The audit addresses binary target labels; multiclass label ambiguity is noted but not the primary concern for the binary task.

---

### 4.2 EXP-002: Supervised Binary Baseline

**Research Question:** What binary intrusion detection performance do Logistic Regression, Decision Tree, and Random Forest achieve, and how does detection degrade on novel attack labels?

**Hypothesis (H2):** Models trained on KDDTrain+ will achieve measurable binary detection but will exhibit reduced detection rates on attack labels not present in the training set.

**Method:** Train each model on the 80% training partition (100,778 rows). Evaluate on KDDTest+ (22,544 rows). Report results for five cohorts: full test, non-overlapping (excluding 610 duplicates), overlapping only, known-attack labels, and novel-attack labels.

**Evaluation Protocol:** No hyperparameter tuning, no class weighting, no resampling. Default decision threshold (0.50) applied to `predict_proba` outputs for all threshold-dependent metrics.

**Results (full KDDTest+):**

| Model | Accuracy | Precision | Detection Rate | FAR | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 75.36% | 91.70% | 62.36% | 7.46% | 74.24% | 0.7973 |
| Decision Tree | 78.73% | 91.27% | 69.26% | 8.75% | 78.76% | 0.8028 |
| Random Forest | 77.22% | 96.79% | 62.04% | 2.72% | 75.61% | 0.9535 |

**Known vs. Novel Attack Detection Rates:**

| Model | Known Attack DR | Novel Attack DR | Δ |
|---|---:|---:|---:|
| Logistic Regression | 71.25% | 40.83% | −30.43 pp |
| Decision Tree | 74.89% | 55.63% | −19.26 pp |
| Random Forest | 76.79% | 26.29% | −50.50 pp |

**Overlap Cohort Accuracy:**

| Model | Full Test | Non-Overlapping | Overlapping | Δ (Overlap − Non-Overlap) |
|---|---:|---:|---:|---:|
| Logistic Regression | 75.36% | 74.77% | 96.72% | +21.95 pp |
| Decision Tree | 78.73% | 78.16% | 99.34% | +21.19 pp |
| Random Forest | 77.22% | 76.59% | 99.67% | +23.08 pp |

**Interpretation:** Detection rates of 62%–69% at the default threshold indicate that a substantial fraction of attack records are misclassified. The overlap cohort shows near-perfect accuracy on the 610 exact training duplicates, demonstrating the expected pattern of memorization. Detection rates on novel attack labels are markedly lower than on known attacks, with Random Forest showing the largest gap (−50.50 pp). This is an observed pattern; EXP-002 does not establish a causal mechanism. All three models achieve high precision (91%–97%), indicating that their positive predictions are predominantly correct, but their recall (detection rate) is considerably lower. Random Forest achieves the highest ROC-AUC (0.9535), reflecting strong ranking ability that is not captured at the default threshold.

**Limitations:** All metrics are from a single random seed. Results reflect the 80% training partition, not the full KDDTrain+. The overlap cohort includes 610 samples, too few for strong statistical conclusions about the effect size. Novel attack degradation may partially reflect test-set distribution shift, not only the absence of novel labels during training.

---

### 4.3 EXP-003: Attack Category Sensitivity

**Research Question (RQ3):** How does binary detection performance vary across the DoS, Probe, R2L, and U2R attack categories?

**Hypothesis (H3):** Detection performance will be lower for the less represented attack categories (R2L, U2R) than for DoS and Probe.

**Method:** Using the same models fitted in EXP-002, evaluate each model separately on each of the four attack categories in KDDTest+. Category detection rate is computed as TP_category / N_category. Subgroup Precision and F1 are computed on the {Normal} ∪ {category} cohort. 95% Wilson score confidence intervals are computed for U2R (N=67) due to its small sample size.

**Results:**

| Model | Category | N | Detection Rate | Precision* | F1* |
|---|---|---:|---:|---:|---:|
| Logistic Regression | DoS | 7,460 | 81.92% | 89.41% | 85.50% |
| Logistic Regression | Probe | 2,421 | 76.17% | 71.81% | 73.92% |
| Logistic Regression | R2L | 2,885 | 1.18% | 4.49% | 1.87% |
| Logistic Regression | U2R | 67 | 20.90% | 1.90% | 3.48% |
| Decision Tree | DoS | 7,460 | 83.70% | 88.02% | 85.80% |
| Decision Tree | Probe | 2,421 | 82.86% | 70.24% | 76.03% |
| Decision Tree | R2L | 2,885 | 21.49% | 42.18% | 28.47% |
| Decision Tree | U2R | 67 | 26.87% | 2.07% | 3.85% |
| Random Forest | DoS | 7,460 | 80.16% | 95.77% | 87.27% |
| Random Forest | Probe | 2,421 | 74.85% | 87.28% | 80.59% |
| Random Forest | R2L | 2,885 | 5.55% | 37.74% | 9.67% |
| Random Forest | U2R | 67 | 13.43% | 3.30% | 5.29% |

*Subgroup Precision and F1 are computed on {Normal} ∪ {category} cohorts; they are not directly equivalent to overall EXP-002 values.

**U2R 95% Wilson Score Confidence Intervals** (N=67):

| Model | U2R DR | 95% Wilson CI |
|---|---:|---|
| Logistic Regression | 20.90% (14/67) | [12.88%, 32.07%] |
| Decision Tree | 26.87% (18/67) | [17.72%, 38.52%] |
| Random Forest | 13.43% (9/67) | [7.23%, 23.60%] |

**Interpretation:** H3 is observationally supported. R2L detection rates (1.18%–21.49%) and U2R detection rates (13.43%–26.87%) are dramatically lower than DoS (80.16%–83.70%) and Probe (74.85%–82.86%). This heterogeneity is masked by aggregate metrics: a classifier reporting 78% overall F1 simultaneously achieves 1.18% detection on R2L. U2R point estimates are highly uncertain due to N=67; the wide Wilson confidence intervals indicate that precise conclusions about U2R detection rates cannot be drawn from this test set. The R2L finding is robust by sample size (N=2,885) and warrants further investigation.

R2L's low detection rate cannot be attributed to category-level sample scarcity alone. Verified evidence shows substantial within-category composition differences between train and test (see EXP-004). The observed R2L results are consistent with multiple plausible explanations including category-level distribution shift, within-category label shift, and attack-specific feature characteristics; EXP-003 does not isolate any single cause.

**Limitations:** The category-level subgroup analysis is post-hoc; models were trained on the binary task and not optimized per category. The U2R test sample (N=67) is too small for reliable detection rate estimation.

---

### 4.4 EXP-004: R2L Within-Category Label Shift

**Research Question (RQ4):** Within the R2L category, do per-label detection rates correlate with per-label training representation?

**Hypothesis (H4):** R2L labels with more training examples will exhibit higher detection rates.

**Method:** For all R2L labels in KDDTest+, compute per-label detection rate using the models from EXP-002. Classify each label by representation status (shared: appears in both train and test; train-only; test-only). Compare detection rates across representation groups. Models are not retrained.

**R2L Label Composition:**

| Label | Train Count | Train Share | Test Count | Test Share | Status |
|---|---:|---:|---:|---:|---|
| warezclient | 890 | 89.45% | 0 | 0.00% | train-only |
| spy | 2 | 0.20% | 0 | 0.00% | train-only |
| guess_passwd | 53 | 5.33% | 1,231 | 42.67% | shared |
| warezmaster | 20 | 2.01% | 944 | 32.72% | shared |
| snmpguess | 0 | 0.00% | 331 | 11.47% | test-only |
| snmpgetattack | 0 | 0.00% | 178 | 6.17% | test-only |
| httptunnel | 0 | 0.00% | 133 | 4.61% | test-only |
| imap | 11 | 1.11% | 1 | 0.03% | shared |
| multihop | 7 | 0.70% | 18 | 0.62% | shared |
| phf | 4 | 0.40% | 2 | 0.07% | shared |
| ftp_write | 8 | 0.80% | 3 | 0.10% | shared |

Note: The two labels dominating KDDTrain+ R2L (warezclient: 890 samples) have zero test instances, while the two labels dominating KDDTest+ R2L (guess_passwd: 1,231; warezmaster: 944) were seen during training with only 53 and 20 examples respectively.

**Selected Per-Label Detection Rates:**

| Label | Train Count | Test Count | LR DR | DT DR | RF DR |
|---|---:|---:|---:|---:|---:|
| guess_passwd | 53 | 1,231 | 0.49% | 0.00% | 0.00% |
| warezmaster | 20 | 944 | 1.59% | 17.69% | 15.04% |
| snmpguess | 0 | 331 | 0.60% | **99.09%** | 0.00% |
| httptunnel | 0 | 133 | 1.50% | **81.95%** | 9.77% |
| snmpgetattack | 0 | 178 | 0.56% | 0.00% | 0.00% |
| imap | 11 | 1 | 100.00% | 100.00% | 0.00% |

**Interpretation:** H4 is not supported by these results. The most striking observations are:

1. **snmpguess** (0 training examples, 331 test instances): Decision Tree detects 99.09%. This high detection in the absence of any training examples for this label is a striking result; it indicates that the Decision Tree's learned decision boundary—based on feature patterns from other training labels—correctly classifies the majority of snmpguess records.

2. **guess_passwd** (53 training examples, 1,231 test instances): Decision Tree detects 0.00%, Random Forest detects 0.00%, Logistic Regression detects 0.49%. This is the dominant test label for R2L and is nearly completely missed despite having training representation.

These two observations together show that training representation count alone does not predict detection rate. The results indicate that attack-specific feature characteristics and their overlap with the decision boundary learned from other attack patterns are plausible contributing factors, but EXP-004 does not isolate their causal effects.

The 2,885 R2L test records are dominated by labels that were rare during training (guess_passwd, warezmaster), while the dominant training label (warezclient, 890 samples) has zero test instances. This structural mismatch is a key characteristic of the benchmark's R2L composition.

**Limitations:** EXP-004 is a post-hoc observational analysis; no causal mechanism for per-label detection differences is established. The imap result (1 test sample, 100% detection for LR and DT) should not be interpreted with precision given N=1. Models are not retrained per-label or per-category.

---

### 4.5 EXP-005: Threshold / Operating-Point Analysis

**Research Question (RQ5):** How does the binary classification threshold affect the trade-off between Detection Rate and False Alarm Rate?

**Hypothesis (H5):** Lowering the classification threshold will increase attack detection rate while also increasing false-alarm rate, producing an observable operating-point trade-off.

**Method:** Using the same models from EXP-002, generate continuous attack probability scores from `predict_proba(X)[:, 1]` on both the validation partition and KDDTest+. Phase 1: sweep 101 candidate thresholds (0.00 to 1.00 at 0.01 increments) on the validation partition, select a threshold per model using the pre-specified rule. Phase 2: apply the frozen threshold to KDDTest+.

**Protocol Correction (Documented):** An initial implementation of the threshold-selection rule specified selecting the *lowest* threshold achieving validation DR ≥ 90%. Because the classifier predicts Attack when score ≥ threshold, setting threshold = 0.00 assigns all records to the positive class, guaranteeing 100% Detection Rate and 100% False Alarm Rate for any model. This degenerate boundary condition was identified during pre-acceptance review. The threshold-selection protocol was formally amended to select the *highest* threshold achieving validation DR ≥ 90%. The initial run (threshold = 0.00 for all models) was rejected and its results are not used. The experiment was rerun under the corrected protocol.

**Corrected Threshold Selection Rule:**
- *Primary:* Select the **highest** validation threshold achieving a validation Detection Rate ≥ 90%.
- *Fallback:* If no threshold achieves 90% validation DR: (1) select the threshold with the highest validation DR; (2) tie-break on lowest validation FAR; (3) further tie-break on highest threshold.
- *Partition isolation:* KDDTest+ was quarantined throughout Phase 1. No test metric, label, or prediction was used to select or adjust any threshold.

**Validation-Selected Thresholds and Validation Metrics:**

| Model | Selected θ | Val. DR | Val. FAR | Val. Precision | Val. F1 | Rule |
|---|:---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.94 | 90.51% | 0.13% | 99.84% | 94.95% | Primary |
| Decision Tree | 1.00 | 99.88% | 0.16% | 99.81% | 99.85% | Primary |
| Random Forest | 1.00 | 95.84% | 0.00% | 100.00% | 97.87% | Primary |

*Note on Decision Tree and Random Forest θ = 1.00:* Decision Tree and Random Forest `predict_proba` values are not calibrated probabilities; they reflect leaf-node class frequency counts (DT) or ensemble vote fractions (RF). Both models produce scores heavily concentrated at 0.0 and 1.0, causing threshold = 1.00 to be the highest qualifying threshold. This is a known limitation of uncalibrated tree-based probability estimates.

**KDDTest+ Results (Frozen Threshold) vs. Default (θ = 0.50) Comparison:**

| Model | θ | Accuracy | Precision | Detection Rate | FAR | F1 | ROC-AUC |
|---|:---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.50 (default) | 75.36% | 91.70% | 62.36% | 7.46% | 74.24% | 0.7973 |
| | **0.94 (selected)** | 71.84% | 92.07% | 55.29% | 6.29% | 69.09% | 0.7973 |
| Decision Tree | 0.50 (default) | 78.73% | 91.27% | 69.26% | 8.75% | 78.76% | 0.8028 |
| | **1.00 (selected)** | 78.73% | 91.27% | 69.25% | 8.75% | 78.75% | 0.8028 |
| Random Forest | 0.50 (default) | 77.22% | 96.79% | 62.04% | 2.72% | 75.61% | 0.9535 |
| | **1.00 (selected)** | 68.02% | 98.67% | 44.42% | 0.79% | 61.26% | 0.9535 |

**Interpretation:** H5 is observationally supported on the validation partition: the threshold sweep clearly demonstrates that Detection Rate and FAR vary jointly with threshold across all three models. The validation partition results at the selected thresholds show high detection rates with very low false alarm rates, which is consistent with the corrected selection rule.

However, the selected thresholds do not generalize straightforwardly to KDDTest+. For Logistic Regression, the selected threshold (0.94) produces a test Detection Rate of 55.29% compared to 90.51% on validation—a gap of approximately 35 percentage points. For Random Forest, selecting θ = 1.00 (the highest threshold achieving ≥90% validation DR) reduces test Detection Rate from 62.04% (default) to 44.42%, while reducing FAR from 2.72% to 0.79%. For Decision Tree, θ = 1.00 produces results nearly identical to the default threshold.

The validation-to-test gap may reflect distribution differences between the validation partition (drawn from KDDTrain+) and KDDTest+ (a separate benchmark with different label composition, novel attack labels, and noted cross-split characteristics). EXP-005 does not establish which factor or combination of factors drives this gap; the benchmark characteristics documented in EXP-001 represent plausible contributing factors.

ROC-AUC values are identical for each model regardless of threshold, confirming that AUC is a threshold-independent measure of ranking ability.

**Limitations:** Threshold selection is based on a single validation partition from a single random split. The 0.01 grid resolution means the exact DR = 90% boundary may not be located precisely; the reported threshold is the highest grid point at or above the target. No calibration of tree probability outputs was performed. The validation-to-test gap is observed and documented but not causally explained.

---

## 5. Results

This section consolidates the most important numerical findings from EXP-001 through EXP-005.

### 5.1 Dataset Structure Summary (EXP-001)

| Metric | Value |
|---|---|
| KDDTrain+ rows | 125,973 |
| KDDTest+ rows | 22,544 |
| Cross-split exact duplicates | 610 (2.71% of test) |
| Test-only attack labels | 17 |
| Feature-identical conflicting-label records | 58 (51 binary, 7 multiclass) |
| Constant features excluded | 1 (`num_outbound_cmds`) |
| R2L distribution shift (train → test) | 0.79% → 12.80% (+12.01 pp) |

### 5.2 Baseline Binary Performance (EXP-002)

| Model | Accuracy | Detection Rate | FAR | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 75.36% | 62.36% | 7.46% | 91.70% | 74.24% | 0.7973 |
| Decision Tree | 78.73% | 69.26% | 8.75% | 91.27% | 78.76% | 0.8028 |
| Random Forest | 77.22% | 62.04% | 2.72% | 96.79% | 75.61% | 0.9535 |

Models were trained on 100,778 samples (80% of KDDTrain+). KDDTest+ (22,544 samples) was the held-out benchmark.

### 5.3 Known vs. Novel Attack Detection (EXP-002)

| Model | Known DR (N=9,083) | Novel DR (N=3,750) | Δ |
|---|---:|---:|---:|
| Logistic Regression | 71.25% | 40.83% | −30.43 pp |
| Decision Tree | 74.89% | 55.63% | −19.26 pp |
| Random Forest | 76.79% | 26.29% | −50.50 pp |

### 5.4 Attack Category Detection Rates (EXP-003)

| Model | DoS DR (N=7,460) | Probe DR (N=2,421) | R2L DR (N=2,885) | U2R DR (N=67) [95% CI] |
|---|---:|---:|---:|---|
| Logistic Regression | 81.92% | 76.17% | 1.18% | 20.90% [12.88%, 32.07%] |
| Decision Tree | 83.70% | 82.86% | 21.49% | 26.87% [17.72%, 38.52%] |
| Random Forest | 80.16% | 74.85% | 5.55% | 13.43% [7.23%, 23.60%] |

### 5.5 R2L Within-Category Disaggregation (EXP-004)

Selected per-label detection rates for the two largest test R2L labels and the most striking counterexample:

| Label | Train N | Test N | Status | LR DR | DT DR | RF DR |
|---|---:|---:|---|---:|---:|---:|
| guess_passwd | 53 | 1,231 | shared | 0.49% | 0.00% | 0.00% |
| warezmaster | 20 | 944 | shared | 1.59% | 17.69% | 15.04% |
| snmpguess | 0 | 331 | test-only | 0.60% | 99.09% | 0.00% |
| httptunnel | 0 | 133 | test-only | 1.50% | 81.95% | 9.77% |

### 5.6 Threshold Operating-Point Trade-off (EXP-005)

| Model | θ | Test DR | Test FAR | Test Precision | Test F1 |
|---|:---:|---:|---:|---:|---:|
| Logistic Regression | 0.50 (default) | 62.36% | 7.46% | 91.70% | 74.24% |
| | 0.94 (selected) | 55.29% | 6.29% | 92.07% | 69.09% |
| Decision Tree | 0.50 (default) | 69.26% | 8.75% | 91.27% | 78.76% |
| | 1.00 (selected) | 69.25% | 8.75% | 91.27% | 78.75% |
| Random Forest | 0.50 (default) | 62.04% | 2.72% | 96.79% | 75.61% |
| | 1.00 (selected) | 44.42% | 0.79% | 98.67% | 61.26% |

---

## 6. Discussion

### 6.1 Benchmark Structure and Evaluation Validity

EXP-001 establishes that NSL-KDD has several structural characteristics that complicate straightforward evaluation. The 610 exact cross-split duplicates cause a modest inflation in reported test accuracy (approximately +0.59–0.63 pp comparing full-test to non-overlapping accuracy). The 51 binary conflicting-label records represent an irreducible Bayes error for deterministic classifiers. The 17 test-only attack labels create a structural performance floor for novel attack detection, independent of model quality.

Most significantly, the R2L category distribution shift (+12.01 pp) means that R2L—which represents only 0.79% of training data—accounts for 12.80% of KDDTest+. A classifier that completely ignores R2L attacks would still achieve approximately 87% accuracy on the test set (by correctly classifying Normal, DoS, and Probe records). This observation implies that aggregate accuracy on KDDTest+ is not a reliable indicator of R2L detection capability.

### 6.2 Aggregate vs. Subgroup Performance

EXP-002 and EXP-003 together demonstrate the magnitude of the gap between aggregate and subgroup performance. Decision Tree achieves 78.73% overall accuracy and 78.76% F1, numbers that might suggest strong general performance. However, its R2L detection rate is 21.49% and its U2R detection rate is 26.87%. Logistic Regression achieves only 1.18% R2L detection—essentially failing entirely on this category while appearing adequate in aggregate metrics.

This finding supports the central thesis: *a single aggregate accuracy or F1 value does not adequately characterize supervised NIDS classifier behavior on NSL-KDD.* Category-level detection rate reporting is necessary to characterize the coverage of an intrusion detector.

For U2R, the 95% Wilson confidence intervals ([7.23%, 23.60%] to [17.72%, 38.52%]) underscore that U2R detection estimates from N=67 test samples have substantial uncertainty. No strong conclusion about U2R detection capability can be drawn from these results.

### 6.3 R2L Composition Shift

EXP-004 reveals a further complexity within the R2L category. The dominant training R2L label—warezclient (890 samples, 89.45% of training R2L)—has zero test instances. The dominant test labels—guess_passwd (1,231 test instances, 42.67% of test R2L) and warezmaster (944 test instances, 32.72% of test R2L)—were seen during training with only 53 and 20 examples respectively.

The result that Decision Tree detects 99.09% of snmpguess (absent from training) while detecting 0.00% of guess_passwd (present with 53 training examples) is a striking counterexample to the hypothesis that more training examples produce higher detection rates. These results are consistent with the interpretation that the learned decision boundary—derived from all training patterns—captures the feature signature of some attack types that were not seen during training, while failing to generalize to the feature distribution of others that were.

EXP-004 does not establish a causal explanation for these differences. Attack-specific feature characteristics and their overlap with the learned decision boundary are plausible contributing factors, but isolating these factors would require controlled experiments not conducted here.

### 6.4 Operating-Point Sensitivity

EXP-005 demonstrates that threshold choice substantially affects the DR/FAR trade-off on the validation partition. The validation sweep confirms H5 observationally: lowering the threshold increases detection rate while increasing false alarm rate across all models.

The notable finding is the divergence between validation-partition performance at the selected threshold and KDDTest+ performance at the same threshold. For Logistic Regression, the selected threshold (0.94) achieves 90.51% validation DR but only 55.29% test DR—a gap of approximately 35 pp. This divergence has plausible benchmark-related explanations (differing label compositions, novel attack labels in KDDTest+), but EXP-005 does not causally isolate any factor.

This divergence should be interpreted as a characteristic of the benchmark's train/test split design, not as a limitation of any specific model. It also illustrates a general challenge in threshold selection: a threshold chosen to achieve a target detection rate on an internal validation sample may not generalize with the same operating characteristics to a held-out benchmark.

The initial threshold-selection protocol error (selecting the *lowest* threshold achieving ≥90% DR, which produced threshold = 0.00 for all models) and its correction are documented here for methodological transparency. The rejected results are not used.

### 6.5 What the Experiments Establish

The five experiments collectively establish:

1. **(EXP-001)** NSL-KDD has measurable structural characteristics—cross-split duplicates, novel test labels, category distribution shift, feature-identical conflicting-label records—that should accompany any evaluation of classifiers on this benchmark.
2. **(EXP-002)** Three standard classifiers achieve aggregate accuracy of 75%–79% and F1 of 74%–79% on KDDTest+, with detection rates substantially lower on novel attack labels (26%–56%) than on known attack labels (71%–77%).
3. **(EXP-002, EXP-003)** Aggregate metrics conceal severe per-category heterogeneity. R2L detection rates (1%–21%) are far lower than DoS detection rates (80%–84%) across all classifiers.
4. **(EXP-004)** Within R2L, per-label training representation count does not predict per-label detection rate. Observed results are inconsistent with a simple "more training data = better detection" hypothesis.
5. **(EXP-005)** The default decision threshold (0.50) is not uniquely correct; threshold selection affects the DR/FAR trade-off, and validation-based threshold selection does not straightforwardly generalize to KDDTest+ in this benchmark.

### 6.6 What the Experiments Do NOT Establish

- **Causation for per-label detection differences:** EXP-004 shows that training count does not correlate with detection rate but does not establish what does.
- **Causation for the validation-to-test threshold gap:** EXP-005 observes the gap; contributing factors are identified as plausible but not isolated.
- **Generalization to real network traffic:** All results are specific to the NSL-KDD benchmark. No inference about real-world detection capability is warranted.
- **Zero-day or novel-attack detection capability:** Performance on test-only attack labels in EXP-002 is observed as a benchmark characteristic; it does not establish a model's general ability to detect previously unseen attacks.
- **Superiority of any model:** No model is identified as superior across all metrics. Decision Tree achieves the highest F1 (78.76%) and DR (69.26%) at the default threshold; Random Forest achieves the highest ROC-AUC (0.9535) and lowest FAR (2.72%). These are observed differences at one operating point.
- **Production readiness or deployment suitability:** No claim is made about real-time performance, scalability, or fitness for deployment.

---

## 7. Limitations

### 7.1 Benchmark Age and Synthetic Origin
NSL-KDD is derived from KDD Cup 1999 data captured in a 1998 simulated network environment. Modern network traffic patterns, protocols, and attack techniques differ substantially. Results on NSL-KDD cannot be generalized to contemporary network environments.

### 7.2 Exact Cross-Split Duplicates
The 610 exact feature-and-label duplicates present in both KDDTrain+ and KDDTest+ modestly inflate reported test accuracy. Results are reported on the full KDDTest+ benchmark for literature comparability, with non-overlapping cohort results reported separately.

### 7.3 Feature-Identical Conflicting-Label Records
The 51 binary Normal-vs-Attack conflicting-label records represent an irreducible source of classification error. These records reflect label noise or ambiguity in the benchmark and should not be attributed to model failure.

### 7.4 Novel and Test-Only Attack Labels
17 attack labels in KDDTest+ have no corresponding training labels. Detection performance on these labels cannot be attributed to learned signatures and should be interpreted in the context of their structural absence from training. These experiments do not establish a zero-shot or generalization detection claim.

### 7.5 Category Distribution Shift
The R2L category undergoes a +12.01 pp distribution shift from training to test. The aggregate test accuracy metric is dominated by the more frequent categories (Normal, DoS) and may not reflect performance on R2L. This is a benchmark design characteristic, not a property of the classifiers alone.

### 7.6 Training on 80% of KDDTrain+
All models are trained on 100,778 rows (80% of KDDTrain+). The 20% validation partition (25,195 rows) is used for internal analysis and threshold selection in EXP-005. Training on the full 125,973 rows would potentially yield different results; this comparison is not performed.

### 7.7 Small U2R Test Sample
The U2R test cohort contains only 67 samples. All U2R detection rate estimates are subject to substantial uncertainty, reflected in the 95% Wilson confidence intervals spanning approximately 15–20 percentage points in width. U2R conclusions should not be drawn from point estimates alone.

### 7.8 Near-Duplicate Analysis Not Performed
The EXP-001 audit identifies exact feature-and-label duplicates but does not characterize near-duplicate records (high feature similarity without exact match). Near-duplicates could further inflate test metrics in ways not captured here.

### 7.9 Single Random Seed
All experiments use random seed 42. Variability in results across different random seeds is not characterized. Results should be interpreted as observations for this specific experimental configuration.

### 7.10 Limited Model Family
Only three model families are evaluated (Logistic Regression, Decision Tree, Random Forest). These represent a useful baseline but do not cover the full range of classifiers relevant to intrusion detection.

### 7.11 No Cost-Sensitive Analysis
The threshold selection in EXP-005 uses a fixed 90% DR target to demonstrate operating-point sensitivity. No empirical security cost analysis (e.g., cost of missed attacks vs. false alarms) is conducted. The 90% target is methodologically motivated, not derived from operational requirements.

### 7.12 No Production Readiness Claim
No claim is made about real-time detection speed, computational resource requirements, scalability, or fitness for deployment in any network environment. These experiments evaluate classification performance on a static benchmark dataset under idealized conditions.

---

## 8. Conclusion

This paper presents five sequential experiments—dataset audit, binary baseline, attack-category sensitivity, R2L within-category disaggregation, and threshold operating-point analysis—on the NSL-KDD intrusion detection benchmark. Taken together, these experiments support a single overarching finding:

**A single aggregate accuracy or F1 value does not adequately characterize the behavior of supervised binary NIDS classifiers on NSL-KDD.**

The following specific claims are supported by experimental evidence:

1. NSL-KDD has measurable structural characteristics—610 exact cross-split duplicates, 17 test-only attack labels, 51 binary conflicting-label records, and a +12.01 pp R2L distribution shift—that complicate straightforward evaluation and should be reported alongside classifier metrics. *(Support: EXP-001)*

2. Classifiers achieving 75%–79% aggregate accuracy and 74%–79% F1 simultaneously achieve near-zero detection rates on R2L (1.18%–21.49%) and low detection rates on U2R (13.43%–26.87%), with U2R estimates subject to high uncertainty due to N=67. *(Support: EXP-002, EXP-003)*

3. Within the R2L category, training representation count is not a reliable predictor of per-label detection rate. A classifier can detect 99.09% of a test-only R2L label (snmpguess) while detecting 0.00% of a shared R2L label with 53 training examples (guess_passwd). *(Support: EXP-004; this is an observed result, not a causal claim.)*

4. The DR/FAR trade-off is sensitive to the decision threshold on the validation partition, but validation-selected thresholds do not straightforwardly generalize to the KDDTest+ operating point, consistent with the benchmark's documented distribution differences. *(Support: EXP-005)*

These findings collectively argue for a richer evaluation protocol for NSL-KDD research: subgroup detection rates per attack category, per-label disaggregation for heterogeneous categories, confidence intervals on small subgroups, and threshold sensitivity analysis alongside aggregate metrics. They also highlight that the benchmark's structural characteristics—not only classifier architecture—substantially influence what can be measured and concluded.

---

## 9. Future Work

The following directions are proposed as natural extensions of this research:

1. **Calibration of classifier probability outputs:** Decision Tree and Random Forest produce uncalibrated probability scores. Applying temperature scaling or isotonic regression calibration before threshold selection may improve the generalization of validation-selected operating points to the test set.

2. **Near-duplicate and similarity analysis:** EXP-001 characterizes exact duplicates but not near-duplicates. A similarity-based analysis (e.g., feature cosine similarity or learned embeddings) could better characterize the train/test overlap structure.

3. **Within-category feature analysis for R2L:** EXP-004 establishes that per-label detection rates are not explained by training representation counts. Future work could analyze whether specific feature distributions (rather than label counts) predict detection rate, without claiming causality until such an analysis is conducted.

4. **Category-aware or hierarchical models:** Training separate binary classifiers per attack category, or using hierarchical classification, may address the heterogeneity observed in EXP-003. The methodological tradeoffs (data requirements, evaluation protocol design) would need careful articulation.

5. **Multi-seed evaluation:** Running EXP-002 through EXP-005 across multiple random seeds would characterize result variance and support stronger statistical claims.

6. **External benchmark evaluation:** Evaluating the same classifiers on a second intrusion detection benchmark (e.g., CICIDS2017, UNSW-NB15) would assess the generalizability of the subgroup-sensitivity pattern beyond NSL-KDD.

7. **Operational cost-sensitive threshold selection:** Incorporating domain-specific cost ratios for false negatives and false positives into the threshold-selection criterion could improve the practical relevance of EXP-005-style analyses.

8. **Conformal prediction for detection uncertainty quantification:** Applying conformal prediction to produce per-record coverage guarantees could complement the Wilson CI approach used for U2R in EXP-003.

---

## 10. References

*Note: References below are marked as [REFERENCE NEEDED] where the citation has not been independently verified. No references are fabricated.*

[REFERENCE NEEDED: Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. *IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA)*. — Original NSL-KDD paper]

[REFERENCE NEEDED: KDD Cup 1999 dataset — original competition data paper/description]

[REFERENCE NEEDED: Survey of machine learning approaches for network intrusion detection (e.g., Buczak, A. L., & Guven, E. (2016). A survey of data mining and machine learning methods for cyber security intrusion detection. *IEEE Communications Surveys & Tutorials*)]

[REFERENCE NEEDED: Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR* — SMOTE oversampling for imbalanced datasets]

[REFERENCE NEEDED: Literature critiquing NSL-KDD limitations (e.g., papers noting the benchmark's age and structural issues)]

[REFERENCE NEEDED: Operating-point selection and ROC analysis reference — e.g., Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*]

[REFERENCE NEEDED: Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association* — source for Wilson score confidence interval]

[REFERENCE NEEDED: scikit-learn documentation / Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*]

---

## 11. Appendix

### A. Experiment Registry

| Experiment | Status | Research Question |
|---|---|---|
| EXP-001 | COMPLETE | Dataset audit: properties, leakage risks, distribution |
| EXP-002 | COMPLETE | Supervised binary baseline with cohort analysis |
| EXP-003 | COMPLETE | Attack-category detection sensitivity |
| EXP-004 | COMPLETE | R2L within-category label-shift disaggregation |
| EXP-005 | COMPLETE | Threshold / operating-point analysis |

All experiments used: NSL-KDD dataset; random seed 42; Python 3.14.2; scikit-learn 1.8.0; pandas 3.0.0; numpy 2.4.2; Windows 11.

### B. Metric Definitions

| Symbol | Name | Formula |
|---|---|---|
| DR | Detection Rate (Recall, TPR) | TP / (TP + FN) |
| FAR | False Alarm Rate (FPR) | FP / (FP + TN) |
| Precision | Positive Predictive Value | TP / (TP + FP) |
| F1 | F1 Score | 2 × Precision × DR / (Precision + DR) |
| Accuracy | Overall Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| ROC-AUC | Area Under ROC Curve | Computed from continuous scores; threshold-independent |

Positive class = Attack (label ≠ "normal"). Negative class = Normal (label = "normal").

### C. Reproducibility Details

- **KDDTrain+ SHA-256:** `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95`
- **KDDTest+ SHA-256:** `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84`
- **Training partition:** 100,778 rows (80% stratified split, seed=42)
- **Validation partition:** 25,195 rows (20% stratified split, seed=42)
- **Test partition:** KDDTest+, 22,544 rows (held out; used only for final evaluation)
- **Features used:** 40 (41 raw features minus `num_outbound_cmds`; `difficulty` excluded as metadata)
- **Model configurations:** As listed in Section 3.6

### D. Frozen Experiment Status and Self-Audit of Conclusions

The following table lists each conclusion from Section 8, its supporting experiment, and whether it is a direct observation or an inference:

| Conclusion | Supporting Experiment | Type |
|---|---|---|
| 610 cross-split exact duplicates | EXP-001 | Direct observation |
| 17 test-only attack labels | EXP-001 | Direct observation |
| 51 binary conflicting-label records | EXP-001 | Direct observation |
| R2L distribution shift +12.01 pp | EXP-001 | Direct observation |
| LR/DT/RF accuracy 75%–79%, F1 74%–79% | EXP-002 | Direct observation |
| R2L detection rate 1.18%–21.49% | EXP-003 | Direct observation |
| U2R detection rate 13.43%–26.87% (N=67) | EXP-003 | Direct observation |
| snmpguess DT detection 99.09% (N=331, train=0) | EXP-004 | Direct observation |
| guess_passwd DT detection 0.00% (N=1231, train=53) | EXP-004 | Direct observation |
| Training count does not predict detection rate | EXP-004 | Inference (supported by observed counterexample) |
| DR/FAR trade-off varies with threshold | EXP-005 | Direct observation (validation partition) |
| Validation-to-test threshold gap exists | EXP-005 | Direct observation |
| Distribution differences are *plausible* contributors to gap | EXP-005 | Interpreted/plausible; not causally established |
| Aggregate metrics insufficient to characterize NIDS | EXP-002, EXP-003 | Inference from observed subgroup heterogeneity |
