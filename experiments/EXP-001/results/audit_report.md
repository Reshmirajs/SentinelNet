# EXP-001 — NSL-KDD Dataset Acquisition and Leakage Audit

*Generated: 2026-09-30T12:31:32Z UTC*  
*Random seed: 42*

> **STATUS:** COMPLETE — All statistics computed from actual dataset files.


## Research Question


What are the statistical properties of the NSL-KDD training and test sets, and are there any data-leakage risks (e.g., train/test overlap, difficulty-score contamination, or feature-level leakage) that would invalidate downstream model evaluations?


## Hypothesis


The NSL-KDD dataset contains measurable class imbalance and a non-trivial overlap between training and test attack categories. The difficulty field encodes per-sample classifier agreement scores that, if included as a feature, would constitute label leakage. Novel attack types in the test set will not appear in the training label set.


## Dataset

| Field | Value |
| :--- | :--- |
| Name | NSL-KDD |
| Source | University of New Brunswick — Canadian Institute for Cybersecurity |
| URL | https://www.unb.ca/cic/datasets/nsl.html |
| Citation | Tavallaee et al. (2009), IEEE CISDA |


### Dataset Files

| File | Rows | SHA-256 |
| :--- | ---: | :--- |
| KDDTrain+.txt | 125,973 | 1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95 |
| KDDTest+.txt | 22,544 | fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84 |


## Dataset Schema


The NSL-KDD dataset has **43 columns**: 41 network-traffic features, 1 target label column, and 1 metadata column (difficulty). There is no header row in the raw files.

| Column Group | Count | Feature Names |
| :--- | ---: | :--- |
| Categorical nominal | 3 | protocol_type, service, flag |
| Binary integer | 6 | land, logged_in, root_shell, su_attempted, is_host_login, is_guest_login |
| Numerical (count/ratio) | 32 | duration, src_bytes, dst_bytes, … (see feature_schema.csv) |
| Target label | 1 | label |
| Metadata (excluded from features) | 1 | difficulty |


## Data Quality Findings


### Training Set

| Metric | Value |
| :--- | ---: |
| Total rows | 125,973 |
| Total feature columns | 41 |
| Missing values (total) | 0 |
| Duplicate rows | 0 |
| Constant features | num_outbound_cmds |
| Near-constant numerical features (≤2 unique values) | land, logged_in, root_shell, num_outbound_cmds, is_host_login, is_guest_login |
| Negative-value anomalies | None |


### Test Set

| Metric | Value |
| :--- | ---: |
| Total rows | 22,544 |
| Total feature columns | 41 |
| Missing values (total) | 0 |
| Duplicate rows | 0 |
| Constant features | num_outbound_cmds |
| Near-constant numerical features (≤2 unique values) | land, logged_in, root_shell, num_outbound_cmds, is_host_login, is_guest_login |
| Negative-value anomalies | None |


## Class Distribution


### Training Set

| Attack Category | Count | Percentage |
| :--- | ---: | ---: |
| Normal | 67,343 | 53.46% |
| DoS | 45,927 | 36.46% |
| Probe | 11,656 | 9.25% |
| R2L | 995 | 0.79% |
| U2R | 52 | 0.04% |
| **TOTAL** | **125,973** | **100.00%** |

**Top 10 individual labels (Training):**

| Label | Category | Count | Percentage |
| :--- | :--- | ---: | ---: |
| normal | Normal | 67,343 | 53.46% |
| neptune | DoS | 41,214 | 32.72% |
| satan | Probe | 3,633 | 2.88% |
| ipsweep | Probe | 3,599 | 2.86% |
| portsweep | Probe | 2,931 | 2.33% |
| smurf | DoS | 2,646 | 2.10% |
| nmap | Probe | 1,493 | 1.19% |
| back | DoS | 956 | 0.76% |
| teardrop | DoS | 892 | 0.71% |
| warezclient | R2L | 890 | 0.71% |


### Test Set

| Attack Category | Count | Percentage |
| :--- | ---: | ---: |
| Normal | 9,711 | 43.08% |
| DoS | 7,460 | 33.09% |
| Probe | 2,421 | 10.74% |
| R2L | 2,885 | 12.80% |
| U2R | 67 | 0.30% |
| **TOTAL** | **22,544** | **100.00%** |

**Top 10 individual labels (Test):**

| Label | Category | Count | Percentage |
| :--- | :--- | ---: | ---: |
| normal | Normal | 9,711 | 43.08% |
| neptune | DoS | 4,657 | 20.66% |
| guess_passwd | R2L | 1,231 | 5.46% |
| mscan | Probe | 996 | 4.42% |
| warezmaster | R2L | 944 | 4.19% |
| apache2 | DoS | 737 | 3.27% |
| satan | Probe | 735 | 3.26% |
| processtable | DoS | 685 | 3.04% |
| smurf | DoS | 665 | 2.95% |
| back | DoS | 359 | 1.59% |


## Train / Test Analysis

| Metric | Value |
| :--- | :--- |
| Train unique labels | 23 |
| Test unique labels | 38 |
| Shared labels | 21 |
| Train-only labels | spy, warezclient |
| Test-only labels | apache2, httptunnel, mailbomb, mscan, named, processtable, ps, saint, sendmail, snmpgetattack, snmpguess, sqlattack, udpstorm, worm, xlock, xsnoop, xterm |
| Unique overlapping patterns | 610 |
| Total overlapping test samples | 610 (2.71% of test) |
| Feature-identical conflicting-label records | 58 (51 binary Normal-vs-Attack, 7 multiclass Attack-vs-Attack) |


### Feature-Identical Records with Conflicting Labels (Label Ambiguity)

In addition to the 610 exact feature-and-label duplicate records, a targeted cross-split feature comparison identifies **58 test records** whose 41 network connection features are 100% identical to training records, but carry conflicting ground-truth attack labels:

- **51 records represent binary Normal-vs-Attack conflicts:**
  - `snmpgetattack` (test, Attack) vs. `normal` (train, Normal): 28 records
  - `normal` (test, Normal) vs. `teardrop` (train, Attack): 9 records
  - `normal` (test, Normal) vs. `pod` (train, Attack): 9 records
  - `land` (test, Attack) vs. `normal` (train, Normal): 3 records
  - `pod` (test, Attack) vs. `normal` (train, Normal): 1 record
  - `warezmaster` (test, Attack) vs. `normal` (train, Normal): 1 record
- **7 records represent multiclass Attack-vs-Attack conflicts:**
  - `saint` (test, Probe) vs. `satan` (train, Probe): 7 records

*Interpretation & Methodological Impact:*  
These records represent label ambiguity in the benchmark and should be considered when interpreting test performance. Importantly, the 7 multiclass-only conflicts (`saint` vs. `satan`) do not create a binary target conflict for SentinelNet's Normal-vs-Attack task because both labels belong to the positive class (Attack, $y=1$). The 51 binary Normal-vs-Attack conflicts represent an inherent Bayes error limit for deterministic classifiers on these specific inputs. These records reflect benchmark label noise and ambiguity rather than data invalidation, but must be reported to avoid misattributing errors to model architectures.


### Category-Level Distribution Shift

| Category | Train % | Test % | Δ (pp) |
| :--- | ---: | ---: | ---: |
| Normal | 53.46% | 43.08% | -10.38 |
| DoS | 36.46% | 33.09% | -3.37 |
| Probe | 9.25% | 10.74% | +1.49 |
| R2L | 0.79% | 12.80% | +12.01 |
| U2R | 0.04% | 0.30% | +0.26 |


## Potential Leakage Risks


### LEAK-001 — HIGH Severity

**Description:** The 'difficulty' column encodes the number of classifiers (out of 21) that correctly classified each record in the original KDD Cup 99 challenge. It is computed *using the label* and thus directly encodes class difficulty. Including it as a training feature would constitute label leakage.

**Evidence:** difficulty column present in both train and test; values are derived from classifier agreement on the true label.

**Mitigation:** Treat difficulty as metadata only. Never include it in feature matrices used for model training or evaluation.


### LEAK-002 — MEDIUM Severity

**Description:** The test set contains attack labels not present in the training set. Models trained on train cannot have learned these classes, which may cause inflated false-negative rates and must be reported.

**Evidence:** Test-only labels: ['apache2', 'httptunnel', 'mailbomb', 'mscan', 'named', 'processtable', 'ps', 'saint', 'sendmail', 'snmpgetattack', 'snmpguess', 'sqlattack', 'udpstorm', 'worm', 'xlock', 'xsnoop', 'xterm']

**Mitigation:** Report per-label test performance separately. Consider a 'known vs unknown attack' evaluation protocol.


### LEAK-003 — LOW Severity

**Description:** Some feature-label rows appear in both the training and test sets. This can inflate test-set metrics if not handled carefully.

**Evidence:** 610 test samples (2.71% of test set) sharing 610 unique feature-label patterns are exact duplicates of training records.

**Mitigation:** Report results both including and excluding overlapping test rows. # RESEARCH DECISION NEEDED: decide whether to remove overlap rows.


### LEAK-004 — LOW Severity

**Description:** One or more features are constant (single unique value) across all samples. Constant features carry zero information and can cause issues with some normalisation schemes.

**Evidence:** Constant features: ['num_outbound_cmds']

**Mitigation:** Remove constant features during preprocessing. Document which features were removed and why.


### LEAK-005 — MEDIUM Severity

**Description:** Substantial class distribution shift between training and test sets (>5 percentage points). Accuracy evaluated on the test set may not reflect deployment performance, and may mask per-class degradation.

**Evidence:** Normal: train=53.5% → test=43.1% (Δ=-10.4pp) | R2L: train=0.8% → test=12.8% (Δ=+12.0pp)

**Mitigation:** Report per-category precision, recall, and F1 in addition to overall accuracy. Use macro-averaged metrics for fair comparison.


## Limitations


1. **Synthetic origin.** NSL-KDD is derived from KDD Cup 99, which was generated in a simulated environment. Results on this dataset may not generalise to real-world network traffic. # RESEARCH DECISION NEEDED: confirm dataset suitability for research goals.

2. **Outdated traffic patterns.** The underlying KDD Cup 99 data was captured in 1998. Modern attack signatures differ substantially.

3. **No preprocessing applied.** This audit examines raw, unprocessed data. Data quality conclusions refer only to the raw state.

4. **Difficulty column.** The difficulty field's exact computation is not fully documented; it is excluded from all downstream feature sets.

5. **Row overlap evaluation.** Cross-split duplication is evaluated via exact vector matching across all 41 features plus the label (excluding difficulty).


## Conclusion


This audit identified **5 leakage risks** (including 1 HIGH severity). The primary concern is the **difficulty column** (LEAK-001), which encodes label-derived classifier agreement and must be excluded from all feature matrices. The test set contains labels absent from training, requiring per-label evaluation reporting. Exact row overlap between splits is present and must be documented when reporting test-set metrics. No preprocessing, normalisation, encoding, or model training was performed in this experiment. All findings are based on the raw dataset files.


## Reproducibility

| Item | Value |
| :--- | :--- |
| Experiment ID | EXP-001 |
| Random seed | 42 |
| Execution timestamp | 2026-09-30T12:31:32Z |
| Python version | 3.14.2 |
| Platform | Windows-11-10.0.26200-SP0 |
| pandas | 3.0.0 |
| numpy | 2.4.2 |
| matplotlib | 3.10.8 |
| scipy | 1.17.0 |
