# EXP-004 — R2L Within-Category Label-Shift Analysis

*Generated: 2026-09-30T14:38:14Z UTC*  
*Random seed: 42*

> **STATUS:** COMPLETE — Disaggregated evaluation computed from actual model predictions.


## Research Question


RQ4: How does binary intrusion-detection performance vary across R2L attack labels according to their representation in the training data?


## Hypothesis


H4: R2L attack labels with greater representation in the training data will generally show higher detection rates than R2L labels with limited or no training representation.

**Evaluation of H4:** H4 is **NOT supported**. Higher training representation did not systematically produce higher detection rates. Shared labels with training exposure (e.g., `guess_passwd`, which had 53 training records) exhibited near-zero detection (0.0%–0.49%), while specific test-only labels with zero training exposure (e.g., `snmpguess` and `httptunnel`) achieved detection rates up to 99.09% and 81.95% in Decision Tree. The results indicate that training representation alone does not explain the observed detection differences. Attack-specific characteristics and feature distributions are plausible contributing factors, but EXP-004 does not isolate their causal effects.


## Motivation from EXP-003


In EXP-003, Remote-to-Local (R2L) attacks exhibited severe detection degradation across all three baseline classifiers (Logistic Regression: 1.18%, Random Forest: 5.55%, Decision Tree: 21.49%). However, audit evidence revealed that R2L cannot be evaluated simply as an underrepresented monolithic category: the attack labels dominating the test set were barely present in training, while the attack label dominating training was entirely absent from the test set. EXP-004 serves as a granular follow-up audit disaggregating this result by individual attack label.


## Dataset and Methodology


- **Dataset Files:** `KDDTrain+.txt` (125,973 rows) and `KDDTest+.txt` (22,544 rows).
- **Training Partition:** Consistent with EXP-002 and EXP-003, models were fitted strictly on the 80% stratified training partition (100,778 rows), leaving 25,195 validation rows and the fully quarantined test set.
- **Target & Preprocessing:** Strictly binary (Normal = 0, Attack = 1). Preprocessing pipeline (OneHotEncoder, StandardScaler, drop `difficulty` and `num_outbound_cmds`) identical to EXP-002/EXP-003.
- **Subgroup Metric:** Primary metric is Detection Rate (Recall) = $TP / N$ for each individual R2L label.


## R2L Intra-Category Composition Shift

| R2L Label | Train Count | Train Share (% R2L) | Test Count | Test Share (% R2L) | Status |
| :--- | ---: | ---: | ---: | ---: | :--- |
| ftp_write | 8 | 0.80% | 3 | 0.10% | shared |
| guess_passwd | 53 | 5.33% | 1,231 | 42.67% | shared |
| httptunnel | 0 | 0.00% | 133 | 4.61% | test_only |
| imap | 11 | 1.11% | 1 | 0.03% | shared |
| multihop | 7 | 0.70% | 18 | 0.62% | shared |
| named | 0 | 0.00% | 17 | 0.59% | test_only |
| phf | 4 | 0.40% | 2 | 0.07% | shared |
| sendmail | 0 | 0.00% | 14 | 0.49% | test_only |
| snmpgetattack | 0 | 0.00% | 178 | 6.17% | test_only |
| snmpguess | 0 | 0.00% | 331 | 11.47% | test_only |
| spy | 2 | 0.20% | 0 | 0.00% | train_only |
| warezclient | 890 | 89.45% | 0 | 0.00% | train_only |
| warezmaster | 20 | 2.01% | 944 | 32.72% | shared |
| xlock | 0 | 0.00% | 9 | 0.31% | test_only |
| xsnoop | 0 | 0.00% | 4 | 0.14% | test_only |


## Summary by Representation Group

| Model | Representation Status | Test Sample Count | TP | FN | Detection Rate (%) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Logistic Regression | shared | 2,199 | 25 | 2,174 | 1.14% |
| Logistic Regression | test_only | 686 | 9 | 677 | 1.31% |
| Decision Tree | shared | 2,199 | 177 | 2,022 | 8.05% |
| Decision Tree | test_only | 686 | 443 | 243 | 64.58% |
| Random Forest | shared | 2,199 | 144 | 2,055 | 6.55% |
| Random Forest | test_only | 686 | 16 | 670 | 2.33% |


> **Note on Train-Only Labels:** The train-only group (`warezclient` [890 samples] and `spy` [2 samples]) comprises 892 records (89.65% of all training R2L data). Because no corresponding records exist in KDDTest+, no test detection rate can be computed for this group. It is documented strictly as composition shift.


## Per-Label Detection Results

| Model | R2L Label | Status | Train N | Test N | TP | FN | Detection Rate (%) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | ftp_write | shared | 8 | 3 | 1 | 2 | 33.33% |
| Logistic Regression | guess_passwd | shared | 53 | 1,231 | 6 | 1,225 | 0.49% |
| Logistic Regression | httptunnel | test_only | 0 | 133 | 2 | 131 | 1.50% |
| Logistic Regression | imap | shared | 11 | 1 | 1 | 0 | 100.00% |
| Logistic Regression | multihop | shared | 7 | 18 | 2 | 16 | 11.11% |
| Logistic Regression | named | test_only | 0 | 17 | 1 | 16 | 5.88% |
| Logistic Regression | phf | shared | 4 | 2 | 0 | 2 | 0.00% |
| Logistic Regression | sendmail | test_only | 0 | 14 | 1 | 13 | 7.14% |
| Logistic Regression | snmpgetattack | test_only | 0 | 178 | 1 | 177 | 0.56% |
| Logistic Regression | snmpguess | test_only | 0 | 331 | 2 | 329 | 0.60% |
| Logistic Regression | spy | train_only | 2 | 0 | — | — | — (Train-Only) |
| Logistic Regression | warezclient | train_only | 890 | 0 | — | — | — (Train-Only) |
| Logistic Regression | warezmaster | shared | 20 | 944 | 15 | 929 | 1.59% |
| Logistic Regression | xlock | test_only | 0 | 9 | 2 | 7 | 22.22% |
| Logistic Regression | xsnoop | test_only | 0 | 4 | 0 | 4 | 0.00% |
| Decision Tree | ftp_write | shared | 8 | 3 | 3 | 0 | 100.00% |
| Decision Tree | guess_passwd | shared | 53 | 1,231 | 0 | 1,231 | 0.00% |
| Decision Tree | httptunnel | test_only | 0 | 133 | 109 | 24 | 81.95% |
| Decision Tree | imap | shared | 11 | 1 | 1 | 0 | 100.00% |
| Decision Tree | multihop | shared | 7 | 18 | 5 | 13 | 27.78% |
| Decision Tree | named | test_only | 0 | 17 | 4 | 13 | 23.53% |
| Decision Tree | phf | shared | 4 | 2 | 1 | 1 | 50.00% |
| Decision Tree | sendmail | test_only | 0 | 14 | 1 | 13 | 7.14% |
| Decision Tree | snmpgetattack | test_only | 0 | 178 | 0 | 178 | 0.00% |
| Decision Tree | snmpguess | test_only | 0 | 331 | 328 | 3 | 99.09% |
| Decision Tree | spy | train_only | 2 | 0 | — | — | — (Train-Only) |
| Decision Tree | warezclient | train_only | 890 | 0 | — | — | — (Train-Only) |
| Decision Tree | warezmaster | shared | 20 | 944 | 167 | 777 | 17.69% |
| Decision Tree | xlock | test_only | 0 | 9 | 0 | 9 | 0.00% |
| Decision Tree | xsnoop | test_only | 0 | 4 | 1 | 3 | 25.00% |
| Random Forest | ftp_write | shared | 8 | 3 | 0 | 3 | 0.00% |
| Random Forest | guess_passwd | shared | 53 | 1,231 | 0 | 1,231 | 0.00% |
| Random Forest | httptunnel | test_only | 0 | 133 | 13 | 120 | 9.77% |
| Random Forest | imap | shared | 11 | 1 | 0 | 1 | 0.00% |
| Random Forest | multihop | shared | 7 | 18 | 1 | 17 | 5.56% |
| Random Forest | named | test_only | 0 | 17 | 2 | 15 | 11.76% |
| Random Forest | phf | shared | 4 | 2 | 1 | 1 | 50.00% |
| Random Forest | sendmail | test_only | 0 | 14 | 0 | 14 | 0.00% |
| Random Forest | snmpgetattack | test_only | 0 | 178 | 0 | 178 | 0.00% |
| Random Forest | snmpguess | test_only | 0 | 331 | 0 | 331 | 0.00% |
| Random Forest | spy | train_only | 2 | 0 | — | — | — (Train-Only) |
| Random Forest | warezclient | train_only | 890 | 0 | — | — | — (Train-Only) |
| Random Forest | warezmaster | shared | 20 | 944 | 142 | 802 | 15.04% |
| Random Forest | xlock | test_only | 0 | 9 | 0 | 9 | 0.00% |
| Random Forest | xsnoop | test_only | 0 | 4 | 1 | 3 | 25.00% |


## Interpretation of Observed Results


1. **Shared Labels with Limited Exposure:** The shared label group accounts for 2,199 test samples (76.22% of test R2L), dominated by `guess_passwd` (1,231) and `warezmaster` (944). Despite appearing in both datasets, these attacks were barely represented in training (53 and 20 samples, respectively). Detection performance on these shared labels was severely degraded across all models: Logistic Regression detected 1.14% (25/2,199), Decision Tree detected 8.05% (177/2,199), and Random Forest detected 6.55% (144/2,199). Notably, both Decision Tree and Random Forest failed completely on `guess_passwd` (0 out of 1,231 detected).

2. **Test-Only Labels:** The 7 test-only labels account for 686 test records (23.78% of test R2L). The models had zero training examples for these exact names. Detection rates varied wildly: Decision Tree achieved a 64.58% aggregate detection rate on test-only labels (catching 328/331 `snmpguess` and 109/133 `httptunnel`), whereas Random Forest detected 2.33% (16/686) and Logistic Regression detected 1.31% (9/686). Detection performance on these test-only labels should be interpreted in the context of their absence from the training labels; it does not establish general zero-shot detection capability.

3. **Train-Only Labels:** `warezclient` (890 samples) made up 89.45% of all R2L training data, but has 0 test instances. The network patterns the classifiers had the greatest opportunity to learn were never tested.

4. **No Causal Attribution:** Lower training representation was associated with lower observed detection in parts of this benchmark, but representation is heavily confounded with attack-label composition and attack-specific connection features. EXP-004 does not establish that training sample volume caused detection outcomes.


## Limitations


1. **Post-Hoc Subgroup Analysis:** EXP-004 evaluates post-hoc subgroups of a binary detector; models were not trained to identify or discriminate individual attack labels.

2. **Small Sample Sizes:** Several individual R2L labels have very small test counts (e.g., `imap` N=1, `phf` N=2, `ftp_write` N=3, `xsnoop` N=4, `xlock` N=9), rendering per-label percentage estimates noisy.

3. **Zero-Shot Generalization:** Performance on test-only labels is observed within this specific benchmark setting and cannot be interpreted as a measure of general zero-shot detection capability; feature overlap between test-only labels and training examples is a possible contributing factor that EXP-004 does not formally assess.

4. **Confounding Factors:** Attack-specific characteristics (e.g., failed login attempts, port numbers, error flags) differ fundamentally across attack types and confound sample-size effects.

5. **Historical Benchmark:** NSL-KDD is based on 1998 network traffic captures; results cannot be generalized to modern remote-access threats.

6. **No Causality:** The experiment measures observational associations; causal impact of training count is unproven.

7. **Multiple Comparisons:** Evaluating 13 individual test labels increases the risk of idiosyncratic findings.

8. **External Validity:** Findings are specific to NSL-KDD and require external validation on modern intrusion detection datasets.


## Reproducibility

| Item | Value |
| :--- | :--- |
| Experiment ID | EXP-004 |
| Random seed | 42 |
| Execution timestamp | 2026-09-30T14:38:14Z |
| Python | 3.14.2 |
| Platform | Windows-11-10.0.26200-SP0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.0 |
| numpy | 2.4.2 |
| matplotlib | 3.10.8 |
| Train file | data/raw/KDDTrain+.txt (125,973 rows, SHA-256: 1b86d2f957b33082...) |
| Test file | data/raw/KDDTest+.txt (22,544 rows, SHA-256: fa46b0935342616a...) |
