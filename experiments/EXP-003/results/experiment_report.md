# EXP-003 — Minority Attack Category Sensitivity

*Generated: 2026-09-30T14:09:51Z UTC*  
*Random seed: 42*

> **STATUS:** COMPLETE — All metrics computed from actual model evaluations.


## Research Question


RQ3: How does binary intrusion-detection performance vary across DoS, Probe, R2L, and U2R attack categories?


## Hypothesis


H3: Detection performance will be lower for the less represented attack categories, particularly R2L and U2R, than for the more represented DoS and Probe categories.

*Evaluation:* As shown below, detection rates for R2L and U2R are markedly lower than those for DoS and Probe across all three baseline classifiers, observationally supporting H3.


## Methodology


1. **Task Definition:** The models are strictly binary classifiers trained to distinguish Normal (0) from Attack (1). They do not output multiclass predictions. Evaluation by attack category is conducted strictly as post-hoc ground-truth subgroup analysis.

2. **No Artificial Rebalancing:** No resampling (SMOTE/undersampling), class-weighting, or threshold tuning was applied. The models operate with standard decision thresholds (0.5).

3. **Training Partition & Quarantine:** KDDTrain+ contains 125,973 records. An internal stratified 80/20 split was created (training partition = 100,778 rows; validation partition = 25,195 rows). All preprocessing transformers (OneHotEncoder, StandardScaler) and baseline models were fitted strictly on the 80% partition (100,778 rows); models were not refitted on 100% of KDDTrain+. KDDTest+ (22,544 rows) remained completely held out for final evaluation.


## Dataset Distribution

| Category | Train Count | Train % (Total) | Train % (Attacks) | Test Count | Test % (Total) | Test % (Attacks) | Δ (pp) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Normal | 67,343 | 53.46% | — | 9,711 | 43.08% | — | -10.38 |
| DoS | 45,927 | 36.46% | 78.33% | 7,460 | 33.09% | 58.13% | -3.37 |
| Probe | 11,656 | 9.25% | 19.88% | 2,421 | 10.74% | 18.87% | +1.49 |
| R2L | 995 | 0.79% | 1.70% | 2,885 | 12.80% | 22.48% | +12.01 |
| U2R | 52 | 0.04% | 0.09% | 67 | 0.30% | 0.52% | +0.26 |


## Category-Level Detection Rates (Primary Comparison)

| Model | DoS DR | Probe DR | R2L DR | U2R DR [95% CI] | Overall DR | Overall FAR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 81.92% | 76.17% | 1.18% | 20.90% (14/67) [12.88%, 32.07%] | 62.36% | 7.46% |
| Decision Tree | 83.70% | 82.86% | 21.49% | 26.87% (18/67) [17.72%, 38.52%] | 69.26% | 8.75% |
| Random Forest | 80.16% | 74.85% | 5.55% | 13.43% (9/67) [7.23%, 23.60%] | 62.04% | 2.72% |

> **Note on U2R Estimation Uncertainty:** U2R contains only N = 67 test samples. Because of this small sample size, U2R detection rate estimates have substantial statistical uncertainty, reflected in the wide 95% Wilson score confidence intervals (e.g., [7.23%, 23.60%] for Random Forest and [17.72%, 38.52%] for Decision Tree). Point estimates should not be interpreted with spurious precision.


## Detailed Subgroup Metrics by Model

> **Methodological Note on Subgroup Precision and F1:** In this subgroup analysis, Precision and F1 are calculated on the cohort {Normal} ∪ {category}. They quantify the binary performance of the classifier when evaluated specifically on distinguishing that attack category from normal traffic. Consequently, they are not directly equivalent to the overall binary Precision/F1 reported in EXP-002 (which evaluates all attack categories simultaneously against normal traffic). In contrast, the Category Detection Rate (Recall) measures TP_C / N_C within each attack subgroup and is mathematically unaffected by the composition of the negative class.


### Logistic Regression

| Category | N Samples | Detection Rate (Recall) | Precision (vs Normal) | F1 (vs Normal) |
| :--- | ---: | ---: | ---: | ---: |
| DoS | 7,460 | 81.92% | 89.41% | 85.50% |
| Probe | 2,421 | 76.17% | 71.81% | 73.92% |
| R2L | 2,885 | 1.18% | 4.49% | 1.87% |
| U2R | 67 | 20.90% | 1.90% | 3.48% |


### Decision Tree

| Category | N Samples | Detection Rate (Recall) | Precision (vs Normal) | F1 (vs Normal) |
| :--- | ---: | ---: | ---: | ---: |
| DoS | 7,460 | 83.70% | 88.02% | 85.80% |
| Probe | 2,421 | 82.86% | 70.24% | 76.03% |
| R2L | 2,885 | 21.49% | 42.18% | 28.47% |
| U2R | 67 | 26.87% | 2.07% | 3.85% |


### Random Forest

| Category | N Samples | Detection Rate (Recall) | Precision (vs Normal) | F1 (vs Normal) |
| :--- | ---: | ---: | ---: | ---: |
| DoS | 7,460 | 80.16% | 95.77% | 87.27% |
| Probe | 2,421 | 74.85% | 87.28% | 80.59% |
| R2L | 2,885 | 5.55% | 37.74% | 9.67% |
| U2R | 67 | 13.43% | 3.30% | 5.29% |


## Interpretation of Observed Results


- **Disparity Across Categories:** Detection performance is markedly uneven. Across all three models, DoS attacks exhibited detection rates between 80.2% and 83.7%, and Probe attacks showed detection rates between 74.8% and 82.9%. Conversely, R2L attacks exhibited severe detection degradation in Logistic Regression (1.18%) and Random Forest (5.55%), though Decision Tree detected 21.49%. U2R attacks showed detection rates ranging from 13.43% (RF) to 26.87% (DT).

- **Model Trade-offs:** Random Forest achieved the lowest False Alarm Rate (2.72%) on normal traffic and high precision, but demonstrated substantial conservatism on minority attacks (R2L DR: 5.55%, U2R DR: 13.43%). Decision Tree detected a broader fraction of minority attacks (R2L DR: 21.49%, U2R DR: 26.87%), but incurred a higher False Alarm Rate (8.75%). No single model dominates across all trade-offs.

- **Within-Category Composition Shift in R2L:** R2L's low detection rate cannot be attributed to category-level sample scarcity alone. Verified evidence reveals substantial within-category train/test composition shift:
  - **Training R2L (995 total samples):** 890 `warezclient` (89.45%), 53 `guess_passwd` (5.33%), 20 `warezmaster` (2.01%), 11 `imap` (1.11%), 8 `ftp_write` (0.80%), 7 `multihop` (0.70%), 4 `phf` (0.40%), 2 `spy` (0.20%).
  - **Test R2L (2,885 total samples):** 1,231 `guess_passwd` (42.67%), 944 `warezmaster` (32.72%), 686 test-only novel R2L samples (`snmpguess`: 331, `snmpgetattack`: 178, `httptunnel`: 133, `named`: 17, `sendmail`: 14, `xlock`: 9, `xsnoop`: 4), and **0 `warezclient`**.
  - **Conclusion:** R2L detection degradation is associated with both low category-level representation and substantial within-category attack-label composition shift between training and testing. EXP-003 does not isolate the causal contribution of either factor.


## Limitations


1. **Post-Hoc Subgroup Analysis:** EXP-003 analyzes subgroups evaluated through a binary detector. The models were not trained to distinguish between attack categories.

2. **Ground-Truth Subgroups:** Attack categories represent ground-truth metadata taxonomy, not model prediction outputs.

3. **Representation Imbalance:** R2L and U2R have substantially lower representation in KDDTrain+ (995 and 52 samples, respectively), providing fewer training examples from which the models can learn patterns associated with these attack categories.

4. **No Causal Claim:** The observational association between representation and detection rate does not establish that sample count alone caused poor performance. Architectural and feature suitability also play roles.

5. **Historical Benchmark:** NSL-KDD originates from 1998 network traffic captures; generalization to modern attack categories or traffic distributions cannot be inferred.

6. **Orthogonal to Novel Label Analysis:** The test set contains 17 novel attack labels (LEAK-002) distributed across these categories; category-level detection aggregates both known and novel variants within each family.


## Reproducibility

| Item | Value |
| :--- | :--- |
| Experiment ID | EXP-003 |
| Random seed | 42 |
| Execution timestamp | 2026-09-30T14:09:51Z |
| Python | 3.14.2 |
| Platform | Windows-11-10.0.26200-SP0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.0 |
| numpy | 2.4.2 |
| matplotlib | 3.10.8 |
| Train file | data/raw/KDDTrain+.txt (125,973 rows, SHA-256: 1b86d2f957b33082...) |
| Test file | data/raw/KDDTest+.txt (22,544 rows, SHA-256: fa46b0935342616a...) |
