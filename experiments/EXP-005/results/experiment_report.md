# EXP-005 — Threshold / Operating-Point Analysis

*Generated: 2026-09-30T15:17:19Z UTC*  
*Random seed: 42*

> **STATUS:** COMPLETE — All threshold evaluations computed from actual model outputs.


## 1. Research Question


RQ5: How does the classification threshold affect the trade-off between attack detection rate and false-alarm rate for the SentinelNet binary classifiers?


## 2. Hypothesis


H5: Lowering the classification threshold will increase attack detection rate while also increasing false-alarm rate, producing an observable operating-point trade-off.

H5 is a directional hypothesis about threshold behavior. It does not predict which model will exhibit the trade-off most strongly, nor does it assume that any specific threshold is optimal.


## 3. Motivation from EXP-002


EXP-002 evaluated the three binary classifiers at the sklearn default threshold (0.50). A notable finding was that Random Forest achieved a substantially higher ROC-AUC (0.9535) than the other classifiers, while its default-threshold detection rate (62.04%) was similar to Logistic Regression (62.36%) and lower than Decision Tree (69.26%). ROC-AUC is a threshold-independent measure of ranking ability; it does not directly imply better detection at any particular threshold. EXP-005 tests whether adjusting the threshold using a pre-specified validation rule alters the detection/FAR trade-off, and whether the choice of operating point has a measurable effect on test performance. EXP-005 does not assume that the ROC-AUC / detection-rate difference observed in EXP-002 is caused by threshold choice.


## 4. Dataset


- **Train file:** `data/raw/KDDTrain+.txt` (125,973 rows, SHA-256: 1b86d2f957b33082...)
- **Test file:** `data/raw/KDDTest+.txt` (22,544 rows, SHA-256: fa46b0935342616a...)
- All NSL-KDD dataset limitations from EXP-001 apply: historical benchmark (1998 captures), train/test distribution differences, 610 exact cross-split duplicates, 17 test-only attack labels, and 58 feature-identical conflicting-label records. EXP-005 does not address these limitations.


## 5. Experimental Setup


- **Models:** Logistic Regression, Decision Tree, Random Forest (identical hyperparameters to EXP-002/003/004).
- **Training partition:** 80% stratified split of KDDTrain+ (100,778 rows). All preprocessing fitted on training partition only.
- **Validation partition:** 20% of KDDTrain+ (25,195 rows). Used exclusively for threshold selection.
- **Test set (KDDTest+):** 22,544 rows. Quarantined until threshold is frozen per model.
- **Score:** Continuous attack probability P(Attack) = `predict_proba(X)[:, 1]`.
- **Prediction rule:** prediction = 1 if attack_score ≥ threshold else 0.
- **Threshold grid:** 0.00 to 1.00 in increments of 0.01 (101 candidate thresholds).


## 6. Threshold-Selection Rule


The threshold selection rule was pre-specified to identify a conservative operating point on the validation partition before evaluating KDDTest+.

**Corrected Primary Rule:** Select the *highest* validation threshold that achieves a validation Detection Rate (Recall) of at least 90%.

**Corrected Fallback Rule:** If no candidate threshold in the 0.00–1.00 grid reaches 90% validation Detection Rate:
1. select the threshold with the highest validation Detection Rate;
2. if multiple thresholds tie, select the one with the lowest validation FAR;
3. if still tied, select the highest threshold.

**Rationale:** Selecting the highest threshold achieving the target ensures the classifier adopts the most conservative decision boundary (minimizing false alarms) while still satisfying the operational detection requirement (≥ 90% Recall).

> **Quarantine Note:** KDDTest+ was fully quarantined during threshold selection. No test labels, test predictions, or test metrics were used to determine or tune the threshold.


## 7. Methodological Correction (Protocol Amendment)


During initial experiment execution and validation, a methodological flaw in the initial protocol specification was identified:

- **Initial Rule Formulation:** The initial implementation specified selecting the *lowest* threshold achieving validation Detection Rate ≥ 90%.
- **Degenerate Result:** Because the classifier predicts Attack when $P(\text{Attack}) \ge \theta$, setting $\theta = 0.00$ assigns all samples to the positive class, producing 100% Detection Rate and 100% False Alarm Rate for all models.
- **Identification and Non-Acceptance:** This degenerate boundary condition was identified during pre-acceptance audit. The initial 0.00-threshold run was rejected and not accepted as a valid scientific result.
- **Protocol Amendment:** The selection rule was formally amended to select the *highest* threshold achieving ≥ 90% validation Detection Rate (with multi-stage tie-breaking for fallback).
- **Execution:** The experiment was completely rerun under the corrected protocol. The results presented in this report reflect exclusively the corrected protocol.


## 8. Validation Threshold Sweep — Selected Thresholds


The following table shows the validation-selected operating point for each model.

| Model | Selection Rule | Selected θ | Val. DR (%) | Val. FAR (%) | Val. Precision (%) | Val. F1 (%) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | primary: highest threshold >= 90% validation detection rate | 0.94 | 90.51% | 0.13% | 99.84% | 94.95% |
| Decision Tree | primary: highest threshold >= 90% validation detection rate | 1.00 | 99.88% | 0.16% | 99.81% | 99.85% |
| Random Forest | primary: highest threshold >= 90% validation detection rate | 1.00 | 95.84% | 0.00% | 100.00% | 97.87% |


## 9. Final KDDTest+ Results at Selected Threshold


Evaluated after threshold was frozen. KDDTest+ was not used in threshold selection.

| Model | Selected θ | Accuracy (%) | Precision (%) | Detection Rate (%) | FAR (%) | F1 (%) | ROC-AUC | TP | TN | FP | FN |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.94 | 71.84% | 92.07% | 55.29% | 6.29% | 69.09% | 0.7973 | 7,095 | 9,100 | 611 | 5,738 |
| Decision Tree | 1.00 | 78.73% | 91.27% | 69.25% | 8.75% | 78.75% | 0.8028 | 8,887 | 8,861 | 850 | 3,946 |
| Random Forest | 1.00 | 68.02% | 98.67% | 44.42% | 0.79% | 61.26% | 0.9535 | 5,700 | 9,634 | 77 | 7,133 |


## 10. Default vs. Selected Threshold Comparison (KDDTest+)


Default threshold values are from EXP-002 (full_test cohort, threshold = 0.50). EXP-002 was not modified. ROC-AUC is threshold-independent and is the same for both rows.

| Model | Threshold | Accuracy (%) | Precision (%) | Detection Rate (%) | FAR (%) | F1 (%) | TP | TN | FP | FN |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.50 | 75.36% | 91.70% | 62.36% | 7.46% | 74.24% | 8,003 | 8,987 | 724 | 4,830 |
| Logistic Regression | 0.94 | 71.84% | 92.07% | 55.29% | 6.29% | 69.09% | 7,095 | 9,100 | 611 | 5,738 |
| Decision Tree | 0.50 | 78.73% | 91.27% | 69.26% | 8.75% | 78.76% | 8,888 | 8,861 | 850 | 3,945 |
| Decision Tree | 1.00 | 78.73% | 91.27% | 69.25% | 8.75% | 78.75% | 8,887 | 8,861 | 850 | 3,946 |
| Random Forest | 0.50 | 77.22% | 96.79% | 62.04% | 2.72% | 75.61% | 7,961 | 9,447 | 264 | 4,872 |
| Random Forest | 1.00 | 68.02% | 98.67% | 44.42% | 0.79% | 61.26% | 5,700 | 9,634 | 77 | 7,133 |


## 11. Interpretation


**Threshold independence of ROC-AUC:** ROC-AUC describes the classifier's ability to rank attack records above normal records across all thresholds. It does not describe performance at any specific threshold. Differences in ROC-AUC across models should not be interpreted as directly implying better detection at the selected operating point.

**Threshold sensitivity:** The validation sweep demonstrates that detection rate and false-alarm rate are jointly sensitive to the classification threshold, consistent with H5. Lowering the threshold increases detection rate while also increasing false-alarm rate across all models evaluated.

**Operating-point trade-off:** The selected thresholds represent the operating point defined by the pre-specified validation rule (highest threshold achieving ≥ 90% validation detection rate, or fallback). This point is not claimed to be optimal. The appropriate operating point depends on the relative cost of missed attacks and false alarms in a given deployment context, which is not measured in this study.

**Comparison with EXP-002 default threshold:** Changes in detection rate and FAR between the default threshold (0.50) and the validation-selected threshold are observed differences; they do not indicate that one threshold is universally preferable.

**Validation-to-test generalization:** Threshold selection was performed on the validation partition. Detection rates and FAR values on KDDTest+ may differ from validation values, with distribution differences between the two splits, including novel attack labels and the 610 cross-split duplicates, representing plausible contributing factors that EXP-005 does not causally isolate.


## 12. Limitations


1. **Historical benchmark:** NSL-KDD is based on 1998 network traffic captures; results cannot be generalized to modern network environments.

2. **Single operating point per model:** A single threshold is selected per model. In practice, multiple operating points along the ROC curve may be relevant.

3. **Validation/test distribution shift:** The validation and test partitions have different label compositions (train-only vs. test-only attack labels); the selected threshold may not generalize perfectly to all test subgroups.

4. **Threshold grid resolution:** The 0.01 increment grid may not locate the exact threshold satisfying the 90% DR target; the reported threshold is the highest grid point at or above the target.

5. **No cost-sensitive analysis:** The 90% DR target was chosen to demonstrate threshold sensitivity; no empirical security cost analysis was conducted.

6. **Decision Tree probability calibration:** Decision Tree predict_proba values are class frequency counts in leaf nodes and may not be well-calibrated probabilities. This affects the threshold sweep behavior for Decision Tree specifically.

7. **Known NSL-KDD limitations from EXP-001:** 610 exact cross-split duplicates, 17 test-only attack labels, 58 feature-identical conflicting-label records, and the constant `num_outbound_cmds` feature remain present and were documented in EXP-001. EXP-005 does not address these.


## 13. Reproducibility

| Item | Value |
| :--- | :--- |
| Experiment ID | EXP-005 |
| Random seed | 42 |
| Execution timestamp | 2026-09-30T15:17:19Z |
| Python | 3.14.2 |
| Platform | Windows-11-10.0.26200-SP0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.0 |
| numpy | 2.4.2 |
| matplotlib | 3.10.8 |
| Threshold grid | 0.00–1.00 step 0.01 (101 values) |
| DR target for selection | 90% |
| Default threshold (EXP-002 baseline) | 0.5 |
| Train file | data/raw/KDDTrain+.txt (125,973 rows, SHA-256: 1b86d2f957b33082...) |
| Test file | data/raw/KDDTest+.txt (22,544 rows, SHA-256: fa46b0935342616a...) |
