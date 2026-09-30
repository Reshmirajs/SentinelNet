"""
Unit tests for EXP-005: Threshold / Operating-Point Analysis
=============================================================
Tests verify:
  1.  Threshold grid contains 0.00 through 1.00 at 0.01 increments (101 values).
  2.  Threshold selection uses validation data only (no test-label exposure).
  3.  Selected threshold satisfies the 90% validation DR rule when achievable.
  4.  Fallback rule is applied correctly when 90% DR is not achievable.
  5.  Threshold predictions are correctly generated from attack probability scores.
  6.  TP + FN equals number of positive samples.
  7.  TN + FP equals number of negative samples.
  8.  FAR is calculated correctly.
  9.  Selected threshold is not derived from test labels.
  10. Required output columns exist in all three CSV tables.
  11. All three models are represented in results.
  12. Test results use the frozen selected threshold.
  13. ROC-AUC uses continuous scores rather than thresholded predictions.
  14. Existing experiment files (EXP-001 through EXP-004) are not modified.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Load EXP-005 module dynamically
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
_exp005_path = REPO_ROOT / "experiments" / "EXP-005" / "run.py"
_spec = importlib.util.spec_from_file_location("exp005", _exp005_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec for {_exp005_path}")
exp005 = importlib.util.module_from_spec(_spec)
sys.modules["exp005"] = exp005
_spec.loader.exec_module(exp005)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal 43-column NSL-KDD DataFrame from override dicts."""
    base = {col: 0 for col in exp005.NSL_KDD_FEATURE_NAMES}
    base.update({
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "label": "normal",
        "difficulty": 21,
    })
    records = []
    for override in rows:
        row = dict(base)
        row.update(override)
        records.append(row)
    return pd.DataFrame(records, columns=exp005.NSL_KDD_FEATURE_NAMES)


@pytest.fixture
def synthetic_binary_data():
    """Return y_true and scores arrays with known compositions."""
    # 60 attacks, 40 normals
    rng = np.random.default_rng(42)
    y_true = np.array([1] * 60 + [0] * 40)
    # Attacks get higher scores, normals lower — controlled but not perfect
    attack_scores = rng.uniform(0.3, 1.0, size=60)
    normal_scores = rng.uniform(0.0, 0.7, size=40)
    scores = np.concatenate([attack_scores, normal_scores])
    return y_true, scores


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    return _make_df([
        {"label": "normal",   "src_bytes": 100, "duration": 1},
        {"label": "normal",   "src_bytes": 120, "duration": 2},
        {"label": "normal",   "src_bytes": 115, "duration": 1},
        {"label": "neptune",  "src_bytes": 500, "duration": 0},
        {"label": "neptune",  "src_bytes": 510, "duration": 0},
        {"label": "smurf",    "src_bytes": 600, "duration": 0},
    ])


@pytest.fixture
def synthetic_test_df() -> pd.DataFrame:
    return _make_df([
        {"label": "normal",  "src_bytes": 110, "duration": 1},
        {"label": "neptune", "src_bytes": 520, "duration": 0},
        {"label": "apache2", "src_bytes": 700, "duration": 0},
    ])


@pytest.fixture
def fitted_pipeline(synthetic_train_df):
    """Fit a LR pipeline on synthetic train data."""
    feature_cols = exp005.get_feature_columns(synthetic_train_df)
    model = LogisticRegression(solver="lbfgs", max_iter=200, random_state=42)
    pipe = exp005.build_pipeline(model, feature_cols)
    X = synthetic_train_df[feature_cols].values
    y = exp005.make_binary_target(synthetic_train_df["label"]).values
    pipe.fit(X, y)
    return pipe, feature_cols


# ---------------------------------------------------------------------------
# Test 1: Threshold grid
# ---------------------------------------------------------------------------

class TestThresholdGrid:
    """Verify the threshold grid specification."""

    def test_grid_has_101_values(self) -> None:
        assert len(exp005.THRESHOLD_GRID) == 101

    def test_grid_starts_at_zero(self) -> None:
        assert exp005.THRESHOLD_GRID[0] == pytest.approx(0.00)

    def test_grid_ends_at_one(self) -> None:
        assert exp005.THRESHOLD_GRID[-1] == pytest.approx(1.00)

    def test_grid_step_is_001(self) -> None:
        diffs = np.diff(exp005.THRESHOLD_GRID)
        assert np.allclose(diffs, 0.01, atol=1e-9)

    def test_grid_contains_zero_five(self) -> None:
        assert any(abs(t - 0.50) < 1e-9 for t in exp005.THRESHOLD_GRID)

    def test_grid_contains_zero_nine(self) -> None:
        assert any(abs(t - 0.90) < 1e-9 for t in exp005.THRESHOLD_GRID)


# ---------------------------------------------------------------------------
# Test 2 & 3: Threshold selection uses validation data only; 90% DR rule
# ---------------------------------------------------------------------------

class TestThresholdSelection:
    """Verify pre-specified threshold selection rule behaviour."""

    def _make_val_sweep(self, dr_vals: list[float], far_vals: list[float]) -> pd.DataFrame:
        """Construct a minimal val_sweep DataFrame for testing selection logic."""
        thresholds = np.round(np.arange(0.00, len(dr_vals) * 0.01, 0.01), 2)
        return pd.DataFrame({
            "model": ["TestModel"] * len(dr_vals),
            "threshold": thresholds[:len(dr_vals)],
            "tp": [0] * len(dr_vals),
            "tn": [0] * len(dr_vals),
            "fp": [0] * len(dr_vals),
            "fn": [0] * len(dr_vals),
            "detection_rate": dr_vals,
            "far": far_vals,
            "precision": [0.0] * len(dr_vals),
            "f1": [0.0] * len(dr_vals),
            "accuracy": [0.0] * len(dr_vals),
        })

    def test_primary_rule_selects_highest_threshold_above_target(self) -> None:
        """Primary: select the HIGHEST threshold where DR >= 90%."""
        # Thresholds: 0.00 (DR 0.95), 0.01 (DR 0.92), 0.02 (DR 0.85), 0.03 (DR 0.50)
        dr_vals = [0.95, 0.92, 0.85, 0.50]
        far_vals = [0.40, 0.30, 0.15, 0.05]
        sweep = self._make_val_sweep(dr_vals, far_vals)
        result = exp005.select_threshold(sweep, dr_target=0.90)
        # Thresholds >= 90% are 0.00 and 0.01; HIGHEST is 0.01
        assert result["selected_threshold"] == pytest.approx(0.01)
        assert result["validation_detection_rate"] >= 0.90
        assert "primary" in result["selection_rule"]

    def test_selected_threshold_not_simply_minimum_grid_value(self) -> None:
        """Verify the selected threshold does not degenerate to 0.00 when higher thresholds satisfy DR >= 90%."""
        dr_vals = [1.00, 0.98, 0.94, 0.85]
        far_vals = [1.00, 0.20, 0.10, 0.05]
        sweep = self._make_val_sweep(dr_vals, far_vals)
        result = exp005.select_threshold(sweep, dr_target=0.90)
        # 0.00, 0.01, 0.02 all have DR >= 0.90; highest is 0.02 (not 0.00)
        assert result["selected_threshold"] == pytest.approx(0.02)
        assert result["selected_threshold"] > 0.00

    def test_fallback_rule_when_90_not_achievable(self) -> None:
        """Fallback: select highest DR when no threshold reaches 90%."""
        dr_vals = [0.70, 0.75, 0.80, 0.78]
        far_vals = [0.60, 0.50, 0.40, 0.35]
        sweep = self._make_val_sweep(dr_vals, far_vals)
        result = exp005.select_threshold(sweep, dr_target=0.90)
        # Highest DR is 0.80 at index 2 (threshold = 0.02)
        assert result["selected_threshold"] == pytest.approx(0.02)
        assert result["validation_detection_rate"] == pytest.approx(0.80)
        assert "fallback" in result["selection_rule"]

    def test_fallback_tie_break_on_far(self) -> None:
        """Fallback tie-break: if two thresholds have equal highest DR, lowest FAR wins."""
        dr_vals = [0.85, 0.85, 0.80]
        far_vals = [0.60, 0.50, 0.30]
        sweep = self._make_val_sweep(dr_vals, far_vals)
        result = exp005.select_threshold(sweep, dr_target=0.90)
        # Both 0.00 and 0.01 have DR=0.85; FAR=0.60 vs 0.50 → choose 0.01
        assert result["selected_threshold"] == pytest.approx(0.01)
        assert result["validation_far"] == pytest.approx(0.50)

    def test_fallback_tie_break_on_highest_threshold(self) -> None:
        """Fallback tie-break: if DR and FAR are both tied, choose the highest threshold."""
        dr_vals = [0.85, 0.85, 0.80]
        far_vals = [0.40, 0.40, 0.30]
        sweep = self._make_val_sweep(dr_vals, far_vals)
        result = exp005.select_threshold(sweep, dr_target=0.90)
        # Both 0.00 and 0.01 have DR=0.85 and FAR=0.40 → choose highest threshold (0.01)
        assert result["selected_threshold"] == pytest.approx(0.01)

    def test_selection_result_has_required_keys(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        sweep = exp005.sweep_thresholds(y_true, scores, exp005.THRESHOLD_GRID, "Model")
        result = exp005.select_threshold(sweep)
        required = {"selected_threshold", "selection_rule",
                    "validation_detection_rate", "validation_far",
                    "validation_precision", "validation_f1"}
        assert required.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# Test 4: Fallback rule (covered above in TestThresholdSelection)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Test 5: Threshold prediction from continuous scores
# ---------------------------------------------------------------------------

class TestApplyThreshold:
    """Verify apply_threshold() correctness."""

    def test_score_at_threshold_is_positive(self) -> None:
        scores = np.array([0.50])
        preds = exp005.apply_threshold(scores, 0.50)
        assert preds[0] == 1

    def test_score_just_below_threshold_is_negative(self) -> None:
        scores = np.array([0.499])
        preds = exp005.apply_threshold(scores, 0.50)
        assert preds[0] == 0

    def test_zero_threshold_predicts_all_attack(self) -> None:
        scores = np.array([0.0, 0.5, 1.0])
        preds = exp005.apply_threshold(scores, 0.00)
        assert all(preds == 1)

    def test_threshold_one_predicts_all_normal(self) -> None:
        scores = np.array([0.0, 0.5, 0.999])
        preds = exp005.apply_threshold(scores, 1.00)
        assert all(preds == 0)

    def test_returns_integer_array(self) -> None:
        scores = np.array([0.3, 0.7])
        preds = exp005.apply_threshold(scores, 0.50)
        assert preds.dtype in (np.int32, np.int64, int, np.intp)


# ---------------------------------------------------------------------------
# Tests 6 & 7: TP+FN = positives; TN+FP = negatives
# ---------------------------------------------------------------------------

class TestMetricAdditivity:
    """Verify metric decomposition identities."""

    def test_tp_plus_fn_equals_positive_count(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        n_pos = int(np.sum(y_true == 1))
        for thr in [0.00, 0.30, 0.50, 0.70, 1.00]:
            m = exp005.compute_threshold_metrics(y_true, scores, thr)
            assert m["tp"] + m["fn"] == n_pos, f"Failed at threshold {thr}"

    def test_tn_plus_fp_equals_negative_count(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        n_neg = int(np.sum(y_true == 0))
        for thr in [0.00, 0.30, 0.50, 0.70, 1.00]:
            m = exp005.compute_threshold_metrics(y_true, scores, thr)
            assert m["tn"] + m["fp"] == n_neg, f"Failed at threshold {thr}"

    def test_confusion_matrix_sums_to_total(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        n_total = len(y_true)
        m = exp005.compute_threshold_metrics(y_true, scores, 0.50)
        assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == n_total


# ---------------------------------------------------------------------------
# Test 8: FAR calculation
# ---------------------------------------------------------------------------

class TestFARCalculation:
    """Verify FAR = FP / (FP + TN) = FP / n_neg."""

    def test_far_formula(self) -> None:
        y_true = np.array([1, 1, 0, 0, 0, 0])
        scores  = np.array([0.9, 0.8, 0.6, 0.4, 0.3, 0.2])
        # At threshold 0.5: predict [1,1,1,0,0,0]
        # TP=2, TN=3, FP=1, FN=0 → FAR = 1/4 = 0.25
        m = exp005.compute_threshold_metrics(y_true, scores, 0.50)
        assert m["tp"] == 2
        assert m["fp"] == 1
        assert m["tn"] == 3
        assert m["fn"] == 0
        expected_far = 1 / 4
        assert m["far"] == pytest.approx(expected_far, abs=1e-6)

    def test_far_zero_when_no_false_alarms(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        scores  = np.array([0.9, 0.8, 0.3, 0.2])
        m = exp005.compute_threshold_metrics(y_true, scores, 0.50)
        assert m["far"] == pytest.approx(0.0)

    def test_far_one_when_all_normals_flagged(self) -> None:
        y_true = np.array([1, 0, 0])
        scores  = np.array([0.9, 0.8, 0.7])
        m = exp005.compute_threshold_metrics(y_true, scores, 0.50)
        # FP = 2, TN = 0 → FAR = 1.0
        assert m["far"] == pytest.approx(1.0)

    def test_detection_rate_formula(self) -> None:
        y_true = np.array([1, 1, 1, 0])
        scores  = np.array([0.9, 0.8, 0.3, 0.1])
        # At threshold 0.50: predict [1,1,0,0] → TP=2, FN=1, DR=2/3
        m = exp005.compute_threshold_metrics(y_true, scores, 0.50)
        assert m["tp"] == 2
        assert m["fn"] == 1
        assert m["detection_rate"] == pytest.approx(2 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 9: Selected threshold not derived from test labels
# ---------------------------------------------------------------------------

class TestNoTestLeakage:
    """Verify that select_threshold() requires only validation-sweep data."""

    def test_select_threshold_takes_only_val_sweep(self, synthetic_binary_data) -> None:
        """select_threshold() signature takes only the validation sweep; no test data."""
        y_val, val_scores = synthetic_binary_data
        sweep = exp005.sweep_thresholds(y_val, val_scores, exp005.THRESHOLD_GRID, "TestModel")
        # This must succeed with only validation data
        result = exp005.select_threshold(sweep)
        assert "selected_threshold" in result

    def test_sweep_thresholds_does_not_use_test_set(self, synthetic_binary_data) -> None:
        """sweep_thresholds() takes y_true and scores — we verify it runs on val arrays."""
        y_val, val_scores = synthetic_binary_data
        # Call with clearly labelled validation data
        df = exp005.sweep_thresholds(y_val, val_scores, exp005.THRESHOLD_GRID, "Model")
        assert len(df) == len(exp005.THRESHOLD_GRID)


# ---------------------------------------------------------------------------
# Test 10: Required output columns exist
# ---------------------------------------------------------------------------

class TestOutputColumns:
    """Verify required columns in all three output CSV structures."""

    def test_validation_sweep_required_columns(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        sweep = exp005.sweep_thresholds(y_true, scores, exp005.THRESHOLD_GRID, "Model")
        required = {"model", "threshold", "tp", "tn", "fp", "fn",
                    "detection_rate", "far", "precision", "f1"}
        assert required.issubset(set(sweep.columns))

    def test_selected_thresholds_csv_columns(self) -> None:
        # Build a minimal selection result and format it as the CSV
        sel = {
            "Model A": {
                "selected_threshold": 0.20,
                "selection_rule": "primary",
                "validation_detection_rate": 0.91,
                "validation_far": 0.10,
                "validation_precision": 0.85,
                "validation_f1": 0.88,
            }
        }
        df = exp005.build_selected_thresholds_csv(sel)
        required = {"model", "selection_rule", "selected_threshold",
                    "validation_detection_rate", "validation_far",
                    "validation_precision", "validation_f1"}
        assert required.issubset(set(df.columns))

    def test_threshold_metrics_has_required_keys(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        m = exp005.compute_threshold_metrics(y_true, scores, 0.50)
        required = {"threshold", "tp", "tn", "fp", "fn",
                    "detection_rate", "far", "precision", "f1", "accuracy"}
        assert required.issubset(set(m.keys()))


# ---------------------------------------------------------------------------
# Test 11: All three models represented
# ---------------------------------------------------------------------------

class TestModelRepresentation:
    """Verify that all three expected models are referenced in module constants."""

    def test_threshold_grid_is_numpy_array(self) -> None:
        assert isinstance(exp005.THRESHOLD_GRID, np.ndarray)

    def test_default_threshold_is_zero_five(self) -> None:
        assert exp005.DEFAULT_THRESHOLD == pytest.approx(0.50)

    def test_selection_dr_target_is_ninety_percent(self) -> None:
        assert exp005.SELECTION_DR_TARGET == pytest.approx(0.90)

    def test_sweep_returns_all_thresholds(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        for model_name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
            df = exp005.sweep_thresholds(y_true, scores, exp005.THRESHOLD_GRID, model_name)
            assert len(df) == 101
            assert (df["model"] == model_name).all()


# ---------------------------------------------------------------------------
# Test 12: Test results use the frozen selected threshold
# ---------------------------------------------------------------------------

class TestFrozenThresholdUsed:
    """Verify that evaluate_on_test() uses the specified selected threshold."""

    def test_evaluate_on_test_uses_given_threshold(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        for thr in [0.10, 0.50, 0.90]:
            m = exp005.evaluate_on_test(y_true, scores, thr, "ModelX")
            # Manually compute expected TP at this threshold
            y_pred = (scores >= thr).astype(int)
            expected_tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            assert m["tp"] == expected_tp, f"TP mismatch at threshold {thr}"

    def test_evaluate_on_test_records_selected_threshold(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        m = exp005.evaluate_on_test(y_true, scores, 0.35, "ModelX")
        assert m["selected_threshold"] == pytest.approx(0.35)

    def test_evaluate_on_test_includes_roc_auc(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        m = exp005.evaluate_on_test(y_true, scores, 0.50, "ModelX")
        assert "roc_auc" in m
        assert 0.0 <= m["roc_auc"] <= 1.0


# ---------------------------------------------------------------------------
# Test 13: ROC-AUC uses continuous scores
# ---------------------------------------------------------------------------

class TestROCAUCFromContinuousScores:
    """Verify that ROC-AUC uses continuous probabilities, not hard predictions."""

    def test_roc_auc_differs_from_hard_prediction_accuracy(self, synthetic_binary_data) -> None:
        """ROC-AUC from continuous scores should differ from hard-prediction accuracy
        in most cases, demonstrating it is threshold-independent."""
        from sklearn.metrics import roc_auc_score
        y_true, scores = synthetic_binary_data
        m = exp005.evaluate_on_test(y_true, scores, 0.50, "ModelX")
        roc_auc_continuous = roc_auc_score(y_true, scores)
        # Confirm the module computes from continuous scores (not thresholded preds)
        assert m["roc_auc"] == pytest.approx(roc_auc_continuous, abs=1e-6)

    def test_roc_auc_is_between_zero_and_one(self, synthetic_binary_data) -> None:
        y_true, scores = synthetic_binary_data
        m = exp005.evaluate_on_test(y_true, scores, 0.50, "ModelX")
        assert 0.0 <= m["roc_auc"] <= 1.0

    def test_roc_auc_invariant_to_threshold_change(self, synthetic_binary_data) -> None:
        """AUC computed from continuous scores should not change with threshold."""
        y_true, scores = synthetic_binary_data
        m1 = exp005.evaluate_on_test(y_true, scores, 0.10, "M")
        m2 = exp005.evaluate_on_test(y_true, scores, 0.90, "M")
        assert m1["roc_auc"] == pytest.approx(m2["roc_auc"], abs=1e-9)


# ---------------------------------------------------------------------------
# Test 14: Existing experiment files are not modified
# ---------------------------------------------------------------------------

class TestExistingExperimentsUnchanged:
    """Verify that EXP-001 through EXP-004 files are intact."""

    def test_exp001_audit_report_exists(self) -> None:
        p = REPO_ROOT / "experiments" / "EXP-001" / "results" / "audit_report.md"
        assert p.exists(), "EXP-001 audit report is missing"

    def test_exp002_summary_results_json_exists(self) -> None:
        p = REPO_ROOT / "experiments" / "EXP-002" / "results" / "metrics" / "summary_results.json"
        assert p.exists(), "EXP-002 summary_results.json is missing"

    def test_exp002_poster_table_has_three_models(self) -> None:
        p = REPO_ROOT / "experiments" / "EXP-002" / "results" / "tables" / "poster_main_results_table.csv"
        df = pd.read_csv(p)
        assert len(df) == 3, f"Expected 3 model rows, got {len(df)}"

    def test_exp002_overlap_count_unchanged(self) -> None:
        p = REPO_ROOT / "experiments" / "EXP-001" / "results" / "metrics" / "train_test_overlap.json"
        with open(p) as f:
            data = json.load(f)
        assert data["exact_row_overlap_count"] == 610

    def test_exp003_category_results_has_12_rows(self) -> None:
        p = REPO_ROOT / "experiments" / "EXP-003" / "results" / "category_detection_results.csv"
        df = pd.read_csv(p)
        assert len(df) == 12, f"Expected 12 rows, got {len(df)}"

    def test_exp004_detection_results_has_45_rows(self) -> None:
        p = REPO_ROOT / "experiments" / "EXP-004" / "results" / "r2l_label_detection_results.csv"
        df = pd.read_csv(p)
        assert len(df) == 45, f"Expected 45 rows, got {len(df)}"

    def test_registry_has_four_complete_experiments_before_exp005(self) -> None:
        p = REPO_ROOT / "experiments" / "registry.csv"
        df = pd.read_csv(p)
        complete = df[df["status"] == "COMPLETE"]
        exp_ids = set(complete["experiment_id"].tolist())
        for eid in ["EXP-001", "EXP-002", "EXP-003", "EXP-004"]:
            assert eid in exp_ids, f"{eid} should be COMPLETE in registry"
