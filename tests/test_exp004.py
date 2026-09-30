"""
Unit tests for EXP-004: R2L Within-Category Label-Shift Analysis
================================================================
Tests verify:
  - R2L composition table structure and group classification
  - Representation status assignment (shared / test_only / train_only)
  - Per-label evaluation metric correctness (TP + FN == test_count, DR = TP / N)
  - Feature column extraction and metadata exclusion (identical to EXP-002/003)
  - summarize_representation_groups aggregation consistency
  - Normal traffic absent from per-label results
  - No hard-coded dataset counts (counts derived from fixture data)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Load EXP-004 module dynamically
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
_exp004_path = REPO_ROOT / "experiments" / "EXP-004" / "run.py"
_spec = importlib.util.spec_from_file_location("exp004", _exp004_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec for {_exp004_path}")
exp004 = importlib.util.module_from_spec(_spec)
sys.modules["exp004"] = exp004
_spec.loader.exec_module(exp004)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal 43-column NSL-KDD DataFrame from a list of override dicts."""
    base = {col: 0 for col in exp004.NSL_KDD_FEATURE_NAMES}
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
    return pd.DataFrame(records, columns=exp004.NSL_KDD_FEATURE_NAMES)


@pytest.fixture
def r2l_composition_train() -> pd.DataFrame:
    """Synthetic train set with specific R2L labels."""
    return _make_df([
        {"label": "normal",       "src_bytes": 100},
        {"label": "normal",       "src_bytes": 120},
        {"label": "guess_passwd", "src_bytes": 50},   # shared R2L
        {"label": "guess_passwd", "src_bytes": 55},   # shared R2L
        {"label": "warezmaster",  "src_bytes": 30},   # shared R2L
        {"label": "warezclient",  "src_bytes": 10},   # train-only R2L
        {"label": "neptune",      "src_bytes": 500},  # DoS (not R2L)
    ])


@pytest.fixture
def r2l_composition_test() -> pd.DataFrame:
    """Synthetic test set with specific R2L labels."""
    return _make_df([
        {"label": "normal",        "src_bytes": 110},
        {"label": "guess_passwd",  "src_bytes": 60},   # shared R2L
        {"label": "warezmaster",   "src_bytes": 35},   # shared R2L
        {"label": "snmpguess",     "src_bytes": 200},  # test-only R2L
        {"label": "httptunnel",    "src_bytes": 300},  # test-only R2L
        {"label": "neptune",       "src_bytes": 600},  # DoS (not R2L)
    ])


@pytest.fixture
def synthetic_pipeline(r2l_composition_train):
    """A fitted DecisionTree pipeline on synthetic train data."""
    feature_cols = exp004.get_feature_columns(r2l_composition_train)
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    pipe = exp004.build_pipeline(model, feature_cols)
    X_tr = r2l_composition_train[feature_cols].values
    y_tr = exp004.make_binary_target(r2l_composition_train["label"]).values
    pipe.fit(X_tr, y_tr)
    return pipe, feature_cols


# ---------------------------------------------------------------------------
# TestR2LComposition
# ---------------------------------------------------------------------------

class TestR2LComposition:
    """Verify compute_r2l_composition() structure and group assignments."""

    def test_returns_dataframe(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        assert isinstance(comp_df, pd.DataFrame)

    def test_required_columns_present(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        required = {
            "r2l_label", "train_count", "train_share_r2l_pct",
            "test_count", "test_share_r2l_pct", "representation_status",
        }
        assert required.issubset(set(comp_df.columns))

    def test_shared_labels_correctly_identified(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        shared = set(comp_df[comp_df["representation_status"] == "shared"]["r2l_label"])
        # guess_passwd and warezmaster appear in both train and test
        assert "guess_passwd" in shared
        assert "warezmaster" in shared

    def test_train_only_label_identified(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        train_only = set(comp_df[comp_df["representation_status"] == "train_only"]["r2l_label"])
        assert "warezclient" in train_only

    def test_test_only_labels_identified(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        test_only = set(comp_df[comp_df["representation_status"] == "test_only"]["r2l_label"])
        assert "snmpguess" in test_only
        assert "httptunnel" in test_only

    def test_normal_not_in_composition_table(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        assert "normal" not in comp_df["r2l_label"].values

    def test_dos_labels_not_in_composition_table(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        assert "neptune" not in comp_df["r2l_label"].values

    def test_train_count_matches_actual_data(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        gp_row = comp_df[comp_df["r2l_label"] == "guess_passwd"].iloc[0]
        # Fixture has 2 guess_passwd in train
        assert gp_row["train_count"] == 2

    def test_test_count_matches_actual_data(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        gp_row = comp_df[comp_df["r2l_label"] == "guess_passwd"].iloc[0]
        # Fixture has 1 guess_passwd in test
        assert gp_row["test_count"] == 1

    def test_train_only_label_has_zero_test_count(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        wc_row = comp_df[comp_df["r2l_label"] == "warezclient"].iloc[0]
        assert wc_row["test_count"] == 0

    def test_test_only_label_has_zero_train_count(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        sg_row = comp_df[comp_df["r2l_label"] == "snmpguess"].iloc[0]
        assert sg_row["train_count"] == 0

    def test_train_shares_sum_to_100_for_present_labels(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        train_present = comp_df[comp_df["train_count"] > 0]
        assert train_present["train_share_r2l_pct"].sum() == pytest.approx(100.0, abs=0.1)

    def test_test_shares_sum_to_100_for_present_labels(
        self, r2l_composition_train: pd.DataFrame, r2l_composition_test: pd.DataFrame
    ) -> None:
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        test_present = comp_df[comp_df["test_count"] > 0]
        assert test_present["test_share_r2l_pct"].sum() == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# TestR2LEvaluation
# ---------------------------------------------------------------------------

class TestR2LEvaluation:
    """Verify evaluate_r2l_labels() metric correctness."""

    def test_tp_plus_fn_equals_test_count(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        pipe, feature_cols = synthetic_pipeline
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        results = exp004.evaluate_r2l_labels(pipe, r2l_composition_test, comp_df, feature_cols, "TestModel")
        for row in results:
            if row["test_count"] > 0:
                assert row["tp"] + row["fn"] == row["test_count"], (
                    f"TP + FN != test_count for label '{row['r2l_label']}'"
                )

    def test_detection_rate_equals_tp_over_n(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        pipe, feature_cols = synthetic_pipeline
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        results = exp004.evaluate_r2l_labels(pipe, r2l_composition_test, comp_df, feature_cols, "TestModel")
        for row in results:
            if row["test_count"] > 0:
                expected_dr = round(row["tp"] / row["test_count"] * 100, 2)
                assert row["detection_rate_pct"] == pytest.approx(expected_dr, abs=0.01), (
                    f"Detection rate mismatch for label '{row['r2l_label']}'"
                )

    def test_train_only_labels_have_none_tp_fn_dr(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        pipe, feature_cols = synthetic_pipeline
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        results = exp004.evaluate_r2l_labels(pipe, r2l_composition_test, comp_df, feature_cols, "TestModel")
        for row in results:
            if row["representation_status"] == "train_only":
                assert row["tp"] is None
                assert row["fn"] is None
                assert row["detection_rate_pct"] is None

    def test_normal_not_in_results(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        pipe, feature_cols = synthetic_pipeline
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        results = exp004.evaluate_r2l_labels(pipe, r2l_composition_test, comp_df, feature_cols, "TestModel")
        labels_in_results = [r["r2l_label"] for r in results]
        assert "normal" not in labels_in_results

    def test_model_name_recorded_in_results(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        pipe, feature_cols = synthetic_pipeline
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        results = exp004.evaluate_r2l_labels(pipe, r2l_composition_test, comp_df, feature_cols, "MyModel")
        for row in results:
            assert row["model"] == "MyModel"

    def test_results_returns_list_of_dicts(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        pipe, feature_cols = synthetic_pipeline
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        results = exp004.evaluate_r2l_labels(pipe, r2l_composition_test, comp_df, feature_cols, "TestModel")
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_required_keys_in_result_rows(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        pipe, feature_cols = synthetic_pipeline
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        results = exp004.evaluate_r2l_labels(pipe, r2l_composition_test, comp_df, feature_cols, "TestModel")
        required_keys = {
            "model", "r2l_label", "train_count", "test_count",
            "train_share_r2l_pct", "test_share_r2l_pct",
            "representation_status", "tp", "fn", "detection_rate_pct",
        }
        for row in results:
            assert required_keys.issubset(set(row.keys()))


# ---------------------------------------------------------------------------
# TestFeatureColumns
# ---------------------------------------------------------------------------

class TestFeatureColumns:
    """Verify feature selection and metadata exclusion (identical to EXP-002/003)."""

    def test_difficulty_excluded(self, r2l_composition_train: pd.DataFrame) -> None:
        cols = exp004.get_feature_columns(r2l_composition_train)
        assert "difficulty" not in cols

    def test_label_excluded(self, r2l_composition_train: pd.DataFrame) -> None:
        cols = exp004.get_feature_columns(r2l_composition_train)
        assert "label" not in cols

    def test_num_outbound_cmds_excluded(self, r2l_composition_train: pd.DataFrame) -> None:
        cols = exp004.get_feature_columns(r2l_composition_train)
        assert "num_outbound_cmds" not in cols

    def test_feature_count_is_40(self, r2l_composition_train: pd.DataFrame) -> None:
        cols = exp004.get_feature_columns(r2l_composition_train)
        assert len(cols) == 40

    def test_categorical_features_present(self, r2l_composition_train: pd.DataFrame) -> None:
        cols = exp004.get_feature_columns(r2l_composition_train)
        for feat in exp004.CATEGORICAL_FEATURES:
            assert feat in cols, f"Expected categorical feature '{feat}' in feature columns"

    def test_binary_features_present(self, r2l_composition_train: pd.DataFrame) -> None:
        cols = exp004.get_feature_columns(r2l_composition_train)
        for feat in exp004.BINARY_FEATURES:
            assert feat in cols, f"Expected binary feature '{feat}' in feature columns"


# ---------------------------------------------------------------------------
# TestRepresentationGroups
# ---------------------------------------------------------------------------

class TestRepresentationGroups:
    """Verify summarize_representation_groups() aggregation correctness."""

    def _make_det_df(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> pd.DataFrame:
        pipe, feature_cols = synthetic_pipeline
        comp_df = exp004.compute_r2l_composition(r2l_composition_train, r2l_composition_test)
        results = exp004.evaluate_r2l_labels(pipe, r2l_composition_test, comp_df, feature_cols, "TestModel")
        return pd.DataFrame(results)

    def test_returns_dataframe(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        det_df = self._make_det_df(synthetic_pipeline, r2l_composition_train, r2l_composition_test)
        grp_df = exp004.summarize_representation_groups(det_df)
        assert isinstance(grp_df, pd.DataFrame)

    def test_required_columns_present(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        det_df = self._make_det_df(synthetic_pipeline, r2l_composition_train, r2l_composition_test)
        grp_df = exp004.summarize_representation_groups(det_df)
        required = {"model", "representation_status", "test_sample_count", "tp", "fn", "detection_rate_pct"}
        assert required.issubset(set(grp_df.columns))

    def test_aggregate_tp_plus_fn_equals_test_sample_count(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        det_df = self._make_det_df(synthetic_pipeline, r2l_composition_train, r2l_composition_test)
        grp_df = exp004.summarize_representation_groups(det_df)
        for _, r in grp_df.iterrows():
            assert r["tp"] + r["fn"] == r["test_sample_count"], (
                f"Aggregate TP + FN != test_sample_count for model={r['model']}, "
                f"status={r['representation_status']}"
            )

    def test_aggregate_dr_equals_tp_over_n(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        det_df = self._make_det_df(synthetic_pipeline, r2l_composition_train, r2l_composition_test)
        grp_df = exp004.summarize_representation_groups(det_df)
        for _, r in grp_df.iterrows():
            if r["test_sample_count"] > 0:
                expected_dr = round(r["tp"] / r["test_sample_count"] * 100, 2)
                assert r["detection_rate_pct"] == pytest.approx(expected_dr, abs=0.01)

    def test_groups_include_shared_and_test_only(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        det_df = self._make_det_df(synthetic_pipeline, r2l_composition_train, r2l_composition_test)
        grp_df = exp004.summarize_representation_groups(det_df)
        statuses = set(grp_df["representation_status"])
        assert "shared" in statuses
        assert "test_only" in statuses

    def test_train_only_excluded_from_groups(
        self,
        synthetic_pipeline,
        r2l_composition_train: pd.DataFrame,
        r2l_composition_test: pd.DataFrame,
    ) -> None:
        """Train-only group has no test samples; should not appear in the summary."""
        det_df = self._make_det_df(synthetic_pipeline, r2l_composition_train, r2l_composition_test)
        grp_df = exp004.summarize_representation_groups(det_df)
        # train_only labels have test_count == 0; summarize should filter them out
        assert "train_only" not in grp_df["representation_status"].values
