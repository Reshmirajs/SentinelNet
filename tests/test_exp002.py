"""
Unit tests for EXP-002: Supervised Baseline Binary IDS
=======================================================
Tests verify:
  - Feature column derivation and metadata exclusion
  - Binary target encoding
  - Overlap mask detection
  - Cohort construction logic
  - Metric computation correctness
  - Pipeline build / train-only preprocessing constraint
  - Full pipeline integration (smoke test on tiny synthetic data)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Load EXP-002 module dynamically (hyphen-safe)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
_exp002_path = REPO_ROOT / "experiments" / "EXP-002" / "run.py"
_spec = importlib.util.spec_from_file_location("exp002", _exp002_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec for {_exp002_path}")
exp002 = importlib.util.module_from_spec(_spec)
sys.modules["exp002"] = exp002
_spec.loader.exec_module(exp002)


# ---------------------------------------------------------------------------
# Helpers — minimal NSL-KDD-like DataFrames
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal 43-column NSL-KDD DataFrame from a list of override dicts."""
    base = {col: 0 for col in exp002.NSL_KDD_FEATURE_NAMES}
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
    return pd.DataFrame(records, columns=exp002.NSL_KDD_FEATURE_NAMES)


_TRAIN_ROWS = [
    {"label": "normal", "src_bytes": 100, "dst_bytes": 200, "duration": 1},
    {"label": "neptune", "src_bytes": 500, "dst_bytes": 0, "duration": 0, "protocol_type": "tcp"},
    {"label": "smurf", "src_bytes": 1000, "dst_bytes": 0, "duration": 0, "protocol_type": "udp"},
]

_TEST_ROWS = [
    # Exact copy of the "neptune" training row → should be in overlapping cohort
    {"label": "neptune", "src_bytes": 500, "dst_bytes": 0, "duration": 0, "protocol_type": "tcp"},
    # Unique normal traffic
    {"label": "normal", "src_bytes": 999, "dst_bytes": 888, "duration": 5},
    # Novel label (not in train)
    {"label": "apache2", "src_bytes": 777, "dst_bytes": 0, "duration": 0},
]


@pytest.fixture
def train_df() -> pd.DataFrame:
    return _make_df(_TRAIN_ROWS)


@pytest.fixture
def test_df() -> pd.DataFrame:
    return _make_df(_TEST_ROWS)


@pytest.fixture
def feature_cols(train_df: pd.DataFrame) -> list[str]:
    return exp002.get_feature_columns(train_df)


# ---------------------------------------------------------------------------
# TestBinaryTarget
# ---------------------------------------------------------------------------

class TestBinaryTarget:
    """Tests for make_binary_target."""

    def test_normal_maps_to_zero(self) -> None:
        labels = pd.Series(["normal", "normal"])
        result = exp002.make_binary_target(labels)
        assert list(result) == [0, 0]

    def test_attack_maps_to_one(self) -> None:
        labels = pd.Series(["neptune", "smurf", "apache2"])
        result = exp002.make_binary_target(labels)
        assert list(result) == [1, 1, 1]

    def test_mixed(self) -> None:
        labels = pd.Series(["normal", "neptune", "normal", "back"])
        result = exp002.make_binary_target(labels)
        assert list(result) == [0, 1, 0, 1]

    def test_returns_integer_dtype(self) -> None:
        labels = pd.Series(["normal", "neptune"])
        result = exp002.make_binary_target(labels)
        assert result.dtype in (np.int32, np.int64, int, "int64", "int32")

    def test_strips_whitespace(self) -> None:
        labels = pd.Series(["  normal  ", " neptune "])
        result = exp002.make_binary_target(labels)
        assert list(result) == [0, 1]


# ---------------------------------------------------------------------------
# TestFeatureColumns
# ---------------------------------------------------------------------------

class TestFeatureColumns:
    """Tests for get_feature_columns."""

    def test_difficulty_excluded(self, train_df: pd.DataFrame) -> None:
        cols = exp002.get_feature_columns(train_df)
        assert "difficulty" not in cols

    def test_label_excluded(self, train_df: pd.DataFrame) -> None:
        cols = exp002.get_feature_columns(train_df)
        assert "label" not in cols

    def test_num_outbound_cmds_excluded(self, train_df: pd.DataFrame) -> None:
        cols = exp002.get_feature_columns(train_df)
        assert "num_outbound_cmds" not in cols

    def test_protocol_type_included(self, train_df: pd.DataFrame) -> None:
        cols = exp002.get_feature_columns(train_df)
        assert "protocol_type" in cols

    def test_count_is_40(self, train_df: pd.DataFrame) -> None:
        """41 total features − 1 (num_outbound_cmds) = 40 modelling features."""
        cols = exp002.get_feature_columns(train_df)
        assert len(cols) == 40


# ---------------------------------------------------------------------------
# TestOverlapMask
# ---------------------------------------------------------------------------

class TestOverlapMask:
    """Tests for identify_overlap_mask."""

    def test_exact_copy_flagged(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        mask = exp002.identify_overlap_mask(train_df, test_df, feature_cols)
        # Row 0 of test_df is an exact copy of the neptune training row
        assert mask.iloc[0] is np.bool_(True)

    def test_unique_row_not_flagged(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        mask = exp002.identify_overlap_mask(train_df, test_df, feature_cols)
        # Row 1 of test_df has distinct feature values
        assert mask.iloc[1] is np.bool_(False)

    def test_novel_label_not_flagged(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        mask = exp002.identify_overlap_mask(train_df, test_df, feature_cols)
        # Row 2 has label "apache2" which is not in train
        assert mask.iloc[2] is np.bool_(False)

    def test_mask_length_equals_test(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        mask = exp002.identify_overlap_mask(train_df, test_df, feature_cols)
        assert len(mask) == len(test_df)

    def test_no_overlap_when_disjoint(self, feature_cols: list[str]) -> None:
        tr = _make_df([{"label": "normal", "src_bytes": 1}])
        te = _make_df([{"label": "normal", "src_bytes": 99999}])
        mask = exp002.identify_overlap_mask(tr, te, feature_cols)
        assert not mask.any()


# ---------------------------------------------------------------------------
# TestBuildCohorts
# ---------------------------------------------------------------------------

class TestBuildCohorts:
    """Tests for build_cohorts."""

    def test_full_test_contains_all_rows(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        mask = exp002.identify_overlap_mask(train_df, test_df, feature_cols)
        train_labels = set(str(x).strip() for x in train_df["label"].unique())
        cohorts = exp002.build_cohorts(test_df, mask, train_labels)
        assert len(cohorts["full_test"]) == len(test_df)

    def test_overlap_and_non_overlap_partition_full_test(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        mask = exp002.identify_overlap_mask(train_df, test_df, feature_cols)
        train_labels = set(str(x).strip() for x in train_df["label"].unique())
        cohorts = exp002.build_cohorts(test_df, mask, train_labels)
        assert len(cohorts["overlapping"]) + len(cohorts["non_overlapping"]) == len(test_df)

    def test_novel_attacks_identified(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        mask = exp002.identify_overlap_mask(train_df, test_df, feature_cols)
        train_labels = set(str(x).strip() for x in train_df["label"].unique())
        cohorts = exp002.build_cohorts(test_df, mask, train_labels)
        # apache2 appears only in test; neptune is in train → known
        novel_labels = set(
            str(x).strip()
            for x in test_df.loc[cohorts["novel_attacks"], "label"].unique()
        )
        assert "apache2" in novel_labels
        assert "neptune" not in novel_labels

    def test_normal_rows_not_in_attack_cohorts(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        mask = exp002.identify_overlap_mask(train_df, test_df, feature_cols)
        train_labels = set(str(x).strip() for x in train_df["label"].unique())
        cohorts = exp002.build_cohorts(test_df, mask, train_labels)
        for cohort_name in ("known_attacks", "novel_attacks"):
            labels_in_cohort = test_df.loc[cohorts[cohort_name], "label"].str.strip().unique()
            assert "normal" not in labels_in_cohort, (
                f"Normal label found in {cohort_name}"
            )


# ---------------------------------------------------------------------------
# TestComputeMetrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for compute_metrics."""

    def test_perfect_classifier_recall_one(self) -> None:
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        m = exp002.compute_metrics(y_true, y_pred, None, "test")
        assert m["recall_detection_rate"] == pytest.approx(1.0)

    def test_perfect_classifier_far_zero(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        m = exp002.compute_metrics(y_true, y_pred, None, "test")
        assert m["false_alarm_rate"] == pytest.approx(0.0)

    def test_all_wrong_recall_zero(self) -> None:
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        m = exp002.compute_metrics(y_true, y_pred, None, "test")
        assert m["recall_detection_rate"] == pytest.approx(0.0)

    def test_far_computed_correctly(self) -> None:
        # 2 normal, both predicted attack → FAR = 1.0
        y_true = np.array([0, 0, 1])
        y_pred = np.array([1, 1, 1])
        m = exp002.compute_metrics(y_true, y_pred, None, "test")
        assert m["false_alarm_rate"] == pytest.approx(1.0)

    def test_n_samples_correct(self) -> None:
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        m = exp002.compute_metrics(y_true, y_pred, None, "test")
        assert m["n_samples"] == 5

    def test_roc_auc_with_probabilities(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        m = exp002.compute_metrics(y_true, y_pred, y_prob, "test")
        assert "roc_auc" in m
        assert m["roc_auc"] == pytest.approx(1.0)

    def test_empty_cohort_returns_gracefully(self) -> None:
        m = exp002.compute_metrics(
            np.array([]), np.array([]), None, "empty"
        )
        assert m["n_samples"] == 0


# ---------------------------------------------------------------------------
# TestBuildPipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    """Tests for build_pipeline."""

    def test_pipeline_has_preprocessor_and_classifier(
        self, feature_cols: list[str]
    ) -> None:
        model = LogisticRegression(max_iter=100, random_state=42)
        pipe = exp002.build_pipeline(model, feature_cols)
        assert "preprocessor" in pipe.named_steps
        assert "classifier" in pipe.named_steps

    def test_pipeline_fits_on_training_data(
        self, train_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        """Verify that the pipeline can be fitted on training data without error."""
        model = LogisticRegression(max_iter=500, random_state=42)
        pipe = exp002.build_pipeline(model, feature_cols)
        X = train_df[feature_cols].values
        y = exp002.make_binary_target(train_df["label"]).values
        # Should not raise
        pipe.fit(X, y)

    def test_pipeline_predicts_binary(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        model = LogisticRegression(max_iter=500, random_state=42)
        pipe = exp002.build_pipeline(model, feature_cols)
        X_tr = train_df[feature_cols].values
        y_tr = exp002.make_binary_target(train_df["label"]).values
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(test_df[feature_cols].values)
        assert set(preds).issubset({0, 1})


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------

class TestIntegration:
    """Smoke test: full pipeline on minimal synthetic data without real files."""

    def test_full_pipeline_does_not_raise(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        """Verify the core modelling loop runs end-to-end without error."""
        from sklearn.tree import DecisionTreeClassifier as DTC

        pipe = exp002.build_pipeline(DTC(random_state=42), feature_cols)
        X_tr = train_df[feature_cols].values
        y_tr = exp002.make_binary_target(train_df["label"]).values
        pipe.fit(X_tr, y_tr)

        mask = exp002.identify_overlap_mask(train_df, test_df, feature_cols)
        train_labels = set(str(x).strip() for x in train_df["label"].unique())
        cohorts = exp002.build_cohorts(test_df, mask, train_labels)

        results = exp002.evaluate_model_on_cohorts(pipe, test_df, cohorts, feature_cols)
        assert "full_test" in results
        assert results["full_test"]["n_samples"] == len(test_df)

    def test_novel_by_category_runs(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]
    ) -> None:
        from sklearn.tree import DecisionTreeClassifier as DTC

        pipe = exp002.build_pipeline(DTC(random_state=42), feature_cols)
        X_tr = train_df[feature_cols].values
        y_tr = exp002.make_binary_target(train_df["label"]).values
        pipe.fit(X_tr, y_tr)

        mask = exp002.identify_overlap_mask(train_df, test_df, feature_cols)
        train_labels = set(str(x).strip() for x in train_df["label"].unique())
        cohorts = exp002.build_cohorts(test_df, mask, train_labels)
        novel_labels = set(
            str(x).strip()
            for x in test_df.loc[cohorts["novel_attacks"], "label"].unique()
        )
        result = exp002.evaluate_novel_attacks_by_category(
            pipe, test_df, feature_cols, novel_labels
        )
        # apache2 → DoS category
        assert any("dos" in key for key in result)
