"""
Unit tests for EXP-003: Minority Attack Category Sensitivity
============================================================
Tests verify:
  - Attack category mapping (taxonomy) correctness
  - Programmatic category distribution computation
  - Feature column extraction and metadata exclusion
  - Subgroup metric calculation (TP, FN, Detection Rate, Precision, F1)
  - Absence of Normal traffic from attack category detection metrics
  - Pipeline fitting and execution on synthetic data
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Load EXP-003 module dynamically
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
_exp003_path = REPO_ROOT / "experiments" / "EXP-003" / "run.py"
_spec = importlib.util.spec_from_file_location("exp003", _exp003_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec for {_exp003_path}")
exp003 = importlib.util.module_from_spec(_spec)
sys.modules["exp003"] = exp003
_spec.loader.exec_module(exp003)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal 43-column NSL-KDD DataFrame from a list of override dicts."""
    base = {col: 0 for col in exp003.NSL_KDD_FEATURE_NAMES}
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
    return pd.DataFrame(records, columns=exp003.NSL_KDD_FEATURE_NAMES)


@pytest.fixture
def synthetic_train() -> pd.DataFrame:
    return _make_df([
        {"label": "normal", "src_bytes": 100, "duration": 1},
        {"label": "normal", "src_bytes": 120, "duration": 2},
        {"label": "neptune", "src_bytes": 500, "duration": 0},      # DoS
        {"label": "satan", "src_bytes": 0, "duration": 0},          # Probe
        {"label": "guess_passwd", "src_bytes": 50, "duration": 1},  # R2L
        {"label": "buffer_overflow", "src_bytes": 200, "duration": 3}, # U2R
    ])


@pytest.fixture
def synthetic_test() -> pd.DataFrame:
    return _make_df([
        {"label": "normal", "src_bytes": 110, "duration": 1},
        {"label": "neptune", "src_bytes": 500, "duration": 0},      # DoS
        {"label": "apache2", "src_bytes": 600, "duration": 0},      # DoS (novel)
        {"label": "mscan", "src_bytes": 10, "duration": 0},         # Probe (novel)
        {"label": "httptunnel", "src_bytes": 300, "duration": 2},   # R2L (novel)
        {"label": "rootkit", "src_bytes": 150, "duration": 1},      # U2R
    ])


# ---------------------------------------------------------------------------
# TestCategoryMapping
# ---------------------------------------------------------------------------

class TestCategoryMapping:
    """Verify attack category taxonomy mapping."""

    def test_normal_maps_to_normal(self) -> None:
        s = pd.Series(["normal", " normal "])
        mapped = exp003.map_attack_category(s)
        assert list(mapped) == ["Normal", "Normal"]

    def test_dos_mapping(self) -> None:
        s = pd.Series(["back", "neptune", "smurf", "apache2", "mailbomb"])
        mapped = exp003.map_attack_category(s)
        assert list(mapped) == ["DoS"] * 5

    def test_probe_mapping(self) -> None:
        s = pd.Series(["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"])
        mapped = exp003.map_attack_category(s)
        assert list(mapped) == ["Probe"] * 6

    def test_r2l_mapping(self) -> None:
        s = pd.Series(["guess_passwd", "warezmaster", "httptunnel", "snmpguess"])
        mapped = exp003.map_attack_category(s)
        assert list(mapped) == ["R2L"] * 4

    def test_u2r_mapping(self) -> None:
        s = pd.Series(["buffer_overflow", "rootkit", "ps", "sqlattack", "xterm"])
        mapped = exp003.map_attack_category(s)
        assert list(mapped) == ["U2R"] * 5

    def test_httptunnel_is_r2l(self) -> None:
        assert exp003.ATTACK_CATEGORY_MAP.get("httptunnel") == "R2L"


# ---------------------------------------------------------------------------
# TestDistributionComputation
# ---------------------------------------------------------------------------

class TestDistributionComputation:
    """Verify programmatic distribution computation."""

    def test_distribution_returns_dataframe(
        self, synthetic_train: pd.DataFrame, synthetic_test: pd.DataFrame
    ) -> None:
        df = exp003.compute_category_distribution(synthetic_train, synthetic_test)
        assert isinstance(df, pd.DataFrame)
        assert set(df["category"]) == {"Normal", "DoS", "Probe", "R2L", "U2R"}

    def test_distribution_counts_match_inputs(
        self, synthetic_train: pd.DataFrame, synthetic_test: pd.DataFrame
    ) -> None:
        df = exp003.compute_category_distribution(synthetic_train, synthetic_test)
        cat_counts = df.set_index("category")["train_count"].to_dict()
        assert cat_counts["Normal"] == 2
        assert cat_counts["DoS"] == 1
        assert cat_counts["Probe"] == 1
        assert cat_counts["R2L"] == 1
        assert cat_counts["U2R"] == 1

    def test_attack_shares_sum_to_100(
        self, synthetic_train: pd.DataFrame, synthetic_test: pd.DataFrame
    ) -> None:
        df = exp003.compute_category_distribution(synthetic_train, synthetic_test)
        attack_rows = df[df["category"] != "Normal"]
        assert attack_rows["train_pct_of_attacks"].sum() == pytest.approx(100.0)
        assert attack_rows["test_pct_of_attacks"].sum() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# TestFeatureExtraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    """Verify feature selection and metadata exclusion."""

    def test_difficulty_excluded(self, synthetic_train: pd.DataFrame) -> None:
        cols = exp003.get_feature_columns(synthetic_train)
        assert "difficulty" not in cols

    def test_label_excluded(self, synthetic_train: pd.DataFrame) -> None:
        cols = exp003.get_feature_columns(synthetic_train)
        assert "label" not in cols

    def test_num_outbound_cmds_excluded(self, synthetic_train: pd.DataFrame) -> None:
        cols = exp003.get_feature_columns(synthetic_train)
        assert "num_outbound_cmds" not in cols

    def test_modelling_feature_count_is_40(self, synthetic_train: pd.DataFrame) -> None:
        cols = exp003.get_feature_columns(synthetic_train)
        assert len(cols) == 40


# ---------------------------------------------------------------------------
# TestSubgroupEvaluation
# ---------------------------------------------------------------------------

class TestSubgroupEvaluation:
    """Verify category sensitivity evaluation logic."""

    def test_attack_categories_contain_no_normal(
        self, synthetic_train: pd.DataFrame, synthetic_test: pd.DataFrame
    ) -> None:
        feature_cols = exp003.get_feature_columns(synthetic_train)
        model = DecisionTreeClassifier(random_state=42)
        pipe = exp003.build_pipeline(model, feature_cols)
        X_tr = synthetic_train[feature_cols].values
        y_tr = exp003.make_binary_target(synthetic_train["label"]).values
        pipe.fit(X_tr, y_tr)

        results = exp003.evaluate_model_category_sensitivity(pipe, synthetic_test, feature_cols)
        assert "Normal" not in results["categories"]
        for cat in ["DoS", "Probe", "R2L", "U2R"]:
            assert cat in results["categories"]

    def test_additivity_of_tp_and_fn(
        self, synthetic_train: pd.DataFrame, synthetic_test: pd.DataFrame
    ) -> None:
        feature_cols = exp003.get_feature_columns(synthetic_train)
        model = DecisionTreeClassifier(random_state=42)
        pipe = exp003.build_pipeline(model, feature_cols)
        X_tr = synthetic_train[feature_cols].values
        y_tr = exp003.make_binary_target(synthetic_train["label"]).values
        pipe.fit(X_tr, y_tr)

        results = exp003.evaluate_model_category_sensitivity(pipe, synthetic_test, feature_cols)
        for cat, m in results["categories"].items():
            assert m["tp"] + m["fn"] == m["sample_count"]
            expected_dr = m["tp"] / m["sample_count"]
            assert m["detection_rate"] == pytest.approx(expected_dr, abs=1e-5)

    def test_category_tp_sums_to_overall_tp(
        self, synthetic_train: pd.DataFrame, synthetic_test: pd.DataFrame
    ) -> None:
        feature_cols = exp003.get_feature_columns(synthetic_train)
        model = DecisionTreeClassifier(random_state=42)
        pipe = exp003.build_pipeline(model, feature_cols)
        X_tr = synthetic_train[feature_cols].values
        y_tr = exp003.make_binary_target(synthetic_train["label"]).values
        pipe.fit(X_tr, y_tr)

        results = exp003.evaluate_model_category_sensitivity(pipe, synthetic_test, feature_cols)
        cat_tp_sum = sum(m["tp"] for m in results["categories"].values())
        cat_fn_sum = sum(m["fn"] for m in results["categories"].values())
        assert cat_tp_sum == results["overall"]["tp"]
        assert cat_fn_sum == results["overall"]["fn"]
