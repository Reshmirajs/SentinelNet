"""
Unit tests for EXP-001: NSL-KDD Dataset Acquisition and Leakage Audit
======================================================================

Tests cover:
  - Feature schema constants
  - Data-loading correctness
  - Data-quality analysis
  - Class-distribution computation
  - Leakage-risk identification logic
  - Utility functions (SHA-256, environment info)

Run from the repository root:
    pytest tests/test_exp001.py -v
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import importlib.util

# ---------------------------------------------------------------------------
# Load EXP-001 module dynamically via importlib (supports hyphen in directory name)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
_exp001_path = REPO_ROOT / "experiments" / "EXP-001" / "run.py"
_spec = importlib.util.spec_from_file_location("exp001", _exp001_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load module spec for {_exp001_path}")
exp001 = importlib.util.module_from_spec(_spec)
sys.modules["exp001"] = exp001
_spec.loader.exec_module(exp001)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal NSL-KDD-like DataFrame with 43 columns
_MINIMAL_ROWS = [
    # duration, protocol_type, service, flag, src_bytes, dst_bytes, land,
    # wrong_fragment, urgent, hot, num_failed_logins, logged_in,
    # num_compromised, root_shell, su_attempted, num_root, num_file_creations,
    # num_shells, num_access_files, num_outbound_cmds, is_host_login,
    # is_guest_login, count, srv_count, serror_rate, srv_serror_rate,
    # rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate,
    # srv_diff_host_rate, dst_host_count, dst_host_srv_count,
    # dst_host_same_srv_rate, dst_host_diff_srv_rate,
    # dst_host_same_src_port_rate, dst_host_srv_diff_host_rate,
    # dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate,
    # dst_host_srv_rerror_rate, label, difficulty
    (0,  "tcp",  "http",     "SF", 100, 200, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     10, 10, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 200, 200, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     "normal", 20),
    (0,  "tcp",  "ftp_data", "SF",   0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1,  1,  0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 150,  25, 0.17, 0.03, 0.17, 0.0, 0.0, 0.0, 0.05, 0.0,
     "neptune", 19),
    (0,  "udp",  "other",    "SF", 146, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     13, 1,  0.0, 0.0, 0.0, 0.0, 0.08, 0.15, 0.0, 255, 1, 0.0, 0.6, 0.88, 0.0, 0.0, 0.0, 0.0, 0.0,
     "smurf", 15),
]


def _make_df(rows: list | None = None) -> pd.DataFrame:
    """Return a minimal DataFrame with the canonical NSL-KDD column names."""
    if rows is None:
        rows = _MINIMAL_ROWS
    return pd.DataFrame(rows, columns=exp001.NSL_KDD_FEATURE_NAMES)


@pytest.fixture
def minimal_train_df() -> pd.DataFrame:
    """Minimal training DataFrame with 3 rows."""
    return _make_df()


@pytest.fixture
def minimal_test_df() -> pd.DataFrame:
    """Minimal test DataFrame with 2 rows (subset of train rows)."""
    return _make_df(_MINIMAL_ROWS[:2])


# ---------------------------------------------------------------------------
# Schema / constant tests
# ---------------------------------------------------------------------------

class TestFeatureSchema:
    """Validate the NSL-KDD feature schema constants."""

    def test_total_columns_count(self) -> None:
        """NSL_KDD_FEATURE_NAMES should have exactly 43 entries."""
        assert len(exp001.NSL_KDD_FEATURE_NAMES) == 43

    def test_label_and_difficulty_present(self) -> None:
        """'label' and 'difficulty' must be in the feature name list."""
        assert "label" in exp001.NSL_KDD_FEATURE_NAMES
        assert "difficulty" in exp001.NSL_KDD_FEATURE_NAMES

    def test_categorical_features_subset_of_all(self) -> None:
        """All categorical features must be declared in the full name list."""
        for f in exp001.CATEGORICAL_FEATURES:
            assert f in exp001.NSL_KDD_FEATURE_NAMES, f"{f} missing from schema"

    def test_binary_features_subset_of_numerical(self) -> None:
        """Binary features should be included in NUMERICAL_FEATURES."""
        for f in exp001.BINARY_FEATURES:
            assert f in exp001.NUMERICAL_FEATURES, (
                f"{f} is binary but missing from NUMERICAL_FEATURES"
            )

    def test_no_overlap_categorical_numerical(self) -> None:
        """No feature should be listed as both categorical and numerical."""
        overlap = set(exp001.CATEGORICAL_FEATURES) & set(exp001.NUMERICAL_FEATURES)
        assert not overlap, f"Overlap: {overlap}"

    def test_attack_category_map_has_normal(self) -> None:
        """'normal' must map to 'Normal' category."""
        assert exp001.ATTACK_CATEGORY_MAP["normal"] == "Normal"

    def test_attack_categories_order_completeness(self) -> None:
        """All 5 expected attack category names must be in order list."""
        expected = {"Normal", "DoS", "Probe", "R2L", "U2R"}
        assert expected == set(exp001.ATTACK_CATEGORIES_ORDER)

    def test_no_feature_named_label_in_numerical(self) -> None:
        """'label' must not be in NUMERICAL_FEATURES."""
        assert "label" not in exp001.NUMERICAL_FEATURES

    def test_no_feature_named_difficulty_in_numerical(self) -> None:
        """'difficulty' must not be in NUMERICAL_FEATURES."""
        assert "difficulty" not in exp001.NUMERICAL_FEATURES

    def test_httptunnel_mapped_to_r2l(self) -> None:
        """httptunnel must map to R2L per MIT Lincoln Lab DARPA 1999 / KDD Cup 99 taxonomy."""
        assert exp001.ATTACK_CATEGORY_MAP["httptunnel"] == "R2L"


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestUtilities:
    """Tests for helper utilities."""

    def test_sha256_file(self, tmp_path: Path) -> None:
        """sha256_file should match hashlib reference for the same content."""
        content = b"sentinelnet test content 12345"
        f = tmp_path / "test.bin"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert exp001.sha256_file(f) == expected

    def test_sha256_empty_file(self, tmp_path: Path) -> None:
        """sha256_file should handle empty files without error."""
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        result = exp001.sha256_file(f)
        assert isinstance(result, str)
        assert len(result) == 64

    def test_get_environment_info_keys(self) -> None:
        """get_environment_info should return required keys."""
        info = exp001.get_environment_info()
        for key in ("python", "python_version_short", "pandas", "numpy", "matplotlib"):
            assert key in info, f"Missing key: {key}"

    def test_get_environment_info_values_nonempty(self) -> None:
        """All environment info values should be non-empty strings."""
        info = exp001.get_environment_info()
        for k, v in info.items():
            assert v, f"Empty value for key: {k}"


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestLoadDataset:
    """Tests for the load_dataset function."""

    def test_load_returns_dataframe(self, tmp_path: Path) -> None:
        """load_dataset should return a pandas DataFrame."""
        # Create a minimal valid NSL-KDD file (43 comma-separated values per row)
        row = ",".join(["0"] * 41 + ["normal", "20"])
        f = tmp_path / "mock_train.txt"
        f.write_text(f"{row}\n{row}\n", encoding="utf-8")
        df = exp001.load_dataset(f)
        assert isinstance(df, pd.DataFrame)

    def test_load_correct_column_count(self, tmp_path: Path) -> None:
        """Loaded DataFrame should have exactly 43 columns."""
        row = ",".join(["0"] * 41 + ["normal", "20"])
        f = tmp_path / "mock_train.txt"
        f.write_text(f"{row}\n", encoding="utf-8")
        df = exp001.load_dataset(f)
        assert df.shape[1] == 43

    def test_load_correct_column_names(self, tmp_path: Path) -> None:
        """Column names should match NSL_KDD_FEATURE_NAMES exactly."""
        row = ",".join(["0"] * 41 + ["normal", "20"])
        f = tmp_path / "mock_train.txt"
        f.write_text(f"{row}\n", encoding="utf-8")
        df = exp001.load_dataset(f)
        assert list(df.columns) == exp001.NSL_KDD_FEATURE_NAMES

    def test_load_raises_for_missing_file(self, tmp_path: Path) -> None:
        """load_dataset should raise FileNotFoundError if file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            exp001.load_dataset(tmp_path / "nonexistent.txt")

    def test_load_preserves_label_values(self, tmp_path: Path) -> None:
        """String labels should be loaded without modification."""
        row = ",".join(["0"] * 41 + ["neptune", "19"])
        f = tmp_path / "mock.txt"
        f.write_text(f"{row}\n", encoding="utf-8")
        df = exp001.load_dataset(f)
        assert df["label"].iloc[0] == "neptune"

    def test_load_preserves_difficulty_as_integer(self, tmp_path: Path) -> None:
        """Difficulty column should be loaded as integer dtype."""
        row = ",".join(["0"] * 41 + ["normal", "21"])
        f = tmp_path / "mock.txt"
        f.write_text(f"{row}\n", encoding="utf-8")
        df = exp001.load_dataset(f)
        assert pd.api.types.is_integer_dtype(df["difficulty"]), (
            f"Expected integer dtype for difficulty, got {df['difficulty'].dtype}"
        )


# ---------------------------------------------------------------------------
# Data quality tests
# ---------------------------------------------------------------------------

class TestDataQuality:
    """Tests for analyse_data_quality."""

    def test_returns_dict_with_required_keys(self, minimal_train_df: pd.DataFrame) -> None:
        """Output must contain all expected keys."""
        result = exp001.analyse_data_quality(minimal_train_df, "train")
        required = {
            "split", "total_rows", "total_feature_columns",
            "total_missing_values", "duplicate_rows_total",
            "constant_features", "near_constant_numerical_features",
        }
        assert required <= set(result.keys())

    def test_no_missing_values_in_clean_data(self, minimal_train_df: pd.DataFrame) -> None:
        """Clean DataFrame should report zero missing values."""
        result = exp001.analyse_data_quality(minimal_train_df, "train")
        assert result["total_missing_values"] == 0

    def test_detects_missing_values(self) -> None:
        """Injected NaN should be detected and counted."""
        df = _make_df()
        df.at[0, "duration"] = np.nan
        result = exp001.analyse_data_quality(df, "train")
        assert result["total_missing_values"] == 1

    def test_detects_duplicate_rows(self) -> None:
        """Duplicated rows should be counted."""
        df = _make_df(_MINIMAL_ROWS[:1] * 3)  # all 3 rows identical
        result = exp001.analyse_data_quality(df, "train")
        assert result["duplicate_rows_total"] == 3

    def test_detects_constant_feature(self) -> None:
        """A feature with only one unique value should appear in constant_features."""
        df = _make_df()
        df["num_outbound_cmds"] = 0  # NSL-KDD is known to have this constant
        result = exp001.analyse_data_quality(df, "train")
        assert "num_outbound_cmds" in result["constant_features"]

    def test_total_rows_matches_input(self, minimal_train_df: pd.DataFrame) -> None:
        """total_rows should match the number of rows in the input DataFrame."""
        result = exp001.analyse_data_quality(minimal_train_df, "train")
        assert result["total_rows"] == len(minimal_train_df)


# ---------------------------------------------------------------------------
# Class distribution tests
# ---------------------------------------------------------------------------

class TestClassDistribution:
    """Tests for compute_class_distribution."""

    def test_returns_dict_with_required_keys(self, minimal_train_df: pd.DataFrame) -> None:
        result = exp001.compute_class_distribution(minimal_train_df, "train")
        for key in ("total_samples", "num_unique_labels", "label_counts",
                    "category_counts", "normal_count", "attack_count"):
            assert key in result, f"Missing key: {key}"

    def test_total_samples_correct(self, minimal_train_df: pd.DataFrame) -> None:
        result = exp001.compute_class_distribution(minimal_train_df, "train")
        assert result["total_samples"] == len(minimal_train_df)

    def test_label_counts_sum_to_total(self, minimal_train_df: pd.DataFrame) -> None:
        result = exp001.compute_class_distribution(minimal_train_df, "train")
        assert sum(result["label_counts"].values()) == result["total_samples"]

    def test_normal_plus_attack_equals_total(self, minimal_train_df: pd.DataFrame) -> None:
        result = exp001.compute_class_distribution(minimal_train_df, "train")
        assert result["normal_count"] + result["attack_count"] == result["total_samples"]

    def test_label_pct_sums_to_100(self, minimal_train_df: pd.DataFrame) -> None:
        result = exp001.compute_class_distribution(minimal_train_df, "train")
        total_pct = sum(result["label_pct"].values())
        assert abs(total_pct - 100.0) < 0.01, f"Percentages sum to {total_pct}"

    def test_category_mapping_applied(self, minimal_train_df: pd.DataFrame) -> None:
        """neptune and smurf should both map to DoS category."""
        result = exp001.compute_class_distribution(minimal_train_df, "train")
        assert "DoS" in result["category_counts"]

    def test_normal_count_correct(self) -> None:
        """Only one normal row in minimal fixture."""
        df = _make_df()
        result = exp001.compute_class_distribution(df, "train")
        assert result["normal_count"] == 1


# ---------------------------------------------------------------------------
# Overlap analysis tests
# ---------------------------------------------------------------------------

class TestTrainTestOverlap:
    """Tests for analyse_train_test_overlap."""

    def test_returns_required_keys(
        self, minimal_train_df: pd.DataFrame, minimal_test_df: pd.DataFrame
    ) -> None:
        result = exp001.analyse_train_test_overlap(minimal_train_df, minimal_test_df)
        for key in (
            "shared_labels",
            "train_only_labels",
            "test_only_labels",
            "unique_overlapping_patterns",
            "total_overlapping_test_samples",
            "total_overlapping_test_pct",
            "exact_row_overlap_count",
        ):
            assert key in result, f"Missing key: {key}"

    def test_exact_overlap_when_identical(self, minimal_train_df: pd.DataFrame) -> None:
        """When test == train, all test rows should overlap."""
        result = exp001.analyse_train_test_overlap(minimal_train_df, minimal_train_df)
        # Overlap count should be > 0
        assert result["exact_row_overlap_count"] > 0
        assert result["unique_overlapping_patterns"] == len(minimal_train_df.drop_duplicates())
        assert result["total_overlapping_test_samples"] == len(minimal_train_df)
        assert result["total_overlapping_test_pct"] == 100.0

    def test_no_overlap_with_disjoint_data(self) -> None:
        """When feature values differ completely, overlap should be zero."""
        train = _make_df(_MINIMAL_ROWS[:1])
        # Build a clearly different test row
        diff_row = list(_MINIMAL_ROWS[0])
        diff_row[4] = 99999  # src_bytes differs
        diff_row[-2] = "portsweep"
        test = _make_df([tuple(diff_row)])
        result = exp001.analyse_train_test_overlap(train, test)
        assert result["exact_row_overlap_count"] == 0

    def test_test_only_labels_detected(self) -> None:
        """Labels in test but not train should appear in test_only_labels."""
        train = _make_df(_MINIMAL_ROWS[:1])  # only 'normal'
        test_row = list(_MINIMAL_ROWS[1])    # 'neptune'
        test = _make_df([tuple(test_row)])
        result = exp001.analyse_train_test_overlap(train, test)
        assert "neptune" in result["test_only_labels"]

    def test_train_only_labels_detected(self) -> None:
        """Labels in train but not test should appear in train_only_labels."""
        train = _make_df(_MINIMAL_ROWS)      # normal, neptune, smurf
        test = _make_df(_MINIMAL_ROWS[:1])   # only normal
        result = exp001.analyse_train_test_overlap(train, test)
        assert "neptune" in result["train_only_labels"]
        assert "smurf"   in result["train_only_labels"]


# ---------------------------------------------------------------------------
# Leakage risk tests
# ---------------------------------------------------------------------------

class TestLeakageRisks:
    """Tests for identify_leakage_risks."""

    def _run_risks(
        self, overlap: dict, train_q: dict, test_q: dict
    ) -> list[dict]:
        return exp001.identify_leakage_risks(overlap, train_q, test_q)

    def _base_quality(self) -> dict:
        return {
            "constant_features": [],
            "near_constant_numerical_features": [],
            "negative_value_anomalies": {},
        }

    def _base_overlap(self) -> dict:
        return {
            "test_only_labels": [],
            "exact_row_overlap_count": 0,
            "exact_row_overlap_pct_of_test": 0.0,
            "train_category_pct": {"Normal": 50.0, "DoS": 50.0},
            "test_category_pct":  {"Normal": 50.0, "DoS": 50.0},
        }

    def test_leak001_always_present(self) -> None:
        """LEAK-001 (difficulty column) must always be reported."""
        risks = self._run_risks(
            self._base_overlap(),
            self._base_quality(),
            self._base_quality(),
        )
        risk_ids = [r["risk_id"] for r in risks]
        assert "LEAK-001" in risk_ids

    def test_leak001_severity_high(self) -> None:
        """LEAK-001 must be HIGH severity."""
        risks = self._run_risks(
            self._base_overlap(),
            self._base_quality(),
            self._base_quality(),
        )
        leak001 = next(r for r in risks if r["risk_id"] == "LEAK-001")
        assert leak001["severity"] == "HIGH"

    def test_leak002_triggered_by_test_only_labels(self) -> None:
        """LEAK-002 should appear when test has labels not in train."""
        overlap = self._base_overlap()
        overlap["test_only_labels"] = ["novel_attack"]
        risks = self._run_risks(overlap, self._base_quality(), self._base_quality())
        assert any(r["risk_id"] == "LEAK-002" for r in risks)

    def test_leak002_absent_without_test_only_labels(self) -> None:
        """LEAK-002 should not appear when all test labels are in train."""
        risks = self._run_risks(
            self._base_overlap(),
            self._base_quality(),
            self._base_quality(),
        )
        assert not any(r["risk_id"] == "LEAK-002" for r in risks)

    def test_leak003_triggered_by_row_overlap(self) -> None:
        """LEAK-003 should appear when there is exact row overlap."""
        overlap = self._base_overlap()
        overlap["exact_row_overlap_count"] = 10
        overlap["exact_row_overlap_pct_of_test"] = 5.0
        risks = self._run_risks(overlap, self._base_quality(), self._base_quality())
        assert any(r["risk_id"] == "LEAK-003" for r in risks)

    def test_leak004_triggered_by_constant_features(self) -> None:
        """LEAK-004 should appear when constant features exist."""
        quality = self._base_quality()
        quality["constant_features"] = ["num_outbound_cmds"]
        risks = self._run_risks(self._base_overlap(), quality, self._base_quality())
        assert any(r["risk_id"] == "LEAK-004" for r in risks)

    def test_leak005_triggered_by_large_distribution_shift(self) -> None:
        """LEAK-005 should appear when category distribution shifts > 5 pp."""
        overlap = self._base_overlap()
        overlap["train_category_pct"] = {"Normal": 70.0, "DoS": 30.0}
        overlap["test_category_pct"]  = {"Normal": 40.0, "DoS": 60.0}
        risks = self._run_risks(overlap, self._base_quality(), self._base_quality())
        assert any(r["risk_id"] == "LEAK-005" for r in risks)

    def test_all_risks_have_required_fields(self) -> None:
        """Every risk dict must have risk_id, severity, description, evidence, mitigation."""
        overlap = self._base_overlap()
        overlap["test_only_labels"] = ["test_attack"]
        overlap["exact_row_overlap_count"] = 5
        overlap["exact_row_overlap_pct_of_test"] = 2.0
        q = self._base_quality()
        q["constant_features"] = ["num_outbound_cmds"]
        risks = self._run_risks(overlap, q, q)
        for risk in risks:
            for field in ("risk_id", "severity", "description", "evidence", "mitigation"):
                assert field in risk, f"Field '{field}' missing from {risk.get('risk_id')}"


# ---------------------------------------------------------------------------
# dtype detection tests
# ---------------------------------------------------------------------------

class TestDetectDtypes:
    """Tests for detect_dtypes."""

    def test_label_type(self, minimal_train_df: pd.DataFrame) -> None:
        dtypes = exp001.detect_dtypes(minimal_train_df)
        assert dtypes["label"] == "target_label"

    def test_difficulty_type(self, minimal_train_df: pd.DataFrame) -> None:
        dtypes = exp001.detect_dtypes(minimal_train_df)
        assert dtypes["difficulty"] == "metadata_integer"

    def test_protocol_type_categorical(self, minimal_train_df: pd.DataFrame) -> None:
        dtypes = exp001.detect_dtypes(minimal_train_df)
        assert dtypes["protocol_type"] == "categorical_nominal"

    def test_logged_in_binary(self, minimal_train_df: pd.DataFrame) -> None:
        dtypes = exp001.detect_dtypes(minimal_train_df)
        assert dtypes["logged_in"] == "binary_integer"

    def test_duration_numerical(self, minimal_train_df: pd.DataFrame) -> None:
        dtypes = exp001.detect_dtypes(minimal_train_df)
        assert dtypes["duration"] == "numerical_continuous_or_count"

    def test_all_columns_covered(self, minimal_train_df: pd.DataFrame) -> None:
        dtypes = exp001.detect_dtypes(minimal_train_df)
        assert set(dtypes.keys()) == set(exp001.NSL_KDD_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Feature schema table tests
# ---------------------------------------------------------------------------

class TestFeatureSchemaTable:
    """Tests for build_feature_schema_table."""

    def test_returns_dataframe(self, minimal_train_df: pd.DataFrame) -> None:
        schema = exp001.build_feature_schema_table(minimal_train_df)
        assert isinstance(schema, pd.DataFrame)

    def test_one_row_per_column(self, minimal_train_df: pd.DataFrame) -> None:
        schema = exp001.build_feature_schema_table(minimal_train_df)
        assert len(schema) == 43

    def test_required_columns_present(self, minimal_train_df: pd.DataFrame) -> None:
        schema = exp001.build_feature_schema_table(minimal_train_df)
        for col in ("feature_name", "semantic_type", "pandas_dtype", "num_unique_values"):
            assert col in schema.columns


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end smoke test using mock dataset files."""

    def test_full_pipeline_runs_without_error(self, tmp_path: Path) -> None:
        """Run the full EXP-001 pipeline on mock data and check outputs exist."""
        # Generate a small mock dataset (50 rows, all 43 columns)
        rng = np.random.default_rng(42)
        n = 50
        labels = (["normal"] * 20 + ["neptune"] * 15
                  + ["smurf"] * 10 + ["ipsweep"] * 5)
        rows = []
        for i, lbl in enumerate(labels):
            numerical = list(rng.integers(0, 100, size=9))   # first 9
            numerical += list(rng.integers(0, 10, size=13))  # middle 13
            numerical += list(rng.random(9).round(2))        # rates
            numerical += list(rng.integers(0, 255, size=2))  # dst_host counts
            numerical += list(rng.random(8).round(2))        # dst_host rates
            row = [numerical[0], "tcp", "http", "SF"] + numerical[1:5] + [0, 0, 0]
            row += numerical[5:18]
            row += numerical[18:]
            row += [lbl, rng.integers(1, 21)]
            rows.append(row)

        # Pad / trim each row to exactly 43 columns
        padded_rows = []
        for r in rows:
            if len(r) < 43:
                r = r + [0] * (43 - len(r))
            padded_rows.append(r[:43])

        train_path = tmp_path / "KDDTrain+.txt"
        test_path  = tmp_path / "KDDTest+.txt"

        def write_csv(path: Path, data: list) -> None:
            lines = [",".join(str(v) for v in row) for row in data]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        write_csv(train_path, padded_rows[:40])
        write_csv(test_path,  padded_rows[40:])

        # Patch output directories to tmp_path
        results_dir = tmp_path / "results"
        fig_dir     = results_dir / "figures"
        metrics_dir = results_dir / "metrics"
        tables_dir  = results_dir / "tables"

        with (
            patch.object(exp001, "RESULTS_DIR", results_dir),
            patch.object(exp001, "FIG_DIR",     fig_dir),
            patch.object(exp001, "METRICS_DIR", metrics_dir),
            patch.object(exp001, "TABLES_DIR",  tables_dir),
        ):
            with patch("sys.argv", [
                "run.py",
                "--train", str(train_path),
                "--test",  str(test_path),
                "--seed",  "42",
            ]):
                exp001.main()

        # Verify key outputs exist
        assert (metrics_dir / "summary_statistics.json").exists()
        assert (metrics_dir / "class_distribution_train.json").exists()
        assert (metrics_dir / "class_distribution_test.json").exists()
        assert (metrics_dir / "data_quality_report.json").exists()
        assert (metrics_dir / "leakage_risks.json").exists()
        assert (results_dir / "audit_report.md").exists()

        # Verify summary JSON is valid
        summary = json.loads((metrics_dir / "summary_statistics.json").read_text())
        assert summary["experiment_id"] == "EXP-001"
        assert summary["seed"] == 42

        # Verify no fabricated values: total_samples must match actual row count
        train_dist = json.loads((metrics_dir / "class_distribution_train.json").read_text())
        assert train_dist["total_samples"] == 40

        test_dist = json.loads((metrics_dir / "class_distribution_test.json").read_text())
        assert test_dist["total_samples"] == 10
