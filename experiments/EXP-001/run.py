"""
EXP-001: NSL-KDD Dataset Acquisition and Leakage Audit
=======================================================
SentinelNet Research Project

Purpose
-------
Perform a complete, reproducible audit of the raw NSL-KDD dataset files.
This script:
  - Loads KDDTrain+.txt and KDDTest+.txt
  - Documents the feature schema
  - Computes data-quality statistics (missing values, duplicates, constants)
  - Analyses class distributions and attack-category mappings
  - Measures train/test overlap at the row and label level
  - Identifies potential data-leakage risks
  - Saves machine-readable JSON/CSV outputs and publication-quality figures
  - Generates a full audit report in Markdown

Constraints (Research Constitution — AGENTS.md / CLAUDE.md)
-----------------------------------------------------------
  - NO model training
  - NO data downloading
  - NO raw-data modification
  - NO fabricated statistics
  - NO normalization, encoding, scaling, or resampling
  - Difficulty column treated as metadata only

Usage
-----
  python experiments/EXP-001/run.py [--train PATH] [--test PATH] [--seed INT]

Outputs
-------
  experiments/EXP-001/results/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for scripts

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import scipy

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("EXP-001")

# ---------------------------------------------------------------------------
# Constants — NSL-KDD Feature Schema
# ---------------------------------------------------------------------------

# 41 feature names (in column order), as defined in the NSL-KDD documentation.
# Source: Tavallaee et al. (2009), Table I.
NSL_KDD_FEATURE_NAMES: list[str] = [
    # Basic features of individual TCP connections (KDD features 1–9)
    "duration",               # f01  Length of the connection (seconds)
    "protocol_type",          # f02  Type of protocol: tcp, udp, icmp
    "service",                # f03  Network service on destination: http, ftp, etc.
    "flag",                   # f04  Normal or error status of the connection
    "src_bytes",              # f05  Bytes from source to destination
    "dst_bytes",              # f06  Bytes from destination to source
    "land",                   # f07  1 if src/dst host/port are the same; 0 otherwise
    "wrong_fragment",         # f08  Number of wrong fragments
    "urgent",                 # f09  Number of urgent packets
    # Content features within a connection (KDD features 10–22)
    "hot",                    # f10  Number of "hot" indicators
    "num_failed_logins",      # f11  Number of failed login attempts
    "logged_in",              # f12  1 if successfully logged in; 0 otherwise
    "num_compromised",        # f13  Number of compromised conditions
    "root_shell",             # f14  1 if root shell is obtained; 0 otherwise
    "su_attempted",           # f15  1 if "su root" command attempted; 0 otherwise
    "num_root",               # f16  Number of root accesses
    "num_file_creations",     # f17  Number of file creation operations
    "num_shells",             # f18  Number of shell prompts
    "num_access_files",       # f19  Number of operations on access control files
    "num_outbound_cmds",      # f20  Number of outbound commands in ftp session
    "is_host_login",          # f21  1 if login belongs to "hot" list; 0 otherwise
    "is_guest_login",         # f22  1 if the login is a guest login; 0 otherwise
    # Traffic features computed using a 2-second time window (KDD features 23–31)
    "count",                  # f23  Connections to same host in past 2 seconds
    "srv_count",              # f24  Connections to same service in past 2 seconds
    "serror_rate",            # f25  % connections with SYN errors (same host)
    "srv_serror_rate",        # f26  % connections with SYN errors (same service)
    "rerror_rate",            # f27  % connections with REJ errors (same host)
    "srv_rerror_rate",        # f28  % connections with REJ errors (same service)
    "same_srv_rate",          # f29  % connections to same service (same host)
    "diff_srv_rate",          # f30  % connections to different services (same host)
    "srv_diff_host_rate",     # f31  % connections to different hosts (same service)
    # Traffic features computed using a 100-connection window (KDD features 32–41)
    "dst_host_count",         # f32  Connections with same destination host
    "dst_host_srv_count",     # f33  Connections with same destination host and service
    "dst_host_same_srv_rate", # f34  % connections with same service (dst host)
    "dst_host_diff_srv_rate", # f35  % connections with diff services (dst host)
    "dst_host_same_src_port_rate",  # f36  % connections with same src port (dst host)
    "dst_host_srv_diff_host_rate",  # f37  % connections to diff hosts (dst host/srv)
    "dst_host_serror_rate",   # f38  % connections with SYN errors (dst host)
    "dst_host_srv_serror_rate",     # f39  % connections with SYN errors (dst host/srv)
    "dst_host_rerror_rate",   # f40  % connections with REJ errors (dst host)
    "dst_host_srv_rerror_rate",     # f41  % connections with REJ errors (dst host/srv)
    # Target and metadata
    "label",                  # Column 42: attack label (string)
    "difficulty",             # Column 43: difficulty score (int, metadata only)
]

# Categorical feature names (nominal)
CATEGORICAL_FEATURES: list[str] = ["protocol_type", "service", "flag"]

# Binary features (0/1 integer semantics)
BINARY_FEATURES: list[str] = [
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
]

# Numerical feature names (continuous or discrete counts)
NUMERICAL_FEATURES: list[str] = [
    f for f in NSL_KDD_FEATURE_NAMES
    if f not in CATEGORICAL_FEATURES + ["label", "difficulty"]
]

# NSL-KDD 5-category attack taxonomy
# Taxonomy provenance: MIT Lincoln Laboratory DARPA 1998/1999 Intrusion Detection Evaluations
# and KDD Cup 1999 task specification, as adopted in NSL-KDD (Tavallaee et al., 2009).
# Note: Tavallaee et al. (2009) evaluates aggregate category distributions but inherits the
# individual attack mappings from the DARPA 1999 / KDD Cup 99 ground-truth taxonomy.
ATTACK_CATEGORY_MAP: dict[str, str] = {
    "normal": "Normal",
    # DoS attacks
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
    # Probe attacks
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L attacks (Remote to Local: unauthorized remote access to local system)
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R attacks (User to Root: unauthorized escalation of local privilege)
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R",
    "xterm": "U2R",
}

ATTACK_CATEGORIES_ORDER: list[str] = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Colour palette (accessible, colourblind-friendly)
PALETTE: dict[str, str] = {
    "Normal": "#2196F3",
    "DoS":    "#F44336",
    "Probe":  "#FF9800",
    "R2L":    "#9C27B0",
    "U2R":    "#4CAF50",
    "Other":  "#607D8B",
}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"

DEFAULT_TRAIN = REPO_ROOT / "data" / "raw" / "KDDTrain+.txt"
DEFAULT_TEST  = REPO_ROOT / "data" / "raw" / "KDDTest+.txt"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file for integrity verification.

    Args:
        path: Absolute path to the file.

    Returns:
        Lowercase hex string of the SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_environment_info() -> dict:
    """Collect Python version and key package version strings.

    Returns:
        Dictionary mapping tool/package name to version string.
    """
    return {
        "python": sys.version,
        "python_version_short": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "scipy": scipy.__version__,
    }


def load_dataset(path: Path) -> pd.DataFrame:
    """Load an NSL-KDD dataset file (no header, 43 columns).

    The function reads the raw CSV with the canonical NSL-KDD column names,
    does NOT modify any values, and preserves all original dtypes.

    Args:
        path: Path to the .txt dataset file.

    Returns:
        DataFrame with columns from NSL_KDD_FEATURE_NAMES (length 43).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file has unexpected column count.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Place KDDTrain+.txt and KDDTest+.txt in data/raw/ before running."
        )
    log.info("Loading %s …", path.name)
    df = pd.read_csv(path, header=None, names=NSL_KDD_FEATURE_NAMES)
    if df.shape[1] != 43:
        raise ValueError(
            f"Expected 43 columns, got {df.shape[1]} in {path.name}"
        )
    log.info("  Loaded %d rows × %d columns", *df.shape)
    return df


def detect_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """Classify each column's semantic type.

    Args:
        df: The loaded DataFrame.

    Returns:
        Dictionary mapping column name → semantic type label.
    """
    dtype_map: dict[str, str] = {}
    for col in df.columns:
        if col == "label":
            dtype_map[col] = "target_label"
        elif col == "difficulty":
            dtype_map[col] = "metadata_integer"
        elif col in CATEGORICAL_FEATURES:
            dtype_map[col] = "categorical_nominal"
        elif col in BINARY_FEATURES:
            dtype_map[col] = "binary_integer"
        else:
            dtype_map[col] = "numerical_continuous_or_count"
    return dtype_map


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyse_data_quality(df: pd.DataFrame, split: str) -> dict:
    """Compute data-quality metrics for one split.

    Checks:
    - Missing values per column
    - Duplicate rows (full-row equality)
    - Constant features (single unique value)
    - Near-constant features (≤ 2 unique values in numerical columns)
    - Negative values in features that should be non-negative
    - Data-type consistency

    Args:
        df: Loaded DataFrame for one split (train or test).
        split: Label string for logging ("train" or "test").

    Returns:
        Dictionary of quality-check results.
    """
    log.info("[%s] Running data-quality checks …", split)

    feature_cols = [c for c in df.columns if c not in ("label", "difficulty")]

    missing_per_col: dict[str, int] = df[feature_cols].isnull().sum().to_dict()
    total_missing: int = int(sum(missing_per_col.values()))

    dup_mask = df[feature_cols + ["label"]].duplicated(keep=False)
    duplicate_rows: int = int(dup_mask.sum())
    unique_duplicate_groups: int = int(df[feature_cols + ["label"]].duplicated(keep="first").sum())

    constant_features: list[str] = [
        c for c in feature_cols if df[c].nunique() == 1
    ]
    near_constant_numerical: list[str] = [
        c for c in NUMERICAL_FEATURES
        if c in df.columns and df[c].nunique() <= 2
    ]

    # num_outbound_cmds is known to be constant (always 0) in NSL-KDD
    negative_check: dict[str, int] = {}
    for col in NUMERICAL_FEATURES:
        if col in df.columns and col not in BINARY_FEATURES:
            n_neg = int((df[col] < 0).sum())
            if n_neg > 0:
                negative_check[col] = n_neg

    report = {
        "split": split,
        "total_rows": len(df),
        "total_feature_columns": len(feature_cols),
        "total_missing_values": total_missing,
        "missing_by_column": {k: v for k, v in missing_per_col.items() if v > 0},
        "duplicate_rows_total": duplicate_rows,
        "unique_duplicate_groups": unique_duplicate_groups,
        "constant_features": constant_features,
        "near_constant_numerical_features": near_constant_numerical,
        "negative_value_anomalies": negative_check,
    }

    log.info(
        "[%s] Missing: %d | Duplicates: %d | Constant features: %s",
        split, total_missing, duplicate_rows, constant_features or "none",
    )
    return report


def compute_class_distribution(df: pd.DataFrame, split: str) -> dict:
    """Compute per-label and per-attack-category class distributions.

    Args:
        df: Loaded DataFrame.
        split: "train" or "test".

    Returns:
        Dictionary with raw label counts, percentages, and category-level counts.
    """
    log.info("[%s] Computing class distributions …", split)
    raw_counts = df["label"].value_counts().to_dict()
    label_counts = {str(k): int(v) for k, v in raw_counts.items()}
    total = len(df)

    label_pct = {k: round(v / total * 100, 4) for k, v in label_counts.items()}

    # Map to 5-category taxonomy
    df_copy = df.copy()
    df_copy["attack_category"] = df_copy["label"].apply(
        lambda x: ATTACK_CATEGORY_MAP.get(str(x).strip(), "Other")
    )
    cat_counts = {str(k): int(v) for k, v in df_copy["attack_category"].value_counts().items()}
    cat_pct = {k: round(v / total * 100, 4) for k, v in cat_counts.items()}

    normal_count = label_counts.get("normal", 0)
    attack_count = total - normal_count

    result = {
        "split": split,
        "total_samples": total,
        "num_unique_labels": len(label_counts),
        "normal_count": normal_count,
        "normal_pct": round(normal_count / total * 100, 4),
        "attack_count": attack_count,
        "attack_pct": round(attack_count / total * 100, 4),
        "label_counts": label_counts,
        "label_pct": label_pct,
        "category_counts": cat_counts,
        "category_pct": cat_pct,
    }
    log.info(
        "[%s] %d total | %d normal | %d attack | %d unique labels",
        split, total, normal_count, attack_count, len(label_counts),
    )
    return result


def analyse_train_test_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """Analyse the relationship between train and test splits.

    Checks:
    - Label overlap (which attack types appear in both / only one split)
    - Exact row duplication across splits
    - Category-level label distribution shift
    - Difficulty score distributions

    Args:
        train_df: Training DataFrame.
        test_df:  Test DataFrame.

    Returns:
        Dictionary of overlap and distribution-shift findings.
    """
    log.info("Analysing train/test overlap …")

    feature_cols = [
        c for c in train_df.columns if c not in ("label", "difficulty")
    ]

    # Label-set overlap
    train_labels: set[str] = set(str(x) for x in train_df["label"].unique())
    test_labels:  set[str] = set(str(x) for x in test_df["label"].unique())
    shared_labels = sorted(train_labels & test_labels)
    train_only_labels = sorted(train_labels - test_labels)
    test_only_labels  = sorted(test_labels - train_labels)

    # Exact row overlap (feature columns + label, NOT difficulty)
    log.info("Computing exact row overlap via vectorized DataFrame matching …")
    match_cols = feature_cols + ["label"]
    train_unique = train_df[match_cols].drop_duplicates()
    test_rows = test_df[match_cols]
    merged = pd.merge(test_rows, train_unique, on=match_cols, how="inner")

    total_overlapping_test_samples = int(len(merged))
    unique_overlapping_patterns = int(len(merged.drop_duplicates()))
    total_overlapping_test_pct = round(
        (total_overlapping_test_samples / len(test_df)) * 100, 4
    ) if len(test_df) > 0 else 0.0

    # Category distribution comparison
    def cat_dist(df: pd.DataFrame) -> dict[str, float]:
        cats = df["label"].apply(
            lambda x: ATTACK_CATEGORY_MAP.get(str(x).strip(), "Other")
        )
        pct = cats.value_counts(normalize=True).mul(100).round(4)
        return pct.to_dict()

    train_cat_pct = cat_dist(train_df)
    test_cat_pct  = cat_dist(test_df)

    # Difficulty metadata comparison
    train_diff_stats = train_df["difficulty"].describe().round(4).to_dict()
    test_diff_stats  = test_df["difficulty"].describe().round(4).to_dict()

    result = {
        "train_unique_labels": len(train_labels),
        "test_unique_labels": len(test_labels),
        "shared_labels_count": len(shared_labels),
        "shared_labels": shared_labels,
        "train_only_labels": train_only_labels,
        "test_only_labels": test_only_labels,
        "unique_overlapping_patterns": unique_overlapping_patterns,
        "total_overlapping_test_samples": total_overlapping_test_samples,
        "total_overlapping_test_pct": total_overlapping_test_pct,
        # Backward-compatibility alias
        "exact_row_overlap_count": unique_overlapping_patterns,
        "exact_row_overlap_pct_of_test": total_overlapping_test_pct,
        "train_category_pct": train_cat_pct,
        "test_category_pct": test_cat_pct,
        "train_difficulty_stats": train_diff_stats,
        "test_difficulty_stats": test_diff_stats,
    }
    log.info(
        "Shared labels: %d | Train-only: %d | Test-only: %d | Overlapping patterns: %d | Overlapping test samples: %d (%.2f%%)",
        len(shared_labels), len(train_only_labels),
        len(test_only_labels), unique_overlapping_patterns,
        total_overlapping_test_samples, total_overlapping_test_pct,
    )
    return result


def identify_leakage_risks(
    overlap: dict,
    train_quality: dict,
    test_quality: dict,
) -> list[dict]:
    """Enumerate potential data-leakage risks based on audit findings.

    Args:
        overlap: Output of analyse_train_test_overlap.
        train_quality: Output of analyse_data_quality for train.
        test_quality: Output of analyse_data_quality for test.

    Returns:
        List of risk dicts, each with keys:
          risk_id, severity, description, evidence, mitigation.
    """
    risks: list[dict] = []

    # Risk 1 — Difficulty column leakage
    risks.append({
        "risk_id": "LEAK-001",
        "severity": "HIGH",
        "description": (
            "The 'difficulty' column encodes the number of classifiers (out of 21) "
            "that correctly classified each record in the original KDD Cup 99 "
            "challenge. It is computed *using the label* and thus directly "
            "encodes class difficulty. Including it as a training feature would "
            "constitute label leakage."
        ),
        "evidence": (
            "difficulty column present in both train and test; "
            "values are derived from classifier agreement on the true label."
        ),
        "mitigation": (
            "Treat difficulty as metadata only. Never include it in feature "
            "matrices used for model training or evaluation."
        ),
    })

    # Risk 2 — Novel test labels
    if overlap["test_only_labels"]:
        risks.append({
            "risk_id": "LEAK-002",
            "severity": "MEDIUM",
            "description": (
                "The test set contains attack labels not present in the training set. "
                "Models trained on train cannot have learned these classes, which "
                "may cause inflated false-negative rates and must be reported."
            ),
            "evidence": f"Test-only labels: {overlap['test_only_labels']}",
            "mitigation": (
                "Report per-label test performance separately. Consider a "
                "'known vs unknown attack' evaluation protocol."
            ),
        })

    # Risk 3 — Exact row overlap across splits
    overlap_samples = overlap.get("total_overlapping_test_samples", overlap.get("exact_row_overlap_count", 0))
    overlap_patterns = overlap.get("unique_overlapping_patterns", overlap.get("exact_row_overlap_count", 0))
    overlap_pct = overlap.get("total_overlapping_test_pct", overlap.get("exact_row_overlap_pct_of_test", 0.0))
    if overlap_samples > 0:
        risks.append({
            "risk_id": "LEAK-003",
            "severity": "LOW",
            "description": (
                "Some feature-label rows appear in both the training and test sets. "
                "This can inflate test-set metrics if not handled carefully."
            ),
            "evidence": (
                f"{overlap_samples:,} test samples ({overlap_pct:.2f}% of test set) "
                f"sharing {overlap_patterns:,} unique feature-label patterns "
                "are exact duplicates of training records."
            ),
            "mitigation": (
                "Report results both including and excluding overlapping test rows. "
                "# RESEARCH DECISION NEEDED: decide whether to remove overlap rows."
            ),
        })

    # Risk 4 — Constant features (num_outbound_cmds)
    constant_union = list(
        set(train_quality["constant_features"]) |
        set(test_quality["constant_features"])
    )
    if constant_union:
        risks.append({
            "risk_id": "LEAK-004",
            "severity": "LOW",
            "description": (
                "One or more features are constant (single unique value) across "
                "all samples. Constant features carry zero information and can "
                "cause issues with some normalisation schemes."
            ),
            "evidence": f"Constant features: {constant_union}",
            "mitigation": (
                "Remove constant features during preprocessing. "
                "Document which features were removed and why."
            ),
        })

    # Risk 5 — Class distribution shift
    train_cats = overlap["train_category_pct"]
    test_cats  = overlap["test_category_pct"]
    large_shifts = []
    all_cats = set(train_cats) | set(test_cats)
    for cat in all_cats:
        t = train_cats.get(cat, 0.0)
        e = test_cats.get(cat, 0.0)
        if abs(t - e) > 5.0:  # >5 percentage-point shift
            large_shifts.append(
                f"{cat}: train={t:.1f}% → test={e:.1f}% (Δ={e-t:+.1f}pp)"
            )
    if large_shifts:
        risks.append({
            "risk_id": "LEAK-005",
            "severity": "MEDIUM",
            "description": (
                "Substantial class distribution shift between training and test sets "
                "(>5 percentage points). Accuracy evaluated on the test set may not "
                "reflect deployment performance, and may mask per-class degradation."
            ),
            "evidence": " | ".join(large_shifts),
            "mitigation": (
                "Report per-category precision, recall, and F1 in addition to "
                "overall accuracy. Use macro-averaged metrics for fair comparison."
            ),
        })

    return risks


def build_feature_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature-schema CSV table from the loaded DataFrame.

    Args:
        df: Loaded DataFrame (train or test — schema is identical).

    Returns:
        DataFrame with one row per feature and columns:
        feature_name, index, semantic_type, pandas_dtype,
        unique_values, sample_values.
    """
    dtype_map = detect_dtypes(df)
    rows = []
    for i, col in enumerate(df.columns):
        n_unique = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:5].tolist()
        rows.append({
            "feature_index": i,
            "feature_name": col,
            "semantic_type": dtype_map[col],
            "pandas_dtype": str(df[col].dtype),
            "num_unique_values": n_unique,
            "sample_values": str(sample_vals),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    """Apply a consistent, publication-friendly Matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def plot_class_distribution(
    dist: dict,
    split: str,
    out_path: Path,
) -> None:
    """Create a horizontal bar chart of per-category class distribution.

    Args:
        dist: Output of compute_class_distribution.
        split: "train" or "test" (used in title).
        out_path: File path to save the figure.
    """
    _apply_style()
    cat_counts = dist["category_counts"]
    ordered_cats = [c for c in ATTACK_CATEGORIES_ORDER if c in cat_counts]
    if "Other" in cat_counts:
        ordered_cats.append("Other")
    counts = [cat_counts[c] for c in ordered_cats]
    colors  = [PALETTE.get(c, PALETTE["Other"]) for c in ordered_cats]
    total   = dist["total_samples"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(ordered_cats, counts, color=colors, edgecolor="white", height=0.6)

    for bar, cnt in zip(bars, counts):
        pct = cnt / total * 100
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:,}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Number of samples")
    ax.set_title(
        f"NSL-KDD {split.title()} Set — Attack Category Distribution\n"
        f"(Total samples: {total:,})",
    )
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, max(counts) * 1.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", out_path.name)


def plot_label_distribution(
    dist: dict,
    split: str,
    out_path: Path,
    top_n: int = 20,
) -> None:
    """Create a bar chart of the top-N individual attack labels.

    Args:
        dist: Output of compute_class_distribution.
        split: "train" or "test".
        out_path: Save path.
        top_n: Number of most frequent labels to show.
    """
    _apply_style()
    label_counts = dist["label_counts"]
    sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:top_n]
    labels, counts = zip(*sorted_labels)
    total = dist["total_samples"]

    cat_colors = []
    for lbl in labels:
        cat = ATTACK_CATEGORY_MAP.get(lbl, "Other")
        cat_colors.append(PALETTE.get(cat, PALETTE["Other"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), counts, color=cat_colors, edgecolor="white")

    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.002,
            f"{cnt:,}",
            ha="center", va="bottom", fontsize=7, rotation=45,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of samples")
    ax.set_title(
        f"NSL-KDD {split.title()} Set — Top {len(labels)} Label Distribution"
    )
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE[c], label=c)
        for c in ATTACK_CATEGORIES_ORDER
        if any(ATTACK_CATEGORY_MAP.get(l, "Other") == c for l in labels)
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", out_path.name)


def plot_train_test_label_comparison(
    train_dist: dict,
    test_dist: dict,
    out_path: Path,
) -> None:
    """Side-by-side grouped bar chart: train vs test category percentages.

    Args:
        train_dist: compute_class_distribution output for train.
        test_dist:  compute_class_distribution output for test.
        out_path: Save path.
    """
    _apply_style()
    all_cats = ATTACK_CATEGORIES_ORDER.copy()
    train_pct = [train_dist["category_pct"].get(c, 0.0) for c in all_cats]
    test_pct  = [test_dist["category_pct"].get(c, 0.0) for c in all_cats]

    x = np.arange(len(all_cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, train_pct, width, label="Train",
                   color=[PALETTE.get(c, PALETTE["Other"]) for c in all_cats],
                   alpha=0.9, edgecolor="white")
    bars2 = ax.bar(x + width / 2, test_pct, width, label="Test",
                   color=[PALETTE.get(c, PALETTE["Other"]) for c in all_cats],
                   alpha=0.5, edgecolor="black", linewidth=0.8)

    for bar, pct in zip(list(bars1) + list(bars2), train_pct + test_pct):
        if pct > 0.5:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(all_cats)
    ax.set_ylabel("Percentage of split (%)")
    ax.set_title("NSL-KDD — Attack Category Distribution: Train vs Test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", out_path.name)


def plot_difficulty_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Histogram of difficulty scores for train and test splits.

    The difficulty column is treated as metadata only.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        out_path: Save path.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, df, split, colour in zip(
        axes,
        [train_df, test_df],
        ["Train", "Test"],
        ["#2196F3", "#FF9800"],
    ):
        vals = df["difficulty"].dropna()
        bins = sorted(vals.unique())
        if len(bins) > 22:
            bins = 22
        ax.hist(vals, bins=bins, color=colour, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Difficulty score")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{split} Set — Difficulty Score Distribution\n"
            f"(metadata only; not used as a feature)"
        )
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", out_path.name)


def plot_feature_types(out_path: Path) -> None:
    """Pie chart showing breakdown of feature semantic types.

    Args:
        out_path: Save path.
    """
    _apply_style()
    type_counts = {
        "Categorical nominal\n(3 features)": len(CATEGORICAL_FEATURES),
        "Binary integer\n(6 features)": len(BINARY_FEATURES),
        "Numerical\n(32 features)": len([
            f for f in NUMERICAL_FEATURES if f not in BINARY_FEATURES
        ]),
        "Target label\n(1 column)": 1,
        "Metadata\n(difficulty, 1 col)": 1,
    }
    labels  = list(type_counts.keys())
    sizes   = list(type_counts.values())
    colors  = ["#FF9800", "#9C27B0", "#2196F3", "#F44336", "#607D8B"]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax.set_title("NSL-KDD Feature Schema — Column Type Breakdown\n(43 total columns)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", out_path.name)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_audit_report(
    train_dist: dict,
    test_dist: dict,
    train_quality: dict,
    test_quality: dict,
    overlap: dict,
    leakage_risks: list[dict],
    env_info: dict,
    checksums: dict[str, str],
    seed: int,
    timestamp: str,
    out_path: Path,
) -> None:
    """Write the full Markdown audit report.

    Args:
        All analysis result dicts and metadata.
        out_path: Path to write the .md report.
    """
    log.info("Generating audit report …")

    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def para(text: str) -> None:
        lines.append(f"\n{text}\n")

    def table_row(*cells: str) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    def table_sep(*aligns: str) -> str:
        mp = {"l": ":---", "r": "---:", "c": ":---:"}
        return "| " + " | ".join(mp.get(a, "---") for a in aligns) + " |"

    # ---- Header
    lines.append("# EXP-001 — NSL-KDD Dataset Acquisition and Leakage Audit")
    lines.append(f"\n*Generated: {timestamp} UTC*  \n*Random seed: {seed}*\n")
    lines.append("> **STATUS:** COMPLETE — All statistics computed from actual dataset files.\n")

    # ---- Research Question
    h(2, "Research Question")
    para(
        "What are the statistical properties of the NSL-KDD training and test sets, "
        "and are there any data-leakage risks (e.g., train/test overlap, "
        "difficulty-score contamination, or feature-level leakage) that would "
        "invalidate downstream model evaluations?"
    )

    # ---- Hypothesis
    h(2, "Hypothesis")
    para(
        "The NSL-KDD dataset contains measurable class imbalance and a non-trivial "
        "overlap between training and test attack categories. The difficulty field "
        "encodes per-sample classifier agreement scores that, if included as a "
        "feature, would constitute label leakage. Novel attack types in the test set "
        "will not appear in the training label set."
    )

    # ---- Dataset
    h(2, "Dataset")
    lines.append(table_row("Field", "Value"))
    lines.append(table_sep("l", "l"))
    lines.append(table_row("Name", "NSL-KDD"))
    lines.append(table_row(
        "Source",
        "University of New Brunswick — Canadian Institute for Cybersecurity"
    ))
    lines.append(table_row("URL", "https://www.unb.ca/cic/datasets/nsl.html"))
    lines.append(table_row(
        "Citation",
        "Tavallaee et al. (2009), IEEE CISDA"
    ))
    lines.append("")

    h(3, "Dataset Files")
    lines.append(table_row("File", "Rows", "SHA-256"))
    lines.append(table_sep("l", "r", "l"))
    lines.append(table_row(
        "KDDTrain+.txt",
        f"{train_dist['total_samples']:,}",
        checksums.get("train", "N/A"),
    ))
    lines.append(table_row(
        "KDDTest+.txt",
        f"{test_dist['total_samples']:,}",
        checksums.get("test", "N/A"),
    ))
    lines.append("")

    # ---- Dataset Schema
    h(2, "Dataset Schema")
    para(
        "The NSL-KDD dataset has **43 columns**: 41 network-traffic features, "
        "1 target label column, and 1 metadata column (difficulty). "
        "There is no header row in the raw files."
    )
    lines.append(table_row("Column Group", "Count", "Feature Names"))
    lines.append(table_sep("l", "r", "l"))
    lines.append(table_row(
        "Categorical nominal", str(len(CATEGORICAL_FEATURES)),
        ", ".join(CATEGORICAL_FEATURES),
    ))
    lines.append(table_row(
        "Binary integer", str(len(BINARY_FEATURES)),
        ", ".join(BINARY_FEATURES),
    ))
    num_only = [f for f in NUMERICAL_FEATURES if f not in BINARY_FEATURES]
    lines.append(table_row(
        "Numerical (count/ratio)", str(len(num_only)),
        "duration, src_bytes, dst_bytes, … (see feature_schema.csv)",
    ))
    lines.append(table_row("Target label", "1", "label"))
    lines.append(table_row("Metadata (excluded from features)", "1", "difficulty"))
    lines.append("")

    # ---- Data Quality Findings
    h(2, "Data Quality Findings")

    for quality, split_name in [(train_quality, "Training"), (test_quality, "Test")]:
        h(3, f"{split_name} Set")
        lines.append(table_row("Metric", "Value"))
        lines.append(table_sep("l", "r"))
        lines.append(table_row("Total rows", f"{quality['total_rows']:,}"))
        lines.append(table_row("Total feature columns", str(quality["total_feature_columns"])))
        lines.append(table_row("Missing values (total)", str(quality["total_missing_values"])))
        lines.append(table_row("Duplicate rows", f"{quality['duplicate_rows_total']:,}"))
        lines.append(table_row(
            "Constant features",
            ", ".join(quality["constant_features"]) or "None",
        ))
        lines.append(table_row(
            "Near-constant numerical features (≤2 unique values)",
            ", ".join(quality["near_constant_numerical_features"]) or "None",
        ))
        lines.append(table_row(
            "Negative-value anomalies",
            str(quality["negative_value_anomalies"]) if quality["negative_value_anomalies"] else "None",
        ))
        lines.append("")

    # ---- Class Distribution
    h(2, "Class Distribution")

    for dist, split_name in [(train_dist, "Training"), (test_dist, "Test")]:
        h(3, f"{split_name} Set")
        lines.append(table_row("Attack Category", "Count", "Percentage"))
        lines.append(table_sep("l", "r", "r"))
        for cat in ATTACK_CATEGORIES_ORDER:
            cnt = dist["category_counts"].get(cat, 0)
            pct = dist["category_pct"].get(cat, 0.0)
            lines.append(table_row(cat, f"{cnt:,}", f"{pct:.2f}%"))
        if "Other" in dist["category_counts"]:
            cnt = dist["category_counts"]["Other"]
            pct = dist["category_pct"].get("Other", 0.0)
            lines.append(table_row("Other (unmapped)", f"{cnt:,}", f"{pct:.2f}%"))
        lines.append(table_row(
            "**TOTAL**",
            f"**{dist['total_samples']:,}**",
            "**100.00%**",
        ))
        lines.append("")

        # Top fine-grained labels
        top10 = sorted(
            dist["label_counts"].items(), key=lambda x: -x[1]
        )[:10]
        lines.append(f"**Top 10 individual labels ({split_name}):**\n")
        lines.append(table_row("Label", "Category", "Count", "Percentage"))
        lines.append(table_sep("l", "l", "r", "r"))
        for lbl, cnt in top10:
            cat = ATTACK_CATEGORY_MAP.get(lbl, "Other")
            pct = cnt / dist["total_samples"] * 100
            lines.append(table_row(lbl, cat, f"{cnt:,}", f"{pct:.2f}%"))
        lines.append("")

    # ---- Train/Test Analysis
    h(2, "Train / Test Analysis")
    lines.append(table_row("Metric", "Value"))
    lines.append(table_sep("l", "l"))
    lines.append(table_row("Train unique labels", str(overlap["train_unique_labels"])))
    lines.append(table_row("Test unique labels", str(overlap["test_unique_labels"])))
    lines.append(table_row("Shared labels", str(overlap["shared_labels_count"])))
    lines.append(table_row(
        "Train-only labels",
        ", ".join(overlap["train_only_labels"]) or "None",
    ))
    lines.append(table_row(
        "Test-only labels",
        ", ".join(overlap["test_only_labels"]) or "None",
    ))
    lines.append(table_row(
        "Unique overlapping patterns",
        f"{overlap['unique_overlapping_patterns']:,}",
    ))
    lines.append(table_row(
        "Total overlapping test samples",
        f"{overlap['total_overlapping_test_samples']:,} "
        f"({overlap['total_overlapping_test_pct']:.2f}% of test)",
    ))
    lines.append("")

    h(3, "Category-Level Distribution Shift")
    lines.append(table_row("Category", "Train %", "Test %", "Δ (pp)"))
    lines.append(table_sep("l", "r", "r", "r"))
    for cat in ATTACK_CATEGORIES_ORDER:
        tp = overlap["train_category_pct"].get(cat, 0.0)
        ep = overlap["test_category_pct"].get(cat, 0.0)
        delta = ep - tp
        lines.append(table_row(cat, f"{tp:.2f}%", f"{ep:.2f}%", f"{delta:+.2f}"))
    lines.append("")

    # ---- Leakage Risks
    h(2, "Potential Leakage Risks")
    for risk in leakage_risks:
        h(3, f"{risk['risk_id']} — {risk['severity']} Severity")
        lines.append(f"**Description:** {risk['description']}\n")
        lines.append(f"**Evidence:** {risk['evidence']}\n")
        lines.append(f"**Mitigation:** {risk['mitigation']}\n")

    # ---- Limitations
    h(2, "Limitations")
    para(
        "1. **Synthetic origin.** NSL-KDD is derived from KDD Cup 99, which was "
        "generated in a simulated environment. Results on this dataset may not "
        "generalise to real-world network traffic. "
        "# RESEARCH DECISION NEEDED: confirm dataset suitability for research goals.\n\n"
        "2. **Outdated traffic patterns.** The underlying KDD Cup 99 data was "
        "captured in 1998. Modern attack signatures differ substantially.\n\n"
        "3. **No preprocessing applied.** This audit examines raw, unprocessed data. "
        "Data quality conclusions refer only to the raw state.\n\n"
        "4. **Difficulty column.** The difficulty field's exact computation is not "
        "fully documented; it is excluded from all downstream feature sets.\n\n"
        "5. **Row overlap evaluation.** Cross-split duplication is evaluated via exact "
        "vector matching across all 41 features plus the label (excluding difficulty)."
    )

    # ---- Conclusion
    h(2, "Conclusion")
    n_risks = len(leakage_risks)
    high_risks = [r for r in leakage_risks if r["severity"] == "HIGH"]
    para(
        f"This audit identified **{n_risks} leakage risks** "
        f"({'including ' + str(len(high_risks)) + ' HIGH severity' if high_risks else 'no HIGH severity risks'}). "
        f"The primary concern is the **difficulty column** (LEAK-001), which encodes "
        f"label-derived classifier agreement and must be excluded from all feature matrices. "
        f"The test set contains labels absent from training, requiring per-label evaluation "
        f"reporting. Exact row overlap between splits is present and must be documented "
        f"when reporting test-set metrics. "
        f"No preprocessing, normalisation, encoding, or model training was performed in "
        f"this experiment. All findings are based on the raw dataset files."
    )

    # ---- Environment
    h(2, "Reproducibility")
    lines.append(table_row("Item", "Value"))
    lines.append(table_sep("l", "l"))
    lines.append(table_row("Experiment ID", "EXP-001"))
    lines.append(table_row("Random seed", str(seed)))
    lines.append(table_row("Execution timestamp", timestamp))
    lines.append(table_row("Python version", env_info["python_version_short"]))
    lines.append(table_row("Platform", env_info["platform"]))
    lines.append(table_row("pandas", env_info["pandas"]))
    lines.append(table_row("numpy", env_info["numpy"]))
    lines.append(table_row("matplotlib", env_info["matplotlib"]))
    lines.append(table_row("scipy", env_info["scipy"]))
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Audit report written to: %s", out_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed Namespace with train, test, and seed fields.
    """
    parser = argparse.ArgumentParser(
        description="EXP-001: NSL-KDD Dataset Acquisition and Leakage Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=DEFAULT_TRAIN,
        help="Path to KDDTrain+.txt (default: data/raw/KDDTrain+.txt)",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=DEFAULT_TEST,
        help="Path to KDDTest+.txt (default: data/raw/KDDTest+.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for any sampling operations (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full EXP-001 audit pipeline."""
    args = parse_args()
    seed: int = args.seed

    # Set random seeds (no stochastic operations in this audit, but set for reproducibility)
    np.random.seed(seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log.info("=" * 60)
    log.info("EXP-001: NSL-KDD Dataset Acquisition and Leakage Audit")
    log.info("Timestamp : %s", timestamp)
    log.info("Seed      : %d", seed)
    log.info("Train     : %s", args.train)
    log.info("Test      : %s", args.test)
    log.info("=" * 60)

    # Create output directories
    for d in (FIG_DIR, METRICS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Environment
    env_info = get_environment_info()
    log.info("Python %s | pandas %s | numpy %s",
             env_info["python_version_short"], env_info["pandas"], env_info["numpy"])

    # ---- Checksums
    log.info("Computing file checksums …")
    checksums = {
        "train": sha256_file(args.train),
        "test":  sha256_file(args.test),
    }
    log.info("Train SHA-256: %s", checksums["train"])
    log.info("Test  SHA-256: %s", checksums["test"])

    # ---- Load datasets
    train_df = load_dataset(args.train)
    test_df  = load_dataset(args.test)

    # ---- Feature schema table
    log.info("Building feature schema table …")
    schema_df = build_feature_schema_table(train_df)
    schema_path = TABLES_DIR / "feature_schema.csv"
    schema_df.to_csv(schema_path, index=False)
    log.info("Saved: %s", schema_path.name)

    # ---- Descriptive statistics (numerical only — raw, no transformation)
    log.info("Computing descriptive statistics …")
    num_cols = [c for c in NUMERICAL_FEATURES if c in train_df.columns]
    train_desc = train_df[num_cols].describe().T
    test_desc  = test_df[num_cols].describe().T
    train_desc.to_csv(TABLES_DIR / "descriptive_stats_train.csv")
    test_desc.to_csv(TABLES_DIR / "descriptive_stats_test.csv")
    log.info("Saved descriptive stats CSVs.")

    # ---- Data quality
    train_quality = analyse_data_quality(train_df, "train")
    test_quality  = analyse_data_quality(test_df,  "test")

    # ---- Class distributions
    train_dist = compute_class_distribution(train_df, "train")
    test_dist  = compute_class_distribution(test_df,  "test")

    # ---- Train/test overlap
    overlap = analyse_train_test_overlap(train_df, test_df)

    # ---- Leakage risks
    leakage_risks = identify_leakage_risks(overlap, train_quality, test_quality)
    log.info("Identified %d leakage risks.", len(leakage_risks))

    # ---- Figures
    log.info("Generating figures …")
    plot_class_distribution(train_dist, "train",
                            FIG_DIR / "train_class_distribution.png")
    plot_class_distribution(test_dist,  "test",
                            FIG_DIR / "test_class_distribution.png")
    plot_label_distribution(train_dist, "train",
                            FIG_DIR / "train_label_distribution.png")
    plot_label_distribution(test_dist,  "test",
                            FIG_DIR / "test_label_distribution.png")
    plot_train_test_label_comparison(train_dist, test_dist,
                                     FIG_DIR / "train_test_label_comparison.png")
    plot_difficulty_distribution(train_df, test_df,
                                 FIG_DIR / "difficulty_distribution.png")
    plot_feature_types(FIG_DIR / "feature_types_overview.png")

    # ---- Save machine-readable JSON outputs
    log.info("Saving JSON metrics …")

    summary = {
        "experiment_id": "EXP-001",
        "timestamp": timestamp,
        "seed": seed,
        "environment": env_info,
        "checksums": checksums,
        "train_file": str(args.train),
        "test_file":  str(args.test),
        "train_shape": list(train_df.shape),
        "test_shape":  list(test_df.shape),
        "num_features": 41,
        "label_column": "label",
        "difficulty_column": "difficulty (metadata only)",
    }
    (METRICS_DIR / "summary_statistics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (METRICS_DIR / "class_distribution_train.json").write_text(
        json.dumps(train_dist, indent=2), encoding="utf-8"
    )
    (METRICS_DIR / "class_distribution_test.json").write_text(
        json.dumps(test_dist, indent=2), encoding="utf-8"
    )
    (METRICS_DIR / "data_quality_report.json").write_text(
        json.dumps({"train": train_quality, "test": test_quality}, indent=2),
        encoding="utf-8",
    )
    (METRICS_DIR / "train_test_overlap.json").write_text(
        json.dumps(overlap, indent=2), encoding="utf-8"
    )
    (METRICS_DIR / "leakage_risks.json").write_text(
        json.dumps(leakage_risks, indent=2), encoding="utf-8"
    )
    log.info("All JSON metrics saved.")

    # ---- Audit report
    generate_audit_report(
        train_dist=train_dist,
        test_dist=test_dist,
        train_quality=train_quality,
        test_quality=test_quality,
        overlap=overlap,
        leakage_risks=leakage_risks,
        env_info=env_info,
        checksums=checksums,
        seed=seed,
        timestamp=timestamp,
        out_path=RESULTS_DIR / "audit_report.md",
    )

    log.info("=" * 60)
    log.info("EXP-001 complete. Results in: %s", RESULTS_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
