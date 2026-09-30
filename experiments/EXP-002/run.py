"""
EXP-002: Supervised Baseline Binary Intrusion Detection and Generalization Audit
================================================================================
SentinelNet Research Project

Purpose
-------
Train and evaluate Logistic Regression, Decision Tree, and Random Forest on
the NSL-KDD dataset for binary (Normal vs. Attack) classification.

Locked methodological decisions
--------------------------------
- Primary test set  : full KDDTest+ (official benchmark)
- Overlap cohorts   : 610 duplicate samples + 21,934 non-overlapping samples
- Novel cohorts     : known vs. novel (test-only) attack labels
- Preprocessing     : fitted strictly on the training partition
- difficulty        : never used as a feature (LEAK-001)
- num_outbound_cmds : dropped (LEAK-004, zero-variance)
- No SMOTE / resampling
- KDDTest+ never used for model selection or hyperparameter tuning

Usage
-----
    python experiments/EXP-002/run.py [--train PATH] [--test PATH] [--seed INT]

Scientific Integrity
--------------------
All reported metrics are computed from actual model outputs on real data.
No statistics are fabricated or estimated. STATUS is set to PENDING until
this script completes successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] EXP-002 | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("EXP-002")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"

DEFAULT_TRAIN = REPO_ROOT / "data" / "raw" / "KDDTrain+.txt"
DEFAULT_TEST = REPO_ROOT / "data" / "raw" / "KDDTest+.txt"

# ---------------------------------------------------------------------------
# NSL-KDD Feature Schema  (inherited from EXP-001)
# ---------------------------------------------------------------------------
NSL_KDD_FEATURE_NAMES: list[str] = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

# Attack taxonomy for category-level novelty reporting
# Source: MIT Lincoln Laboratory DARPA 1998/1999 / KDD Cup 99 task specification
ATTACK_CATEGORY_MAP: dict[str, str] = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
BINARY_FEATURES = [
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
]
# Columns dropped before modelling
DROP_FEATURES = ["difficulty", "num_outbound_cmds"]

RANDOM_SEED = 42
VALIDATION_SIZE = 0.20  # 80/20 stratified split within KDDTrain+


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hexdigest of a file without loading it all at once."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_environment_info() -> dict[str, str]:
    """Collect Python, library, and platform version strings."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """Load a raw NSL-KDD file with the canonical 43-column schema.

    Args:
        path: Absolute path to a KDDTrain+.txt or KDDTest+.txt file.

    Returns:
        DataFrame with exactly 43 named columns.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If column count is wrong.
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


# ---------------------------------------------------------------------------
# Feature extraction and binary target encoding
# ---------------------------------------------------------------------------

def make_binary_target(labels: pd.Series) -> pd.Series:
    """Encode the label column as binary: normal=0, attack=1.

    Args:
        labels: Raw NSL-KDD label column (string values).

    Returns:
        Integer Series of 0/1 values.
    """
    return (labels.str.strip() != "normal").astype(int)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of feature columns after dropping metadata.

    Args:
        df: Full 43-column NSL-KDD DataFrame.

    Returns:
        List of column names suitable for modelling.
    """
    exclude = {"label", "difficulty"} | set(DROP_FEATURES)
    return [c for c in df.columns if c not in exclude]


def get_numerical_features(feature_cols: list[str]) -> list[str]:
    """Return the numerical (non-categorical, non-binary) feature names.

    Args:
        feature_cols: The active modelling feature column names.

    Returns:
        List of numerical feature names.
    """
    exclude = set(CATEGORICAL_FEATURES) | set(BINARY_FEATURES)
    return [f for f in feature_cols if f not in exclude]


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def build_pipeline(model: Any, feature_cols: list[str]) -> Pipeline:
    """Build a leakage-safe sklearn Pipeline for a given estimator.

    All transformers are fitted exclusively on the training partition via the
    Pipeline API — no test data is ever passed to ``fit``.

    Args:
        model: An instantiated, unfitted sklearn estimator.
        feature_cols: Ordered list of input feature column names.

    Returns:
        An unfitted sklearn Pipeline containing preprocessor + estimator.
    """
    numerical_feats = get_numerical_features(feature_cols)
    cat_indices = [feature_cols.index(c) for c in CATEGORICAL_FEATURES if c in feature_cols]
    bin_indices = [feature_cols.index(c) for c in BINARY_FEATURES if c in feature_cols]
    num_indices = [feature_cols.index(c) for c in numerical_feats]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_indices,
            ),
            (
                "bin",
                "passthrough",
                bin_indices,
            ),
            (
                "num",
                StandardScaler(),
                num_indices,
            ),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------

def identify_overlap_mask(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """Return a boolean Series marking test rows that exactly duplicate a train row.

    Uses vectorized DataFrame merge on all feature columns + label, excluding
    the difficulty metadata column, consistent with EXP-001 methodology.

    Args:
        train_df: Full training DataFrame (43 columns).
        test_df: Full test DataFrame (43 columns).
        feature_cols: Feature columns to match on (excludes difficulty and label).

    Returns:
        Boolean Series aligned to test_df.index; True = overlapping sample.
    """
    match_cols = feature_cols + ["label"]
    train_unique = train_df[match_cols].drop_duplicates()
    test_rows = test_df[match_cols].reset_index(drop=True)
    # Indicator merge
    merged = pd.merge(
        test_rows.reset_index().rename(columns={"index": "_test_idx"}),
        train_unique,
        on=match_cols,
        how="left",
        indicator=True,
    )
    overlap_indices = set(merged.loc[merged["_merge"] == "both", "_test_idx"].tolist())
    mask = pd.Series(
        [i in overlap_indices for i in range(len(test_df))],
        index=test_df.index,
        dtype=bool,
    )
    return mask


def build_cohorts(
    test_df: pd.DataFrame,
    overlap_mask: pd.Series,
    train_labels: set[str],
) -> dict[str, pd.Index]:
    """Build the evaluation cohort index sets from the test DataFrame.

    Cohorts are derived programmatically from the actual data — counts are
    never hard-coded.

    Args:
        test_df: Full test DataFrame.
        overlap_mask: Boolean Series; True = row duplicates a training row.
        train_labels: Set of label strings present in the training set.

    Returns:
        Dictionary mapping cohort name → pd.Index of rows in test_df.
    """
    attack_mask = test_df["label"].str.strip() != "normal"
    known_mask = test_df["label"].str.strip().isin(train_labels - {"normal"})
    novel_mask = attack_mask & ~known_mask

    return {
        "full_test": test_df.index,
        "non_overlapping": test_df.index[~overlap_mask],
        "overlapping": test_df.index[overlap_mask],
        "known_attacks": test_df.index[known_mask],
        "novel_attacks": test_df.index[novel_mask],
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    cohort_name: str,
) -> dict[str, Any]:
    """Compute the full EXP-002 metric set for a single evaluation cohort.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Model-predicted binary labels.
        y_prob: Predicted probabilities for the positive class (may be None).
        cohort_name: Identifier string for logging.

    Returns:
        Dictionary of computed metric values.
    """
    n = len(y_true)
    if n == 0:
        log.warning("Cohort '%s' is empty — skipping metrics.", cohort_name)
        return {"n_samples": 0, "cohort": cohort_name}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)

    # Attack class (positive=1) metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)   # Detection Rate
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # False Alarm Rate: Normal traffic flagged as Attack
    total_normal = tn + fp
    far = fp / total_normal if total_normal > 0 else 0.0

    roc_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_prob)

    result = {
        "cohort": cohort_name,
        "n_samples": int(n),
        "n_normal": int(tn + fp),
        "n_attack": int(fn + tp),
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall_detection_rate": round(float(recall), 6),
        "false_alarm_rate": round(float(far), 6),
        "f1_score": round(float(f1), 6),
        "macro_f1": round(float(macro_f1), 6),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    if roc_auc is not None:
        result["roc_auc"] = round(float(roc_auc), 6)

    return result


def evaluate_model_on_cohorts(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    cohorts: dict[str, pd.Index],
    feature_cols: list[str],
) -> dict[str, dict]:
    """Evaluate a fitted pipeline across all defined test cohorts.

    Args:
        pipeline: A fitted sklearn Pipeline.
        test_df: Full test DataFrame (43 columns, not yet transformed).
        cohorts: Dictionary of cohort name → row indices in test_df.
        feature_cols: Ordered list of feature column names.

    Returns:
        Dictionary mapping cohort name → metric dict.
    """
    X_all = test_df[feature_cols].values
    y_all = make_binary_target(test_df["label"]).values

    # Get predictions for all rows at once (faster than per-cohort)
    y_pred_all = pipeline.predict(X_all)
    has_proba = hasattr(pipeline.named_steps["classifier"], "predict_proba")
    y_prob_all = (
        pipeline.predict_proba(X_all)[:, 1] if has_proba else None
    )

    results = {}
    for cohort_name, idx in cohorts.items():
        positions = test_df.index.get_indexer(idx)
        y_true_c = y_all[positions]
        y_pred_c = y_pred_all[positions]
        y_prob_c = y_prob_all[positions] if y_prob_all is not None else None
        results[cohort_name] = compute_metrics(
            y_true_c, y_pred_c, y_prob_c, cohort_name
        )
    return results


def evaluate_novel_attacks_by_category(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    novel_labels: set[str],
) -> dict[str, dict]:
    """Compute detection rates for novel attacks grouped by attack category.

    Args:
        pipeline: Fitted pipeline.
        test_df: Full test DataFrame.
        feature_cols: Active feature column names.
        novel_labels: Set of attack labels unique to the test set.

    Returns:
        Dictionary mapping category name → metric dict.
    """
    results: dict[str, dict] = {}
    attack_to_category = {
        label: ATTACK_CATEGORY_MAP.get(label, "Other")
        for label in novel_labels
    }
    novel_categories = sorted(set(attack_to_category.values()))

    for category in novel_categories:
        cat_labels = {
            lbl for lbl, cat in attack_to_category.items() if cat == category
        }
        mask = test_df["label"].str.strip().isin(cat_labels)
        sub = test_df[mask]
        if sub.empty:
            continue
        X_sub = sub[feature_cols].values
        y_true_sub = make_binary_target(sub["label"]).values
        y_pred_sub = pipeline.predict(X_sub)
        has_proba = hasattr(pipeline.named_steps["classifier"], "predict_proba")
        y_prob_sub = (
            pipeline.predict_proba(X_sub)[:, 1] if has_proba else None
        )
        cohort_name = f"novel_{category.lower()}_attacks"
        results[cohort_name] = compute_metrics(
            y_true_sub, y_pred_sub, y_prob_sub, cohort_name
        )
        results[cohort_name]["attack_labels"] = sorted(cat_labels)
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_fig(name: str) -> None:
    """Save the current matplotlib figure and close it."""
    path = FIG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved figure: %s", name)


def plot_model_comparison_bar(
    all_results: dict[str, dict[str, dict]],
) -> None:
    """Grouped bar chart comparing Accuracy, Precision, Recall, F1 on full test.

    Args:
        all_results: {model_name: {cohort_name: metric_dict}}
    """
    metrics = ["accuracy", "precision", "recall_detection_rate", "f1_score"]
    labels = ["Accuracy", "Precision", "Detection Rate", "F1-Score"]
    model_names = list(all_results.keys())
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(model_names):
        vals = [
            all_results[model_name]["full_test"].get(m, 0) * 100
            for m in metrics
        ]
        bars = ax.bar(x + i * width, vals, width, label=model_name)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=7, padding=2)

    ax.set_ylabel("Score (%)")
    ax.set_title("Binary IDS Performance — Full KDDTest+\n"
                 "(Note: benchmark includes novel and duplicate test samples)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_fig("poster_model_comparison_bar.png")


def plot_known_vs_novel(
    all_results: dict[str, dict[str, dict]],
) -> None:
    """Grouped bar chart comparing Detection Rate on known vs. novel attacks.

    Args:
        all_results: {model_name: {cohort_name: metric_dict}}
    """
    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    width = 0.35

    known_dr = [
        all_results[m].get("known_attacks", {}).get("recall_detection_rate", 0) * 100
        for m in model_names
    ]
    novel_dr = [
        all_results[m].get("novel_attacks", {}).get("recall_detection_rate", 0) * 100
        for m in model_names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, known_dr, width, label="Known Attacks", color="steelblue")
    b2 = ax.bar(x + width / 2, novel_dr, width, label="Novel Attacks", color="tomato")
    ax.bar_label(b1, fmt="%.1f%%", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.1f%%", fontsize=9, padding=2)

    ax.set_ylabel("Detection Rate (%)")
    ax.set_title("Detection Rate: Known vs. Novel (Test-Only) Attacks")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 115)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_fig("known_vs_novel_detection_rate.png")


def plot_overlap_inflation(
    all_results: dict[str, dict[str, dict]],
) -> None:
    """Grouped bar chart: accuracy on overlapping vs. non-overlapping samples.

    Args:
        all_results: {model_name: {cohort_name: metric_dict}}
    """
    model_names = list(all_results.keys())
    x = np.arange(len(model_names))
    width = 0.35

    non_ov = [
        all_results[m].get("non_overlapping", {}).get("accuracy", 0) * 100
        for m in model_names
    ]
    ov = [
        all_results[m].get("overlapping", {}).get("accuracy", 0) * 100
        for m in model_names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width / 2, non_ov, width, label="Non-Overlapping (N=21,934)", color="mediumseagreen")
    b2 = ax.bar(x + width / 2, ov, width, label="Overlapping Duplicates (N=610)", color="darkorange")
    ax.bar_label(b1, fmt="%.1f%%", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.1f%%", fontsize=9, padding=2)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy: Non-Overlapping vs. Train-Duplicate Test Samples\n"
                 "(Δ measures memorization inflation)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 115)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_fig("overlap_inflation_comparison.png")


def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
) -> None:
    """ROC curves for all models on the full test set.

    Args:
        roc_data: {model_name: (fpr_array, tpr_array, roc_auc_score)}
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "tomato", "mediumseagreen"]
    for (model_name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{model_name} (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Alarm Rate (FPR)")
    ax.set_ylabel("Detection Rate (TPR)")
    ax.set_title("ROC Curves — Full KDDTest+")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save_fig("roc_curves_comparison.png")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_experiment_report(
    env_info: dict,
    train_info: dict,
    test_info: dict,
    all_results: dict[str, dict[str, dict]],
    novel_by_category: dict[str, dict[str, dict]],
    cohort_sizes: dict[str, int],
    timestamp: str,
    seed: int,
) -> str:
    """Generate the full Markdown experiment report.

    Args:
        env_info: Environment version strings.
        train_info: Training set info (rows, checksum, etc.).
        test_info: Test set info.
        all_results: {model_name: {cohort: metric_dict}}
        novel_by_category: {model_name: {category_cohort: metric_dict}}
        cohort_sizes: Programmatically derived cohort sample counts.
        timestamp: ISO 8601 execution timestamp.
        seed: Random seed used.

    Returns:
        Markdown string of the full report.
    """
    lines: list[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def para(text: str) -> None:
        lines.append(f"\n{text}\n")

    def table_row(*cells: str) -> str:
        return "| " + " | ".join(str(c) for c in cells) + " |"

    def table_sep(*aligns: str) -> str:
        mapping = {"l": ":---", "r": "---:", "c": ":---:"}
        return "| " + " | ".join(mapping.get(a, "---") for a in aligns) + " |"

    lines.append("# EXP-002 — Supervised Baseline Binary Intrusion Detection\n")
    lines.append(f"*Generated: {timestamp} UTC*  \n*Random seed: {seed}*\n")
    lines.append("> **STATUS:** COMPLETE — All metrics computed from actual model outputs.\n")

    h(2, "Research Question")
    para(
        "What baseline binary classification performance (Normal vs. Attack) do "
        "Logistic Regression, Decision Tree, and Random Forest achieve under strict "
        "leakage-free evaluation on NSL-KDD? How does detection rate degrade on novel "
        "attack types? Do exact train/test duplicate samples inflate reported performance?"
    )

    h(2, "Dataset")
    lines.append(table_row("File", "Rows", "SHA-256"))
    lines.append(table_sep("l", "r", "l"))
    lines.append(table_row(
        "KDDTrain+.txt", f"{train_info['rows']:,}", train_info["sha256"]
    ))
    lines.append(table_row(
        "KDDTest+.txt", f"{test_info['rows']:,}", test_info["sha256"]
    ))
    lines.append("")

    h(2, "Evaluation Cohort Sizes")
    lines.append(table_row("Cohort", "N", "Description"))
    lines.append(table_sep("l", "r", "l"))
    descriptions = {
        "full_test": "Official primary benchmark (all KDDTest+ samples)",
        "non_overlapping": "Strictly unseen feature-label vectors",
        "overlapping": "Exact duplicates of training records (LEAK-003)",
        "known_attacks": "Test attacks with label seen in training",
        "novel_attacks": "Test attacks with label NOT seen in training",
    }
    for cohort, desc in descriptions.items():
        lines.append(table_row(cohort, f"{cohort_sizes.get(cohort, '?'):,}", desc))
    lines.append("")
    para(
        "> **Note:** The full KDDTest+ result is the primary benchmark for "
        "literature comparison but is **not** a pure unseen-sample generalization "
        "estimate due to the 610 exact train/test duplicate samples."
    )

    h(2, "Primary Results — Full KDDTest+")
    lines.append(table_row(
        "Model", "Accuracy", "Precision", "Detection Rate", "FAR", "F1", "ROC-AUC", "Train Time (s)"
    ))
    lines.append(table_sep("l", "r", "r", "r", "r", "r", "r", "r"))
    model_names = list(all_results.keys())
    for model_name in model_names:
        m = all_results[model_name].get("full_test", {})
        lines.append(table_row(
            model_name,
            f"{m.get('accuracy', 0)*100:.2f}%",
            f"{m.get('precision', 0)*100:.2f}%",
            f"{m.get('recall_detection_rate', 0)*100:.2f}%",
            f"{m.get('false_alarm_rate', 0)*100:.2f}%",
            f"{m.get('f1_score', 0)*100:.2f}%",
            f"{m.get('roc_auc', float('nan')):.4f}",
            f"{m.get('train_time_s', float('nan')):.2f}",
        ))
    lines.append("")

    h(2, "Overlap Cohort Analysis")
    lines.append(table_row("Model", "Full KDDTest+ Acc.", "Non-Overlapping Acc.", "Overlapping Acc.", "Δ (Overlap − Non-Overlap)"))
    lines.append(table_sep("l", "r", "r", "r", "r"))
    for model_name in model_names:
        full_acc = all_results[model_name].get("full_test", {}).get("accuracy", 0) * 100
        non_ov_acc = all_results[model_name].get("non_overlapping", {}).get("accuracy", 0) * 100
        ov_acc = all_results[model_name].get("overlapping", {}).get("accuracy", 0) * 100
        delta = ov_acc - non_ov_acc
        lines.append(table_row(
            model_name,
            f"{full_acc:.2f}%",
            f"{non_ov_acc:.2f}%",
            f"{ov_acc:.2f}%",
            f"{delta:+.2f}pp",
        ))
    lines.append("")

    h(2, "Known vs. Novel Attack Detection Rates")
    lines.append(table_row("Model", "Known Attack DR", "Novel Attack DR", "Δ (Known − Novel)"))
    lines.append(table_sep("l", "r", "r", "r"))
    for model_name in model_names:
        known_dr = all_results[model_name].get("known_attacks", {}).get("recall_detection_rate", 0) * 100
        novel_dr = all_results[model_name].get("novel_attacks", {}).get("recall_detection_rate", 0) * 100
        delta = known_dr - novel_dr
        lines.append(table_row(
            model_name,
            f"{known_dr:.2f}%",
            f"{novel_dr:.2f}%",
            f"{delta:+.2f}pp",
        ))
    lines.append("")

    h(2, "Novel Attack Detection by Category")
    for model_name in model_names:
        h(3, model_name)
        cat_results = novel_by_category.get(model_name, {})
        if not cat_results:
            para("No novel attack category results.")
            continue
        lines.append(table_row("Category", "N Samples", "Detection Rate", "Attack Labels"))
        lines.append(table_sep("l", "r", "r", "l"))
        for cohort_key, m in sorted(cat_results.items()):
            lbl_str = ", ".join(m.get("attack_labels", []))
            lines.append(table_row(
                cohort_key.replace("novel_", "").replace("_attacks", "").upper(),
                f"{m.get('n_samples', 0):,}",
                f"{m.get('recall_detection_rate', 0)*100:.2f}%",
                lbl_str,
            ))
        lines.append("")

    h(2, "Reproducibility")
    lines.append(table_row("Item", "Value"))
    lines.append(table_sep("l", "l"))
    lines.append(table_row("Experiment ID", "EXP-002"))
    lines.append(table_row("Random seed", str(seed)))
    lines.append(table_row("Execution timestamp", timestamp))
    lines.append(table_row("Python", env_info.get("python", "?")))
    lines.append(table_row("Platform", env_info.get("platform", "?")))
    lines.append(table_row("scikit-learn", env_info.get("scikit_learn", "?")))
    lines.append(table_row("pandas", env_info.get("pandas", "?")))
    lines.append(table_row("numpy", env_info.get("numpy", "?")))
    lines.append("")

    h(2, "Limitations")
    para(
        "1. Binary classification groups high-severity attacks (U2R/R2L) with "
        "high-volume attacks (DoS) — per-category detection rates provide finer resolution.\n\n"
        "2. `num_outbound_cmds` dropped (zero-variance per EXP-001); "
        "40 features used for modelling.\n\n"
        "3. Results evaluate historical DARPA/KDD benchmark data (1998). "
        "Generalisation to modern network traffic is not implied."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(
    train_path: Path = DEFAULT_TRAIN,
    test_path: Path = DEFAULT_TEST,
    seed: int = RANDOM_SEED,
) -> None:
    """Run the full EXP-002 experiment pipeline.

    Args:
        train_path: Path to KDDTrain+.txt.
        test_path: Path to KDDTest+.txt.
        seed: Random seed for all stochastic operations.
    """
    import datetime
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Create output directories
    for d in (FIG_DIR, METRICS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("EXP-002: Supervised Baseline Binary IDS")
    log.info("Timestamp : %s", timestamp)
    log.info("Seed      : %d", seed)
    log.info("Train     : %s", train_path)
    log.info("Test      : %s", test_path)
    log.info("=" * 60)

    env_info = get_environment_info()
    log.info("Python %s | pandas %s | scikit-learn %s",
             env_info["python"], env_info["pandas"], env_info["scikit_learn"])

    # ---- Checksums
    log.info("Computing file checksums …")
    train_sha = sha256_file(train_path)
    test_sha = sha256_file(test_path)
    log.info("Train SHA-256: %s", train_sha)
    log.info("Test  SHA-256: %s", test_sha)

    # ---- Load datasets
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    # ---- Feature columns (drop metadata)
    feature_cols = get_feature_columns(train_df)
    log.info("Feature columns: %d (after dropping difficulty and num_outbound_cmds)",
             len(feature_cols))

    # ---- Binary targets
    y_train_full = make_binary_target(train_df["label"])
    y_test = make_binary_target(test_df["label"])

    # ---- Internal validation split (within KDDTrain+ only)
    log.info("Creating internal 80/20 stratified validation split …")
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=VALIDATION_SIZE,
        stratify=y_train_full.values,
        random_state=seed,
    )
    X_tr = train_df.iloc[train_idx][feature_cols].values
    y_tr = y_train_full.iloc[train_idx].values
    X_val = train_df.iloc[val_idx][feature_cols].values
    y_val = y_train_full.iloc[val_idx].values
    log.info("Train partition: %d rows | Validation partition: %d rows",
             len(train_idx), len(val_idx))

    # ---- Derive cohort indices BEFORE any training (uses train labels only)
    log.info("Deriving evaluation cohorts …")
    train_labels: set[str] = set(str(x).strip() for x in train_df["label"].unique())
    overlap_mask = identify_overlap_mask(train_df, test_df, feature_cols)
    cohorts = build_cohorts(test_df, overlap_mask, train_labels)

    cohort_sizes = {name: len(idx) for name, idx in cohorts.items()}
    novel_labels: set[str] = {
        str(x).strip()
        for x in test_df.loc[cohorts["novel_attacks"], "label"].unique()
    }
    log.info("Cohort sizes: %s", cohort_sizes)
    log.info("Novel test-only labels (%d): %s", len(novel_labels), sorted(novel_labels))

    # ---- Model definitions
    model_defs: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=1000, random_state=seed
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=20, random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
    }

    # ---- Train, validate, evaluate
    all_results: dict[str, dict[str, dict]] = {}
    novel_by_category: dict[str, dict[str, dict]] = {}
    train_times: dict[str, float] = {}
    roc_data: dict[str, tuple] = {}

    from sklearn.metrics import roc_curve

    for model_name, model_obj in model_defs.items():
        log.info("-" * 50)
        log.info("Training: %s …", model_name)

        # Build pipeline (preprocessing fitted on train partition)
        pipeline = build_pipeline(model_obj, feature_cols)

        t0 = time.time()
        pipeline.fit(X_tr, y_tr)
        train_time = round(time.time() - t0, 4)
        train_times[model_name] = train_time
        log.info("  Training time: %.2f seconds", train_time)

        # Validation sanity check
        val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        log.info("  Validation — Accuracy: %.4f | F1: %.4f", val_acc, val_f1)

        # Evaluate on all test cohorts
        cohort_results = evaluate_model_on_cohorts(
            pipeline, test_df, cohorts, feature_cols
        )
        # Attach training time to full_test metric
        cohort_results["full_test"]["train_time_s"] = train_time

        all_results[model_name] = cohort_results

        # Novel attacks by category
        novel_by_category[model_name] = evaluate_novel_attacks_by_category(
            pipeline, test_df, feature_cols, novel_labels
        )

        # ROC curve data (for models that support predict_proba)
        has_proba = hasattr(pipeline.named_steps["classifier"], "predict_proba")
        if has_proba:
            X_test_all = test_df[feature_cols].values
            y_test_np = y_test.values
            y_prob_all = pipeline.predict_proba(X_test_all)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_np, y_prob_all)
            auc_val = cohort_results["full_test"].get("roc_auc", 0.0)
            roc_data[model_name] = (fpr, tpr, auc_val)

        log.info("  Full KDDTest+ — Acc: %.4f | DR: %.4f | FAR: %.4f | F1: %.4f",
                 cohort_results["full_test"]["accuracy"],
                 cohort_results["full_test"]["recall_detection_rate"],
                 cohort_results["full_test"]["false_alarm_rate"],
                 cohort_results["full_test"]["f1_score"])

    # ---- Generate figures
    log.info("Generating figures …")
    plot_model_comparison_bar(all_results)
    plot_known_vs_novel(all_results)
    plot_overlap_inflation(all_results)
    if roc_data:
        plot_roc_curves(roc_data)

    # ---- Save JSON metrics
    log.info("Saving JSON metrics …")
    summary = {
        "experiment_id": "EXP-002",
        "timestamp": timestamp,
        "random_seed": seed,
        "environment": env_info,
        "train_file": {"path": str(train_path), "rows": len(train_df), "sha256": train_sha},
        "test_file": {"path": str(test_path), "rows": len(test_df), "sha256": test_sha},
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "cohort_sizes": cohort_sizes,
        "novel_labels": sorted(novel_labels),
        "train_times_s": train_times,
        "results": all_results,
    }
    (METRICS_DIR / "summary_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    cohort_breakdown = {
        model: {
            **cohort_results,
            "novel_by_category": novel_by_category.get(model, {}),
        }
        for model, cohort_results in all_results.items()
    }
    (METRICS_DIR / "cohort_breakdowns.json").write_text(
        json.dumps(cohort_breakdown, indent=2), encoding="utf-8"
    )
    log.info("JSON metrics saved.")

    # ---- Save CSV tables
    log.info("Saving CSV tables …")
    # Main results table
    main_rows = []
    for model_name in model_defs:
        m = all_results[model_name]["full_test"]
        main_rows.append({
            "model": model_name,
            "accuracy_pct": round(m.get("accuracy", 0) * 100, 2),
            "precision_pct": round(m.get("precision", 0) * 100, 2),
            "detection_rate_pct": round(m.get("recall_detection_rate", 0) * 100, 2),
            "false_alarm_rate_pct": round(m.get("false_alarm_rate", 0) * 100, 2),
            "f1_score_pct": round(m.get("f1_score", 0) * 100, 2),
            "roc_auc": round(m.get("roc_auc", float("nan")), 4),
            "train_time_s": round(m.get("train_time_s", 0), 2),
        })
    pd.DataFrame(main_rows).to_csv(
        TABLES_DIR / "poster_main_results_table.csv", index=False
    )

    # Overlap breakdown table
    ov_rows = []
    for model_name in model_defs:
        ov_rows.append({
            "model": model_name,
            "full_test_accuracy_pct": round(
                all_results[model_name]["full_test"].get("accuracy", 0) * 100, 2),
            "non_overlapping_accuracy_pct": round(
                all_results[model_name]["non_overlapping"].get("accuracy", 0) * 100, 2),
            "overlapping_accuracy_pct": round(
                all_results[model_name]["overlapping"].get("accuracy", 0) * 100, 2),
            "delta_pp": round(
                (all_results[model_name]["overlapping"].get("accuracy", 0)
                 - all_results[model_name]["non_overlapping"].get("accuracy", 0)) * 100, 2),
        })
    pd.DataFrame(ov_rows).to_csv(
        TABLES_DIR / "overlap_breakdown_table.csv", index=False
    )

    # Novel attacks by category table
    novel_rows = []
    for model_name in model_defs:
        for cat_key, m in novel_by_category.get(model_name, {}).items():
            novel_rows.append({
                "model": model_name,
                "category_cohort": cat_key,
                "n_samples": m.get("n_samples", 0),
                "detection_rate_pct": round(m.get("recall_detection_rate", 0) * 100, 2),
                "attack_labels": "; ".join(m.get("attack_labels", [])),
            })
    pd.DataFrame(novel_rows).to_csv(
        TABLES_DIR / "novel_attacks_by_category.csv", index=False
    )
    log.info("CSV tables saved.")

    # ---- Generate and save the Markdown experiment report
    log.info("Generating experiment report …")
    report_text = generate_experiment_report(
        env_info=env_info,
        train_info={"rows": len(train_df), "sha256": train_sha},
        test_info={"rows": len(test_df), "sha256": test_sha},
        all_results=all_results,
        novel_by_category=novel_by_category,
        cohort_sizes=cohort_sizes,
        timestamp=timestamp,
        seed=seed,
    )
    report_path = RESULTS_DIR / "experiment_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Report written to: %s", report_path)

    log.info("=" * 60)
    log.info("EXP-002 complete. Results in: %s", RESULTS_DIR)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EXP-002: Supervised Baseline Binary IDS — SentinelNet"
    )
    parser.add_argument(
        "--train", type=Path, default=DEFAULT_TRAIN,
        help="Path to KDDTrain+.txt (default: data/raw/KDDTrain+.txt)"
    )
    parser.add_argument(
        "--test", type=Path, default=DEFAULT_TEST,
        help="Path to KDDTest+.txt (default: data/raw/KDDTest+.txt)"
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()
    main(train_path=args.train, test_path=args.test, seed=args.seed)
