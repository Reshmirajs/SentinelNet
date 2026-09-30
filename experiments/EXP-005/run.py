"""
EXP-005: Threshold / Operating-Point Analysis
=============================================
SentinelNet Research Project

Purpose
-------
Investigate how the classification threshold affects the trade-off between
attack detection rate and false-alarm rate for the three binary classifiers
evaluated in EXP-002 (Logistic Regression, Decision Tree, Random Forest).

Key Methodological Decisions
-----------------------------
- Same pipeline (preprocessing, features, hyperparameters, seed) as EXP-002/003/004.
- Threshold grid: 0.00 to 1.00 in steps of 0.01 (101 candidate thresholds).
- Attack probability score: predict_proba(X)[:, 1]  — continuous score, not hard prediction.
- Threshold selection: validation partition only (20% of KDDTrain+).
- PRE-SPECIFIED selection rule:
    Primary: lowest threshold achieving >= 90% validation Detection Rate,
             provided FAR is finite and defined.
    Fallback: threshold with highest validation Detection Rate (FAR tie-break).
- KDDTest+ is quarantined until threshold is frozen for ALL models.
- EXP-002 results (default threshold 0.50) are read-only reference for comparison.
- No model is called "best".
- No causal claims.
- No production-readiness claims.

Usage
-----
    python experiments/EXP-005/run.py [--train PATH] [--test PATH] [--seed INT]

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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] EXP-005 | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("EXP-005")

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
EXP002_SUMMARY = REPO_ROOT / "experiments" / "EXP-002" / "results" / "metrics" / "summary_results.json"

# ---------------------------------------------------------------------------
# Constants (identical to EXP-002/003/004)
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

CATEGORICAL_FEATURES: list[str] = ["protocol_type", "service", "flag"]
BINARY_FEATURES: list[str] = [
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
]
DROP_FEATURES: list[str] = ["difficulty", "num_outbound_cmds"]

RANDOM_SEED: int = 42
VALIDATION_SIZE: float = 0.20

# Threshold grid: 0.00, 0.01, ..., 1.00  (101 values)
THRESHOLD_GRID: np.ndarray = np.round(np.arange(0.00, 1.01, 0.01), 2)

# Pre-specified threshold selection rule (must not use KDDTest+)
SELECTION_DR_TARGET: float = 0.90   # 90% validation Detection Rate target
DEFAULT_THRESHOLD: float = 0.50     # EXP-002 baseline threshold


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hexdigest of a file.

    Args:
        path: Path to the file to hash.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_environment_info() -> dict[str, str]:
    """Collect platform and library version metadata.

    Returns:
        Dictionary of environment strings.
    """
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
    """Load a raw NSL-KDD CSV file with the canonical 43-column schema.

    Args:
        path: Path to KDDTrain+.txt or KDDTest+.txt.

    Returns:
        DataFrame with 43 named columns.

    Raises:
        FileNotFoundError: If the file does not exist.
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
        raise ValueError(f"Expected 43 columns, got {df.shape[1]} in {path.name}")
    log.info("  Loaded %d rows × %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Feature extraction and target encoding
# ---------------------------------------------------------------------------

def make_binary_target(labels: pd.Series) -> pd.Series:
    """Encode label column as binary: normal=0, attack=1.

    Args:
        labels: Raw NSL-KDD label column (string values).

    Returns:
        Integer Series of 0/1 values.
    """
    return (labels.str.strip() != "normal").astype(int)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return ordered feature column names excluding label and dropped metadata.

    Args:
        df: Full 43-column NSL-KDD DataFrame.

    Returns:
        List of column names suitable for modelling (40 features).
    """
    exclude = {"label", "difficulty"} | set(DROP_FEATURES)
    return [c for c in df.columns if c not in exclude]


def get_numerical_features(feature_cols: list[str]) -> list[str]:
    """Return numerical (non-categorical, non-binary) feature names.

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
    """Build a leakage-safe sklearn Pipeline (identical to EXP-002/003/004).

    All transformers are fitted exclusively on the training partition.

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
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_indices),
            ("bin", "passthrough", bin_indices),
            ("num", StandardScaler(), num_indices),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])


# ---------------------------------------------------------------------------
# Threshold sweep utilities
# ---------------------------------------------------------------------------

def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a scalar threshold to continuous attack probability scores.

    Args:
        scores: 1-D array of P(Attack) values in [0, 1].
        threshold: Decision boundary; score >= threshold → predicted Attack (1).

    Returns:
        Integer array of predictions (0 = Normal, 1 = Attack).
    """
    return (scores >= threshold).astype(int)


def compute_threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Compute binary classification metrics at a given threshold.

    Args:
        y_true: Ground-truth binary labels (0 = Normal, 1 = Attack).
        scores: Continuous attack probability scores P(Attack).
        threshold: Decision threshold.

    Returns:
        Dictionary of metrics: tp, tn, fp, fn, detection_rate, far,
        precision, f1, accuracy, threshold.
    """
    y_pred = apply_threshold(scores, threshold)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n_pos = tp + fn  # total positives (attacks)
    n_neg = tn + fp  # total negatives (normals)

    detection_rate = tp / n_pos if n_pos > 0 else 0.0
    far = fp / n_neg if n_neg > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * detection_rate / (precision + detection_rate)
          if (precision + detection_rate) > 0 else 0.0)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "detection_rate": round(detection_rate, 6),
        "far": round(far, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    grid: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Evaluate all candidate thresholds on a dataset split.

    Args:
        y_true: Ground-truth binary labels.
        scores: Continuous attack probability scores.
        grid: Array of threshold candidates to evaluate.
        model_name: Name of the model (for the 'model' column).

    Returns:
        DataFrame with one row per threshold, containing all metrics.
    """
    rows = []
    for thr in grid:
        m = compute_threshold_metrics(y_true, scores, thr)
        m["model"] = model_name
        rows.append(m)
    df = pd.DataFrame(rows)
    df = df[["model", "threshold", "tp", "tn", "fp", "fn",
             "detection_rate", "far", "precision", "f1", "accuracy"]]
    return df


# ---------------------------------------------------------------------------
# Threshold selection (validation data only — PRE-SPECIFIED RULE)
# ---------------------------------------------------------------------------

def select_threshold(
    val_sweep: pd.DataFrame,
    dr_target: float = SELECTION_DR_TARGET,
) -> dict[str, Any]:
    """Select operating threshold using the pre-specified validation rule.

    Pre-specified rule (must not use KDDTest+):
        Primary: Select the HIGHEST threshold that achieves a validation
                 Detection Rate >= dr_target (90%).
        Fallback: If no threshold meets the target:
                  1. select threshold with highest validation Detection Rate;
                  2. if tied, select threshold with lowest validation FAR;
                  3. if still tied, select highest threshold.

    Args:
        val_sweep: DataFrame of threshold sweep results on validation data.
        dr_target: Minimum validation Detection Rate target (default 0.90).

    Returns:
        Dictionary with selection metadata: selected_threshold, selection_rule,
        validation_detection_rate, validation_far, validation_precision,
        validation_f1.
    """
    # Primary: candidates that reach the DR target
    primary_candidates = val_sweep[val_sweep["detection_rate"] >= dr_target]

    if len(primary_candidates) > 0:
        # HIGHEST threshold achieving the target
        chosen_row = primary_candidates.loc[primary_candidates["threshold"].idxmax()]
        rule = f"primary: highest threshold >= {dr_target:.0%} validation detection rate"
    else:
        # Fallback:
        # 1. select highest DR
        # 2. if tied, lowest FAR
        # 3. if still tied, highest threshold
        max_dr = val_sweep["detection_rate"].max()
        dr_candidates = val_sweep[val_sweep["detection_rate"] == max_dr]
        min_far = dr_candidates["far"].min()
        far_candidates = dr_candidates[dr_candidates["far"] == min_far]
        chosen_row = far_candidates.loc[far_candidates["threshold"].idxmax()]
        rule = (
            f"fallback: no threshold achieved >= {dr_target:.0%} validation DR; "
            f"selected highest validation DR ({max_dr:.4f}), lowest FAR on tie, highest threshold on tie"
        )

    return {
        "selected_threshold": float(chosen_row["threshold"]),
        "selection_rule": rule,
        "validation_detection_rate": float(chosen_row["detection_rate"]),
        "validation_far": float(chosen_row["far"]),
        "validation_precision": float(chosen_row["precision"]),
        "validation_f1": float(chosen_row["f1"]),
    }


# ---------------------------------------------------------------------------
# Final test evaluation (called AFTER threshold is frozen)
# ---------------------------------------------------------------------------

def evaluate_on_test(
    y_test: np.ndarray,
    test_scores: np.ndarray,
    selected_threshold: float,
    model_name: str,
) -> dict[str, Any]:
    """Evaluate a frozen threshold on the quarantined KDDTest+ set.

    Args:
        y_test: Ground-truth binary labels from KDDTest+.
        test_scores: Continuous attack probability scores from KDDTest+.
        selected_threshold: The threshold frozen from validation selection.
        model_name: Name of the model.

    Returns:
        Dictionary of test-set metrics at the selected threshold,
        plus threshold-independent ROC-AUC.
    """
    metrics = compute_threshold_metrics(y_test, test_scores, selected_threshold)
    roc_auc = float(roc_auc_score(y_test, test_scores))
    metrics["roc_auc"] = round(roc_auc, 6)
    metrics["model"] = model_name
    metrics["selected_threshold"] = selected_threshold
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Save figure to results/figures/ and results/ root.

    Args:
        fig: matplotlib Figure to save.
        filename: Filename (basename only, .png extension).
    """
    for dest in (FIG_DIR / filename, RESULTS_DIR / filename):
        fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", filename)


def plot_threshold_vs_detection_far(
    val_sweeps: dict[str, pd.DataFrame],
    selected: dict[str, dict],
) -> None:
    """Plot Detection Rate and FAR vs. threshold for each model.

    Args:
        val_sweeps: Dict mapping model name → validation sweep DataFrame.
        selected: Dict mapping model name → selection result dict.
    """
    model_names = list(val_sweeps.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    colors_dr = ["#1f77b4", "#2ca02c", "#d62728"]
    colors_far = ["#aec7e8", "#98df8a", "#ff9896"]

    for ax, model_name, c_dr, c_far in zip(axes, model_names, colors_dr, colors_far):
        sweep = val_sweeps[model_name]
        sel = selected[model_name]
        thr_sel = sel["selected_threshold"]

        ax.plot(sweep["threshold"], sweep["detection_rate"] * 100,
                color=c_dr, lw=1.8, label="Detection Rate (%)")
        ax.plot(sweep["threshold"], sweep["far"] * 100,
                color=c_far, lw=1.8, linestyle="--", label="FAR (%)")

        # Mark selected threshold
        sel_dr = sel["validation_detection_rate"] * 100
        sel_far = sel["validation_far"] * 100
        ax.axvline(thr_sel, color="black", linestyle=":", lw=1.2,
                   label=f"Selected θ = {thr_sel:.2f}")
        ax.scatter([thr_sel], [sel_dr], color=c_dr, zorder=5, s=60)
        ax.scatter([thr_sel], [sel_far], color=c_far, zorder=5, s=60)

        # Mark default threshold
        ax.axvline(DEFAULT_THRESHOLD, color="gray", linestyle="--", lw=1.0,
                   alpha=0.7, label=f"Default θ = {DEFAULT_THRESHOLD:.2f}")

        ax.set_xlabel("Threshold", fontsize=10)
        ax.set_ylabel("Rate (%)", fontsize=10)
        ax.set_title(model_name, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7.5, loc="center right")
        ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Validation Detection Rate & FAR vs. Classification Threshold\n"
        "(Vertical lines: selected and default thresholds)",
        fontsize=11,
    )
    plt.tight_layout()
    _save_figure(fig, "threshold_vs_detection_far.png")


def plot_roc_curves(
    y_test: np.ndarray,
    test_scores_dict: dict[str, np.ndarray],
    selected: dict[str, dict],
    test_metrics_dict: dict[str, dict],
) -> None:
    """Plot ROC curves for all models, marking the selected operating point.

    Args:
        y_test: Ground-truth binary labels from KDDTest+.
        test_scores_dict: Dict mapping model name → test continuous scores.
        selected: Dict mapping model name → selection result dict.
        test_metrics_dict: Dict mapping model name → test evaluation metrics.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^"]

    for (model_name, scores), color, marker in zip(
        test_scores_dict.items(), colors, markers
    ):
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc = roc_auc_score(y_test, scores)
        ax.plot(fpr * 100, tpr * 100, color=color, lw=1.8,
                label=f"{model_name} (AUC = {auc:.4f})")

        # Mark the selected operating point
        tm = test_metrics_dict[model_name]
        op_far = tm["far"] * 100
        op_dr = tm["detection_rate"] * 100
        sel_thr = selected[model_name]["selected_threshold"]
        ax.scatter([op_far], [op_dr], color=color, marker=marker, s=90, zorder=5,
                   label=f"  Selected θ={sel_thr:.2f} (FAR={op_far:.1f}%, DR={op_dr:.1f}%)")

    ax.plot([0, 100], [0, 100], "k--", lw=0.8, alpha=0.5, label="Random classifier")
    ax.set_xlabel("False Alarm Rate (%) [= 100 × FPR]", fontsize=10)
    ax.set_ylabel("Detection Rate (%) [= 100 × TPR]", fontsize=10)
    ax.set_title(
        "ROC Curves — KDDTest+ (Continuous Scores)\n"
        "Markers indicate validation-selected operating points",
        fontsize=10,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save_figure(fig, "roc_curve_thresholds.png")


def plot_default_vs_selected(
    comparison_df: pd.DataFrame,
) -> None:
    """Compare default threshold vs. validation-selected threshold on KDDTest+.

    Args:
        comparison_df: DataFrame from build_comparison_table().
    """
    models = comparison_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.30

    metrics = [
        ("detection_rate", "Detection Rate (%)", ["default_detection_rate", "selected_detection_rate"]),
        ("far", "FAR (%)", ["default_far", "selected_far"]),
        ("f1", "F1 Score (%)", ["default_f1", "selected_f1"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, (metric_key, ylabel, cols) in zip(axes, metrics):
        default_vals = comparison_df[cols[0]].tolist()
        selected_vals = comparison_df[cols[1]].tolist()

        b1 = ax.bar(x - width / 2, default_vals, width,
                    label=f"Default θ={DEFAULT_THRESHOLD:.2f}", color="#6baed6", edgecolor="white")
        b2 = ax.bar(x + width / 2, selected_vals, width,
                    label="Validation-selected θ", color="#fd8d3c", edgecolor="white")

        for bar in b1:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h + 0.5),
                        ha="center", va="bottom", fontsize=7.5)
        for bar in b2:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h + 0.5),
                        ha="center", va="bottom", fontsize=7.5)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
        ax.legend(fontsize=8)
        ax.set_ylim(0, max(max(default_vals), max(selected_vals)) * 1.20 + 5)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.suptitle(
        "KDDTest+ Performance: Default Threshold (0.50) vs. Validation-Selected Threshold\n"
        "Note: appropriate operating point depends on relative cost of missed attacks and false alarms.",
        fontsize=10,
    )
    plt.tight_layout()
    _save_figure(fig, "default_vs_selected_test_metrics.png")


# ---------------------------------------------------------------------------
# Results table builders
# ---------------------------------------------------------------------------

def build_validation_sweep_csv(
    val_sweeps: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine per-model validation sweep DataFrames into one CSV-ready table.

    Args:
        val_sweeps: Dict mapping model name → validation sweep DataFrame.

    Returns:
        Combined DataFrame with required columns for validation_threshold_sweep.csv.
    """
    parts = []
    for model_name, df in val_sweeps.items():
        part = df[["model", "threshold", "tp", "tn", "fp", "fn",
                   "detection_rate", "far", "precision", "f1"]].copy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def build_selected_thresholds_csv(
    selected: dict[str, dict],
) -> pd.DataFrame:
    """Build the selected_thresholds.csv table.

    Args:
        selected: Dict mapping model name → selection result dict.

    Returns:
        DataFrame with one row per model.
    """
    rows = []
    for model_name, sel in selected.items():
        rows.append({
            "model": model_name,
            "selection_rule": sel["selection_rule"],
            "selected_threshold": sel["selected_threshold"],
            "validation_detection_rate": sel["validation_detection_rate"],
            "validation_far": sel["validation_far"],
            "validation_precision": sel["validation_precision"],
            "validation_f1": sel["validation_f1"],
        })
    return pd.DataFrame(rows)


def build_comparison_table(
    selected: dict[str, dict],
    test_metrics_dict: dict[str, dict],
    exp002_results: dict[str, Any],
) -> pd.DataFrame:
    """Build the threshold_operating_point_results.csv comparison table.

    Default (0.50) values are read from EXP-002 results; selected-threshold
    values come from EXP-005 test evaluation. EXP-002 is not modified.

    Args:
        selected: Dict mapping model name → selection dict.
        test_metrics_dict: Dict mapping model name → test evaluation metrics
                           at the selected threshold.
        exp002_results: Parsed EXP-002 summary_results.json content.

    Returns:
        DataFrame with one row per model and all comparison columns.
    """
    model_names = list(test_metrics_dict.keys())
    rows = []

    for model_name in model_names:
        sel = selected[model_name]
        tm = test_metrics_dict[model_name]

        # EXP-002 default-threshold results (read-only reference)
        e2 = exp002_results["results"][model_name]["full_test"]

        rows.append({
            "model": model_name,
            "default_threshold": DEFAULT_THRESHOLD,
            "selected_threshold": sel["selected_threshold"],
            # Default (EXP-002) metrics — expressed as % for display
            "default_accuracy": round(e2["accuracy"] * 100, 2),
            "selected_accuracy": round(tm["accuracy"] * 100, 2),
            "default_precision": round(e2["precision"] * 100, 2),
            "selected_precision": round(tm["precision"] * 100, 2),
            "default_detection_rate": round(e2["recall_detection_rate"] * 100, 2),
            "selected_detection_rate": round(tm["detection_rate"] * 100, 2),
            "default_far": round(e2["false_alarm_rate"] * 100, 2),
            "selected_far": round(tm["far"] * 100, 2),
            "default_f1": round(e2["f1_score"] * 100, 2),
            "selected_f1": round(tm["f1"] * 100, 2),
            "roc_auc": tm["roc_auc"],
            "default_tp": e2["tp"],
            "default_tn": e2["tn"],
            "default_fp": e2["fp"],
            "default_fn": e2["fn"],
            "selected_tp": tm["tp"],
            "selected_tn": tm["tn"],
            "selected_fp": tm["fp"],
            "selected_fn": tm["fn"],
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_experiment_report(
    env_info: dict,
    train_info: dict,
    test_info: dict,
    selected_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    val_sweeps: dict[str, pd.DataFrame],
    timestamp: str,
    seed: int,
) -> str:
    """Generate the Markdown experiment report for EXP-005.

    Args:
        env_info: Environment metadata dict.
        train_info: Train file info dict (path, rows, sha256).
        test_info: Test file info dict (path, rows, sha256).
        selected_df: selected_thresholds DataFrame.
        comparison_df: threshold_operating_point_results DataFrame.
        val_sweeps: Dict mapping model name → validation sweep DataFrame.
        timestamp: ISO 8601 UTC execution timestamp.
        seed: Random seed used.

    Returns:
        Markdown report string.
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

    lines.append("# EXP-005 — Threshold / Operating-Point Analysis\n")
    lines.append(f"*Generated: {timestamp} UTC*  \n*Random seed: {seed}*\n")
    lines.append("> **STATUS:** COMPLETE — All threshold evaluations computed from actual model outputs.\n")

    h(2, "1. Research Question")
    para(
        "RQ5: How does the classification threshold affect the trade-off between "
        "attack detection rate and false-alarm rate for the SentinelNet binary classifiers?"
    )

    h(2, "2. Hypothesis")
    para(
        "H5: Lowering the classification threshold will increase attack detection rate "
        "while also increasing false-alarm rate, producing an observable operating-point trade-off.\n\n"
        "H5 is a directional hypothesis about threshold behavior. "
        "It does not predict which model will exhibit the trade-off most strongly, "
        "nor does it assume that any specific threshold is optimal."
    )

    h(2, "3. Motivation from EXP-002")
    para(
        "EXP-002 evaluated the three binary classifiers at the sklearn default threshold (0.50). "
        "A notable finding was that Random Forest achieved a substantially higher ROC-AUC (0.9535) "
        "than the other classifiers, while its default-threshold detection rate (62.04%) was similar "
        "to Logistic Regression (62.36%) and lower than Decision Tree (69.26%). "
        "ROC-AUC is a threshold-independent measure of ranking ability; it does not directly imply "
        "better detection at any particular threshold. EXP-005 tests whether adjusting the threshold "
        "using a pre-specified validation rule alters the detection/FAR trade-off, "
        "and whether the choice of operating point has a measurable effect on test performance. "
        "EXP-005 does not assume that the ROC-AUC / detection-rate difference observed in EXP-002 "
        "is caused by threshold choice."
    )

    h(2, "4. Dataset")
    para(
        f"- **Train file:** `{train_info['path']}` ({train_info['rows']:,} rows, "
        f"SHA-256: {train_info['sha256'][:16]}...)\n"
        f"- **Test file:** `{test_info['path']}` ({test_info['rows']:,} rows, "
        f"SHA-256: {test_info['sha256'][:16]}...)\n"
        "- All NSL-KDD dataset limitations from EXP-001 apply: historical benchmark "
        "(1998 captures), train/test distribution differences, 610 exact cross-split duplicates, "
        "17 test-only attack labels, and 58 feature-identical conflicting-label records. "
        "EXP-005 does not address these limitations."
    )

    h(2, "5. Experimental Setup")
    para(
        "- **Models:** Logistic Regression, Decision Tree, Random Forest "
        "(identical hyperparameters to EXP-002/003/004).\n"
        "- **Training partition:** 80% stratified split of KDDTrain+ (100,778 rows). "
        "All preprocessing fitted on training partition only.\n"
        "- **Validation partition:** 20% of KDDTrain+ (25,195 rows). "
        "Used exclusively for threshold selection.\n"
        "- **Test set (KDDTest+):** 22,544 rows. Quarantined until threshold is frozen per model.\n"
        "- **Score:** Continuous attack probability P(Attack) = `predict_proba(X)[:, 1]`.\n"
        "- **Prediction rule:** prediction = 1 if attack_score ≥ threshold else 0.\n"
        "- **Threshold grid:** 0.00 to 1.00 in increments of 0.01 (101 candidate thresholds)."
    )

    h(2, "6. Threshold-Selection Rule")
    para(
        "The threshold selection rule was pre-specified to identify a conservative operating point "
        "on the validation partition before evaluating KDDTest+.\n\n"
        "**Corrected Primary Rule:** Select the *highest* validation threshold that achieves a validation "
        "Detection Rate (Recall) of at least 90%.\n\n"
        "**Corrected Fallback Rule:** If no candidate threshold in the 0.00–1.00 grid reaches 90% validation Detection Rate:\n"
        "1. select the threshold with the highest validation Detection Rate;\n"
        "2. if multiple thresholds tie, select the one with the lowest validation FAR;\n"
        "3. if still tied, select the highest threshold.\n\n"
        "**Rationale:** Selecting the highest threshold achieving the target ensures the classifier adopts "
        "the most conservative decision boundary (minimizing false alarms) while still satisfying the "
        "operational detection requirement (≥ 90% Recall).\n\n"
        "> **Quarantine Note:** KDDTest+ was fully quarantined during threshold selection. "
        "No test labels, test predictions, or test metrics were used to determine or tune the threshold."
    )

    h(2, "7. Methodological Correction (Protocol Amendment)")
    para(
        "During initial experiment execution and validation, a methodological flaw in the initial protocol specification was identified:\n\n"
        "- **Initial Rule Formulation:** The initial implementation specified selecting the *lowest* threshold achieving validation Detection Rate ≥ 90%.\n"
        "- **Degenerate Result:** Because the classifier predicts Attack when $P(\\text{Attack}) \\ge \\theta$, setting $\\theta = 0.00$ assigns all samples to the positive class, producing 100% Detection Rate and 100% False Alarm Rate for all models.\n"
        "- **Identification and Non-Acceptance:** This degenerate boundary condition was identified during pre-acceptance audit. The initial 0.00-threshold run was rejected and not accepted as a valid scientific result.\n"
        "- **Protocol Amendment:** The selection rule was formally amended to select the *highest* threshold achieving ≥ 90% validation Detection Rate (with multi-stage tie-breaking for fallback).\n"
        "- **Execution:** The experiment was completely rerun under the corrected protocol. The results presented in this report reflect exclusively the corrected protocol."
    )

    h(2, "8. Validation Threshold Sweep — Selected Thresholds")
    para("The following table shows the validation-selected operating point for each model.")
    lines.append(table_row("Model", "Selection Rule", "Selected θ",
                            "Val. DR (%)", "Val. FAR (%)", "Val. Precision (%)", "Val. F1 (%)"))
    lines.append(table_sep("l", "l", "r", "r", "r", "r", "r"))
    for _, r in selected_df.iterrows():
        lines.append(table_row(
            r["model"],
            r["selection_rule"],
            f"{r['selected_threshold']:.2f}",
            f"{r['validation_detection_rate'] * 100:.2f}%",
            f"{r['validation_far'] * 100:.2f}%",
            f"{r['validation_precision'] * 100:.2f}%",
            f"{r['validation_f1'] * 100:.2f}%",
        ))
    lines.append("")

    h(2, "9. Final KDDTest+ Results at Selected Threshold")
    para("Evaluated after threshold was frozen. KDDTest+ was not used in threshold selection.")
    lines.append(table_row("Model", "Selected θ", "Accuracy (%)", "Precision (%)",
                            "Detection Rate (%)", "FAR (%)", "F1 (%)", "ROC-AUC",
                            "TP", "TN", "FP", "FN"))
    lines.append(table_sep("l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"))
    for _, r in comparison_df.iterrows():
        lines.append(table_row(
            r["model"],
            f"{r['selected_threshold']:.2f}",
            f"{r['selected_accuracy']:.2f}%",
            f"{r['selected_precision']:.2f}%",
            f"{r['selected_detection_rate']:.2f}%",
            f"{r['selected_far']:.2f}%",
            f"{r['selected_f1']:.2f}%",
            f"{r['roc_auc']:.4f}",
            f"{r['selected_tp']:,}",
            f"{r['selected_tn']:,}",
            f"{r['selected_fp']:,}",
            f"{r['selected_fn']:,}",
        ))
    lines.append("")

    h(2, "10. Default vs. Selected Threshold Comparison (KDDTest+)")
    para(
        "Default threshold values are from EXP-002 (full_test cohort, threshold = 0.50). "
        "EXP-002 was not modified. ROC-AUC is threshold-independent and is the same for both rows."
    )
    lines.append(table_row("Model", "Threshold", "Accuracy (%)", "Precision (%)",
                            "Detection Rate (%)", "FAR (%)", "F1 (%)", "TP", "TN", "FP", "FN"))
    lines.append(table_sep("l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"))
    for _, r in comparison_df.iterrows():
        lines.append(table_row(
            r["model"], f"{r['default_threshold']:.2f}",
            f"{r['default_accuracy']:.2f}%", f"{r['default_precision']:.2f}%",
            f"{r['default_detection_rate']:.2f}%", f"{r['default_far']:.2f}%",
            f"{r['default_f1']:.2f}%",
            f"{r['default_tp']:,}", f"{r['default_tn']:,}",
            f"{r['default_fp']:,}", f"{r['default_fn']:,}",
        ))
        lines.append(table_row(
            r["model"], f"{r['selected_threshold']:.2f}",
            f"{r['selected_accuracy']:.2f}%", f"{r['selected_precision']:.2f}%",
            f"{r['selected_detection_rate']:.2f}%", f"{r['selected_far']:.2f}%",
            f"{r['selected_f1']:.2f}%",
            f"{r['selected_tp']:,}", f"{r['selected_tn']:,}",
            f"{r['selected_fp']:,}", f"{r['selected_fn']:,}",
        ))
    lines.append("")

    h(2, "11. Interpretation")
    para(
        "**Threshold independence of ROC-AUC:** ROC-AUC describes the classifier's ability "
        "to rank attack records above normal records across all thresholds. "
        "It does not describe performance at any specific threshold. "
        "Differences in ROC-AUC across models should not be interpreted as directly "
        "implying better detection at the selected operating point.\n\n"
        "**Threshold sensitivity:** The validation sweep demonstrates that detection rate "
        "and false-alarm rate are jointly sensitive to the classification threshold, "
        "consistent with H5. Lowering the threshold increases detection rate "
        "while also increasing false-alarm rate across all models evaluated.\n\n"
        "**Operating-point trade-off:** The selected thresholds represent the operating "
        "point defined by the pre-specified validation rule (highest threshold achieving "
        "≥ 90% validation detection rate, or fallback). "
        "This point is not claimed to be optimal. "
        "The appropriate operating point depends on the relative cost of missed attacks "
        "and false alarms in a given deployment context, which is not measured in this study.\n\n"
        "**Comparison with EXP-002 default threshold:** Changes in detection rate and FAR "
        "between the default threshold (0.50) and the validation-selected threshold are "
        "observed differences; they do not indicate that one threshold is universally preferable.\n\n"
        "**Validation-to-test generalization:** Threshold selection was performed on the "
        "validation partition. Detection rates and FAR values on KDDTest+ may differ from "
        "validation values due to distribution differences between the two splits, "
        "including the presence of test-only attack labels and the 610 cross-split duplicates "
        "documented in EXP-001."
    )

    h(2, "12. Limitations")
    para(
        "1. **Historical benchmark:** NSL-KDD is based on 1998 network traffic captures; "
        "results cannot be generalized to modern network environments.\n\n"
        "2. **Single operating point per model:** A single threshold is selected per model. "
        "In practice, multiple operating points along the ROC curve may be relevant.\n\n"
        "3. **Validation/test distribution shift:** The validation and test partitions have "
        "different label compositions (train-only vs. test-only attack labels); "
        "the selected threshold may not generalize perfectly to all test subgroups.\n\n"
        "4. **Threshold grid resolution:** The 0.01 increment grid may not locate the "
        "exact threshold satisfying the 90% DR target; the reported threshold is the "
        "lowest grid point at or above the target.\n\n"
        "5. **No cost-sensitive analysis:** The 90% DR target was chosen to demonstrate "
        "threshold sensitivity; no empirical security cost analysis was conducted.\n\n"
        "6. **Decision Tree probability calibration:** Decision Tree predict_proba values "
        "are class frequency counts in leaf nodes and may not be well-calibrated probabilities. "
        "This affects the threshold sweep behavior for Decision Tree specifically.\n\n"
        "7. **Known NSL-KDD limitations from EXP-001:** 610 exact cross-split duplicates, "
        "17 test-only attack labels, 58 feature-identical conflicting-label records, "
        "and the constant `num_outbound_cmds` feature remain present and were documented in EXP-001. "
        "EXP-005 does not address these."
    )

    h(2, "13. Reproducibility")
    lines.append(table_row("Item", "Value"))
    lines.append(table_sep("l", "l"))
    lines.append(table_row("Experiment ID", "EXP-005"))
    lines.append(table_row("Random seed", str(seed)))
    lines.append(table_row("Execution timestamp", timestamp))
    lines.append(table_row("Python", env_info.get("python", "?")))
    lines.append(table_row("Platform", env_info.get("platform", "?")))
    lines.append(table_row("scikit-learn", env_info.get("scikit_learn", "?")))
    lines.append(table_row("pandas", env_info.get("pandas", "?")))
    lines.append(table_row("numpy", env_info.get("numpy", "?")))
    lines.append(table_row("matplotlib", env_info.get("matplotlib", "?")))
    lines.append(table_row("Threshold grid", "0.00–1.00 step 0.01 (101 values)"))
    lines.append(table_row("DR target for selection", f"{SELECTION_DR_TARGET:.0%}"))
    lines.append(table_row("Default threshold (EXP-002 baseline)", str(DEFAULT_THRESHOLD)))
    lines.append(table_row(
        "Train file",
        f"{train_info['path']} ({train_info['rows']:,} rows, SHA-256: {train_info['sha256'][:16]}...)"
    ))
    lines.append(table_row(
        "Test file",
        f"{test_info['path']} ({test_info['rows']:,} rows, SHA-256: {test_info['sha256'][:16]}...)"
    ))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(
    train_path: Path = DEFAULT_TRAIN,
    test_path: Path = DEFAULT_TEST,
    seed: int = RANDOM_SEED,
) -> None:
    """Run full EXP-005 experiment pipeline.

    Threshold selection occurs exclusively on the validation partition.
    KDDTest+ is evaluated only after thresholds are frozen.

    Args:
        train_path: Path to KDDTrain+.txt.
        test_path: Path to KDDTest+.txt.
        seed: Random seed for reproducibility.
    """
    import datetime
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for d in (RESULTS_DIR, FIG_DIR, METRICS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("EXP-005: Threshold / Operating-Point Analysis")
    log.info("Timestamp : %s", timestamp)
    log.info("Seed      : %d", seed)
    log.info("Train     : %s", train_path)
    log.info("Test      : %s", test_path)
    log.info("=" * 60)

    env_info = get_environment_info()
    train_sha = sha256_file(train_path)
    test_sha = sha256_file(test_path)

    # Load datasets
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    # Load EXP-002 results (read-only baseline reference)
    log.info("Loading EXP-002 baseline results for comparison (read-only)…")
    with open(EXP002_SUMMARY, encoding="utf-8") as f:
        exp002_results = json.load(f)

    # Feature columns and binary targets
    feature_cols = get_feature_columns(train_df)
    y_train_full = make_binary_target(train_df["label"])
    y_test = make_binary_target(test_df["label"]).values

    # 80/20 train/validation split (identical to EXP-002/003/004)
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
    X_test = test_df[feature_cols].values

    log.info("Training partition: %d rows | Validation: %d rows | Test: %d rows",
             len(X_tr), len(X_val), len(X_test))

    # Model definitions (identical to EXP-002/003/004)
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

    # -----------------------------------------------------------------------
    # Phase 1: Fit models, sweep thresholds on VALIDATION only, select threshold
    # KDDTest+ is NOT used in this phase.
    # -----------------------------------------------------------------------
    val_sweeps: dict[str, pd.DataFrame] = {}
    selected: dict[str, dict] = {}
    test_scores_dict: dict[str, np.ndarray] = {}
    fitted_pipes: dict[str, Pipeline] = {}

    log.info("=" * 60)
    log.info("PHASE 1: Threshold selection on VALIDATION partition only.")
    log.info("KDDTest+ is quarantined until all thresholds are frozen.")
    log.info("=" * 60)

    for model_name, model_obj in model_defs.items():
        log.info("-" * 50)
        log.info("Fitting model: %s …", model_name)
        t0 = time.time()
        pipe = build_pipeline(model_obj, feature_cols)
        pipe.fit(X_tr, y_tr)
        fitted_pipes[model_name] = pipe
        train_time = time.time() - t0
        log.info("  Fitted in %.2f s", train_time)

        # Validation scores (for threshold sweep)
        val_scores = pipe.predict_proba(X_val)[:, 1]

        # Threshold sweep on validation
        log.info("  Sweeping %d thresholds on validation partition…", len(THRESHOLD_GRID))
        sweep_df = sweep_thresholds(y_val, val_scores, THRESHOLD_GRID, model_name)
        val_sweeps[model_name] = sweep_df

        # Select threshold (validation only)
        sel = select_threshold(sweep_df, dr_target=SELECTION_DR_TARGET)
        selected[model_name] = sel
        log.info(
            "  Threshold selected: %.2f | Val DR: %.2f%% | Val FAR: %.2f%% | Rule: %s",
            sel["selected_threshold"],
            sel["validation_detection_rate"] * 100,
            sel["validation_far"] * 100,
            sel["selection_rule"],
        )

        # Pre-compute test scores — stored but NOT used yet for threshold selection
        # (scores are computed here for efficiency; no labels are examined)
        test_scores_dict[model_name] = pipe.predict_proba(X_test)[:, 1]

    log.info("=" * 60)
    log.info("PHASE 1 COMPLETE. Thresholds frozen for all models.")
    log.info("Frozen thresholds: %s",
             {m: sel["selected_threshold"] for m, sel in selected.items()})
    log.info("=" * 60)

    # -----------------------------------------------------------------------
    # Phase 2: Evaluate frozen thresholds on KDDTest+
    # Thresholds are now fixed; test labels are used for the first time.
    # -----------------------------------------------------------------------
    log.info("PHASE 2: Evaluating frozen thresholds on KDDTest+ …")
    test_metrics_dict: dict[str, dict] = {}
    for model_name in model_defs:
        sel_thr = selected[model_name]["selected_threshold"]
        tm = evaluate_on_test(y_test, test_scores_dict[model_name], sel_thr, model_name)
        test_metrics_dict[model_name] = tm
        log.info(
            "  %s @ theta=%.2f | Test DR: %.2f%% | Test FAR: %.2f%% | ROC-AUC: %.4f",
            model_name, sel_thr,
            tm["detection_rate"] * 100,
            tm["far"] * 100,
            tm["roc_auc"],
        )

    # -----------------------------------------------------------------------
    # Build result tables
    # -----------------------------------------------------------------------
    val_sweep_combined = build_validation_sweep_csv(val_sweeps)
    selected_df = build_selected_thresholds_csv(selected)
    comparison_df = build_comparison_table(selected, test_metrics_dict, exp002_results)

    # Save CSVs
    val_sweep_combined.to_csv(RESULTS_DIR / "validation_threshold_sweep.csv", index=False)
    val_sweep_combined.to_csv(TABLES_DIR / "validation_threshold_sweep.csv", index=False)

    selected_df.to_csv(RESULTS_DIR / "selected_thresholds.csv", index=False)
    selected_df.to_csv(TABLES_DIR / "selected_thresholds.csv", index=False)

    comparison_df.to_csv(RESULTS_DIR / "threshold_operating_point_results.csv", index=False)
    comparison_df.to_csv(TABLES_DIR / "threshold_operating_point_results.csv", index=False)

    log.info("Saved CSV tables.")

    # Save JSON metrics
    metrics_obj = {
        "experiment_id": "EXP-005",
        "timestamp": timestamp,
        "random_seed": seed,
        "environment": env_info,
        "train_file": {"path": str(train_path), "rows": len(train_df), "sha256": train_sha},
        "test_file": {"path": str(test_path), "rows": len(test_df), "sha256": test_sha},
        "threshold_grid": {"start": 0.00, "stop": 1.00, "step": 0.01, "n_thresholds": len(THRESHOLD_GRID)},
        "selection_rule": {
            "primary": f"highest threshold >= {SELECTION_DR_TARGET:.0%} validation DR",
            "fallback": "highest validation DR, lowest FAR on tie, highest threshold on tie",
            "data_used": "20% KDDTrain+ validation partition only",
        },
        "selected_thresholds": {m: sel["selected_threshold"] for m, sel in selected.items()},
        "validation_results_at_selected": {m: sel for m, sel in selected.items()},
        "test_results_at_selected": test_metrics_dict,
        "exp002_baseline_reference": {
            m: {
                "threshold": DEFAULT_THRESHOLD,
                "accuracy": exp002_results["results"][m]["full_test"]["accuracy"],
                "precision": exp002_results["results"][m]["full_test"]["precision"],
                "detection_rate": exp002_results["results"][m]["full_test"]["recall_detection_rate"],
                "far": exp002_results["results"][m]["full_test"]["false_alarm_rate"],
                "f1": exp002_results["results"][m]["full_test"]["f1_score"],
                "roc_auc": exp002_results["results"][m]["full_test"]["roc_auc"],
            }
            for m in model_defs
        },
    }
    (METRICS_DIR / "exp005_metrics.json").write_text(
        json.dumps(metrics_obj, indent=2, default=str), encoding="utf-8"
    )
    log.info("Saved metrics JSON.")

    # -----------------------------------------------------------------------
    # Generate figures
    # -----------------------------------------------------------------------
    log.info("Generating figures …")
    plot_threshold_vs_detection_far(val_sweeps, selected)
    plot_roc_curves(y_test, test_scores_dict, selected, test_metrics_dict)
    plot_default_vs_selected(comparison_df)

    # -----------------------------------------------------------------------
    # Generate report
    # -----------------------------------------------------------------------
    log.info("Generating report …")
    report_text = generate_experiment_report(
        env_info=env_info,
        train_info={"path": str(train_path), "rows": len(train_df), "sha256": train_sha},
        test_info={"path": str(test_path), "rows": len(test_df), "sha256": test_sha},
        selected_df=selected_df,
        comparison_df=comparison_df,
        val_sweeps=val_sweeps,
        timestamp=timestamp,
        seed=seed,
    )
    report_path = RESULTS_DIR / "experiment_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Report written to %s", report_path)

    log.info("=" * 60)
    log.info("EXP-005 complete. Results in: %s", RESULTS_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EXP-005: Threshold / Operating-Point Analysis — SentinelNet"
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN,
                        help="Path to KDDTrain+.txt")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST,
                        help="Path to KDDTest+.txt")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed (default: 42)")
    args = parser.parse_args()
    main(train_path=args.train, test_path=args.test, seed=args.seed)
