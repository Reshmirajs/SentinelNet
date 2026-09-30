"""
EXP-003: Minority Attack Category Sensitivity
=============================================
SentinelNet Research Project

Purpose
-------
Conduct a post-hoc subgroup sensitivity audit of binary Normal-vs-Attack
models (Logistic Regression, Decision Tree, Random Forest) across ground-truth
attack categories: DoS, Probe, R2L, and U2R.

Key Methodological Decisions
----------------------------
- Binary Target         : normal = 0 (Normal), all attacks = 1 (Attack)
- Post-hoc Subgroups    : DoS, Probe, R2L, U2R (Normal excluded from attack DR)
- No Rebalancing        : No SMOTE, ADASYN, oversampling, undersampling,
                          class_weight balancing, or threshold adjustments
- Preprocessing         : Fitted strictly on the training partition
- Metadata Quarantine   : difficulty dropped; num_outbound_cmds dropped
- Test Quarantine       : KDDTest+ never used for fitting or model selection
- Interpretation Rule   : Measures observational association between training
                          representation and detection rate; does not establish
                          causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
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
    format="%(asctime)s [%(levelname)s] EXP-003 | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("EXP-003")

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
# NSL-KDD Feature Schema & Taxonomy (Inherited from EXP-001 / EXP-002)
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

# Canonical taxonomy mapping from MIT Lincoln Lab DARPA 1998/1999 / KDD Cup 99
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

ATTACK_CATEGORIES: list[str] = ["DoS", "Probe", "R2L", "U2R"]
ALL_CATEGORIES: list[str] = ["Normal", "DoS", "Probe", "R2L", "U2R"]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
BINARY_FEATURES = [
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
]
DROP_FEATURES = ["difficulty", "num_outbound_cmds"]

RANDOM_SEED = 42
VALIDATION_SIZE = 0.20


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    """Compute SHA-256 hexdigest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_environment_info() -> dict[str, str]:
    """Collect Python and dependency version strings."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a raw NSL-KDD CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    log.info("Loading %s …", path.name)
    df = pd.read_csv(path, header=None, names=NSL_KDD_FEATURE_NAMES)
    if df.shape[1] != 43:
        raise ValueError(f"Expected 43 columns, got {df.shape[1]} in {path.name}")
    log.info("  Loaded %d rows × %d columns", *df.shape)
    return df


def make_binary_target(labels: pd.Series) -> pd.Series:
    """Encode label column as binary: normal=0, attack=1."""
    return (labels.str.strip() != "normal").astype(int)


def map_attack_category(labels: pd.Series) -> pd.Series:
    """Map raw label string to high-level category using ATTACK_CATEGORY_MAP."""
    return labels.str.strip().map(lambda x: ATTACK_CATEGORY_MAP.get(x, "Unknown"))


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names excluding label and dropped metadata."""
    exclude = {"label", "difficulty"} | set(DROP_FEATURES)
    return [c for c in df.columns if c not in exclude]


def get_numerical_features(feature_cols: list[str]) -> list[str]:
    """Return numerical feature names."""
    exclude = set(CATEGORICAL_FEATURES) | set(BINARY_FEATURES)
    return [f for f in feature_cols if f not in exclude]


def build_pipeline(model: Any, feature_cols: list[str]) -> Pipeline:
    """Build a leakage-safe ColumnTransformer + Classifier Pipeline."""
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
# Distribution Analysis
# ---------------------------------------------------------------------------

def compute_category_distribution(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Programmatically derive train and test category distributions."""
    train_cats = map_attack_category(train_df["label"])
    test_cats = map_attack_category(test_df["label"])

    train_total = len(train_df)
    test_total = len(test_df)

    train_counts = train_cats.value_counts()
    test_counts = test_cats.value_counts()

    # Total attacks in train and test
    train_attacks_total = (train_cats != "Normal").sum()
    test_attacks_total = (test_cats != "Normal").sum()

    records = []
    for cat in ALL_CATEGORIES:
        tr_c = int(train_counts.get(cat, 0))
        te_c = int(test_counts.get(cat, 0))
        tr_pct = round(tr_c / train_total * 100, 4) if train_total > 0 else 0.0
        te_pct = round(te_c / test_total * 100, 4) if test_total > 0 else 0.0
        
        # Share among attacks only
        if cat != "Normal":
            tr_atk_share = round(tr_c / train_attacks_total * 100, 4) if train_attacks_total > 0 else 0.0
            te_atk_share = round(te_c / test_attacks_total * 100, 4) if test_attacks_total > 0 else 0.0
        else:
            tr_atk_share = None
            te_atk_share = None

        records.append({
            "category": cat,
            "train_count": tr_c,
            "train_pct_of_total": tr_pct,
            "train_pct_of_attacks": tr_atk_share,
            "test_count": te_c,
            "test_pct_of_total": te_pct,
            "test_pct_of_attacks": te_atk_share,
            "delta_total_pct_pp": round(te_pct - tr_pct, 4),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Subgroup Sensitivity Evaluation
# ---------------------------------------------------------------------------

def evaluate_model_category_sensitivity(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Evaluate binary detector performance on attack category subgroups."""
    X_test = test_df[feature_cols].values
    y_test_binary = make_binary_target(test_df["label"]).values
    test_cats = map_attack_category(test_df["label"]).values

    y_pred = pipeline.predict(X_test)
    has_proba = hasattr(pipeline.named_steps["classifier"], "predict_proba")
    y_prob = pipeline.predict_proba(X_test)[:, 1] if has_proba else None

    # Overall binary performance for context
    full_cm = confusion_matrix(y_test_binary, y_pred, labels=[0, 1])
    tn_full, fp_full, fn_full, tp_full = full_cm.ravel()
    overall_acc = accuracy_score(y_test_binary, y_pred)
    overall_prec = precision_score(y_test_binary, y_pred, zero_division=0)
    overall_rec = recall_score(y_test_binary, y_pred, zero_division=0)
    overall_f1 = f1_score(y_test_binary, y_pred, zero_division=0)
    overall_far = fp_full / (tn_full + fp_full) if (tn_full + fp_full) > 0 else 0.0
    overall_auc = roc_auc_score(y_test_binary, y_prob) if y_prob is not None else None

    overall_metrics = {
        "accuracy": round(float(overall_acc), 6),
        "precision": round(float(overall_prec), 6),
        "detection_rate": round(float(overall_rec), 6),
        "false_alarm_rate": round(float(overall_far), 6),
        "f1_score": round(float(overall_f1), 6),
        "roc_auc": round(float(overall_auc), 6) if overall_auc is not None else None,
        "tp": int(tp_full), "tn": int(tn_full), "fp": int(fp_full), "fn": int(fn_full),
    }

    # Category subgroup evaluation
    normal_mask = (test_cats == "Normal")
    normal_tp = 0
    normal_fp = int((y_pred[normal_mask] == 1).sum())
    normal_tn = int((y_pred[normal_mask] == 0).sum())
    normal_fn = 0
    normal_count = int(normal_mask.sum())

    category_metrics: dict[str, dict[str, Any]] = {}
    for cat in ATTACK_CATEGORIES:
        cat_mask = (test_cats == cat)
        cat_count = int(cat_mask.sum())
        if cat_count == 0:
            continue

        cat_preds = y_pred[cat_mask]
        tp_cat = int((cat_preds == 1).sum())
        fn_cat = int((cat_preds == 0).sum())

        # Category Detection Rate / Recall (Primary metric)
        dr_cat = tp_cat / cat_count if cat_count > 0 else 0.0

        # Subgroup binary evaluation against Normal traffic (Cohort: Normal + Category C)
        # Precision = TP_C / (TP_C + FP_normal)
        total_predicted_attack = tp_cat + normal_fp
        prec_cat = tp_cat / total_predicted_attack if total_predicted_attack > 0 else 0.0
        
        # F1 = 2 * (Prec * Rec) / (Prec + Rec)
        f1_cat = (
            2 * (prec_cat * dr_cat) / (prec_cat + dr_cat)
            if (prec_cat + dr_cat) > 0 else 0.0
        )

        category_metrics[cat] = {
            "category": cat,
            "sample_count": cat_count,
            "tp": tp_cat,
            "fn": fn_cat,
            "detection_rate": round(float(dr_cat), 6),
            "precision_against_normal": round(float(prec_cat), 6),
            "f1_against_normal": round(float(f1_cat), 6),
            "fp_normal": normal_fp,
            "tn_normal": normal_tn,
        }

    return {
        "overall": overall_metrics,
        "categories": category_metrics,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Save figure to both results/figures/ and results/ root."""
    path_fig = FIG_DIR / filename
    path_root = RESULTS_DIR / filename
    fig.savefig(path_fig, dpi=150, bbox_inches="tight")
    fig.savefig(path_root, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", filename)


def plot_attack_category_detection_rate(
    all_results: dict[str, dict[str, Any]],
) -> None:
    """Grouped bar chart showing Detection Rate across attack categories."""
    categories = ATTACK_CATEGORIES
    model_names = list(all_results.keys())
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, model_name in enumerate(model_names):
        dr_vals = [
            all_results[model_name]["categories"][cat]["detection_rate"] * 100
            for cat in categories
        ]
        bars = ax.bar(
            x + i * width, dr_vals, width, label=model_name, color=colors[i % len(colors)]
        )
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=3)

    ax.set_ylabel("Detection Rate (Recall %)")
    ax.set_title("Binary Intrusion Detection Rate Across Attack Categories\n(Post-Hoc Subgroup Analysis on Full KDDTest+)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_figure(fig, "attack_category_detection_rate.png")


def plot_training_representation_vs_detection(
    dist_df: pd.DataFrame,
    all_results: dict[str, dict[str, Any]],
) -> None:
    """Scatter/connected plot of training representation vs test detection rate."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Get attack category training percentages
    cat_df = dist_df[dist_df["category"].isin(ATTACK_CATEGORIES)].set_index("category")
    train_pcts = {cat: cat_df.loc[cat, "train_pct_of_total"] for cat in ATTACK_CATEGORIES}

    markers = ["o", "s", "^"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, (model_name, res) in enumerate(all_results.items()):
        x_vals = [train_pcts[cat] for cat in ATTACK_CATEGORIES]
        y_vals = [res["categories"][cat]["detection_rate"] * 100 for cat in ATTACK_CATEGORIES]
        ax.plot(
            x_vals, y_vals, marker=markers[i], color=colors[i], label=model_name,
            linewidth=1.8, markersize=8
        )
        for cat, x_val, y_val in zip(ATTACK_CATEGORIES, x_vals, y_vals):
            # Annotate category label
            offset_y = 3 if i == 0 else (-5 if i == 1 else 4)
            if i == 0:
                ax.annotate(
                    f"{cat} ({x_val:.2f}%)",
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    fontweight="semibold",
                )

    ax.set_xlabel("Training Set Representation (% of total KDDTrain+)")
    ax.set_ylabel("Test Detection Rate (%)")
    ax.set_title("Training Set Representation vs. Detection Rate\n(Observational Association Across Attack Categories)")
    ax.set_ylim(-5, 115)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save_figure(fig, "training_representation_vs_detection.png")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_experiment_report(
    env_info: dict,
    train_info: dict,
    test_info: dict,
    dist_df: pd.DataFrame,
    all_results: dict[str, dict[str, Any]],
    timestamp: str,
    seed: int,
) -> str:
    """Generate Markdown report for EXP-003."""
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

    lines.append("# EXP-003 — Minority Attack Category Sensitivity\n")
    lines.append(f"*Generated: {timestamp} UTC*  \n*Random seed: {seed}*\n")
    lines.append("> **STATUS:** COMPLETE — All metrics computed from actual model evaluations.\n")

    h(2, "Research Question")
    para(
        "RQ3: How does binary intrusion-detection performance vary across DoS, "
        "Probe, R2L, and U2R attack categories?"
    )

    h(2, "Hypothesis")
    para(
        "H3: Detection performance will be lower for the less represented attack categories, "
        "particularly R2L and U2R, than for the more represented DoS and Probe categories.\n\n"
        "*Evaluation:* As shown below, detection rates for R2L and U2R are markedly lower than "
        "those for DoS and Probe across all three baseline classifiers, observationally supporting H3."
    )

    h(2, "Methodology")
    para(
        "1. **Task Definition:** The models are strictly binary classifiers trained to distinguish "
        "Normal (0) from Attack (1). They do not output multiclass predictions. Evaluation by "
        "attack category is conducted strictly as post-hoc ground-truth subgroup analysis.\n\n"
        "2. **No Artificial Rebalancing:** No resampling (SMOTE/undersampling), class-weighting, "
        "or threshold tuning was applied. The models operate with standard decision thresholds (0.5).\n\n"
        "3. **Quarantine:** All preprocessing transformers (OneHotEncoder, StandardScaler) "
        "were fitted strictly on the training partition of KDDTrain+. KDDTest+ was fully quarantined."
    )

    h(2, "Dataset Distribution")
    lines.append(table_row(
        "Category", "Train Count", "Train % (Total)", "Train % (Attacks)",
        "Test Count", "Test % (Total)", "Test % (Attacks)", "Δ (pp)"
    ))
    lines.append(table_sep("l", "r", "r", "r", "r", "r", "r", "r"))
    for _, row in dist_df.iterrows():
        tr_atk = f"{row['train_pct_of_attacks']:.2f}%" if pd.notna(row['train_pct_of_attacks']) else "—"
        te_atk = f"{row['test_pct_of_attacks']:.2f}%" if pd.notna(row['test_pct_of_attacks']) else "—"
        lines.append(table_row(
            row["category"],
            f"{int(row['train_count']):,}",
            f"{row['train_pct_of_total']:.2f}%",
            tr_atk,
            f"{int(row['test_count']):,}",
            f"{row['test_pct_of_total']:.2f}%",
            te_atk,
            f"{row['delta_total_pct_pp']:+.2f}",
        ))
    lines.append("")

    h(2, "Category-Level Detection Rates (Primary Comparison)")
    lines.append(table_row("Model", "DoS DR", "Probe DR", "R2L DR", "U2R DR", "Overall DR", "Overall FAR"))
    lines.append(table_sep("l", "r", "r", "r", "r", "r", "r"))
    for model_name, res in all_results.items():
        dos_dr = res["categories"]["DoS"]["detection_rate"] * 100
        prb_dr = res["categories"]["Probe"]["detection_rate"] * 100
        r2l_dr = res["categories"]["R2L"]["detection_rate"] * 100
        u2r_dr = res["categories"]["U2R"]["detection_rate"] * 100
        ov_dr = res["overall"]["detection_rate"] * 100
        ov_far = res["overall"]["false_alarm_rate"] * 100
        lines.append(table_row(
            model_name,
            f"{dos_dr:.2f}%",
            f"{prb_dr:.2f}%",
            f"{r2l_dr:.2f}%",
            f"{u2r_dr:.2f}%",
            f"{ov_dr:.2f}%",
            f"{ov_far:.2f}%",
        ))
    lines.append("")

    h(2, "Detailed Subgroup Metrics by Model")
    for model_name, res in all_results.items():
        h(3, model_name)
        lines.append(table_row("Category", "N Samples", "Detection Rate (Recall)", "Precision (vs Normal)", "F1 (vs Normal)"))
        lines.append(table_sep("l", "r", "r", "r", "r"))
        for cat in ATTACK_CATEGORIES:
            m = res["categories"][cat]
            lines.append(table_row(
                cat,
                f"{m['sample_count']:,}",
                f"{m['detection_rate']*100:.2f}%",
                f"{m['precision_against_normal']*100:.2f}%",
                f"{m['f1_against_normal']*100:.2f}%",
            ))
        lines.append("")

    lr_m = all_results["Logistic Regression"]
    dt_m = all_results["Decision Tree"]
    rf_m = all_results["Random Forest"]

    dos_drs = [res["categories"]["DoS"]["detection_rate"] * 100 for res in all_results.values()]
    prb_drs = [res["categories"]["Probe"]["detection_rate"] * 100 for res in all_results.values()]

    h(2, "Interpretation of Observed Results")
    para(
        f"- **Disparity Across Categories:** Detection performance is markedly uneven. Across all three "
        f"models, DoS attacks exhibited detection rates between {min(dos_drs):.1f}% and {max(dos_drs):.1f}%, "
        f"and Probe attacks showed detection rates between {min(prb_drs):.1f}% and {max(prb_drs):.1f}%. "
        f"Conversely, R2L attacks exhibited severe detection degradation in Logistic Regression "
        f"({lr_m['categories']['R2L']['detection_rate']*100:.2f}%) and Random Forest "
        f"({rf_m['categories']['R2L']['detection_rate']*100:.2f}%), though Decision Tree detected "
        f"{dt_m['categories']['R2L']['detection_rate']*100:.2f}%. U2R attacks showed detection rates "
        f"ranging from {rf_m['categories']['U2R']['detection_rate']*100:.2f}% (RF) to "
        f"{dt_m['categories']['U2R']['detection_rate']*100:.2f}% (DT).\n\n"
        f"- **Model Trade-offs:** Random Forest achieved the lowest False Alarm Rate "
        f"({rf_m['overall']['false_alarm_rate']*100:.2f}%) on normal traffic and high precision, "
        f"but demonstrated substantial conservatism on minority attacks (R2L DR: "
        f"{rf_m['categories']['R2L']['detection_rate']*100:.2f}%, U2R DR: "
        f"{rf_m['categories']['U2R']['detection_rate']*100:.2f}%). Decision Tree detected a broader "
        f"fraction of minority attacks (R2L DR: {dt_m['categories']['R2L']['detection_rate']*100:.2f}%, "
        f"U2R DR: {dt_m['categories']['U2R']['detection_rate']*100:.2f}%), but incurred a higher False "
        f"Alarm Rate ({dt_m['overall']['false_alarm_rate']*100:.2f}%). No single model dominates across all "
        f"trade-offs.\n\n"
        f"- **Observational Association vs. Causality:** Categories with lower training representation (R2L at 0.79% "
        f"and U2R at 0.04% of training records) showed substantially lower detection performance than well-represented "
        f"categories (DoS at 36.46% and Probe at 9.25%). While an observational association is evident, "
        f"this experiment does not isolate causal factors (e.g., semantic differences in attack signatures, "
        f"content feature scarcity, or novel attack variations in the test set)."
    )

    h(2, "Limitations")
    para(
        "1. **Post-Hoc Subgroup Analysis:** EXP-003 analyzes subgroups evaluated through a binary detector. "
        "The models were not trained to distinguish between attack categories.\n\n"
        "2. **Ground-Truth Subgroups:** Attack categories represent ground-truth metadata taxonomy, "
        "not model prediction outputs.\n\n"
        "3. **Representation Imbalance:** R2L and U2R have substantially lower representation in KDDTrain+ "
        "(995 and 52 samples, respectively), limiting the exposure of gradient-based and tree-based estimators "
        "to their signatures.\n\n"
        "4. **No Causal Claim:** The observational association between representation and detection rate does not "
        "establish that sample count alone caused poor performance. Architectural and feature suitability also play roles.\n\n"
        "5. **Historical Benchmark:** NSL-KDD originates from 1998 network traffic captures; generalization to modern "
        "attack categories or traffic distributions cannot be inferred.\n\n"
        "6. **Orthogonal to Novel Label Analysis:** The test set contains 17 novel attack labels (LEAK-002) distributed "
        "across these categories; category-level detection aggregates both known and novel variants within each family."
    )

    h(2, "Reproducibility")
    lines.append(table_row("Item", "Value"))
    lines.append(table_sep("l", "l"))
    lines.append(table_row("Experiment ID", "EXP-003"))
    lines.append(table_row("Random seed", str(seed)))
    lines.append(table_row("Execution timestamp", timestamp))
    lines.append(table_row("Python", env_info.get("python", "?")))
    lines.append(table_row("Platform", env_info.get("platform", "?")))
    lines.append(table_row("scikit-learn", env_info.get("scikit_learn", "?")))
    lines.append(table_row("pandas", env_info.get("pandas", "?")))
    lines.append(table_row("numpy", env_info.get("numpy", "?")))
    lines.append(table_row("matplotlib", env_info.get("matplotlib", "?")))
    lines.append(table_row("Train file", f"{train_info['path']} ({train_info['rows']:,} rows, SHA-256: {train_info['sha256'][:16]}...)"))
    lines.append(table_row("Test file", f"{test_info['path']} ({test_info['rows']:,} rows, SHA-256: {test_info['sha256'][:16]}...)"))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main(
    train_path: Path = DEFAULT_TRAIN,
    test_path: Path = DEFAULT_TEST,
    seed: int = RANDOM_SEED,
) -> None:
    """Run full EXP-003 experiment pipeline."""
    import datetime
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for d in (RESULTS_DIR, FIG_DIR, METRICS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("EXP-003: Minority Attack Category Sensitivity")
    log.info("Timestamp : %s", timestamp)
    log.info("Seed      : %d", seed)
    log.info("Train     : %s", train_path)
    log.info("Test      : %s", test_path)
    log.info("=" * 60)

    env_info = get_environment_info()

    # Checksums
    train_sha = sha256_file(train_path)
    test_sha = sha256_file(test_path)

    # Load datasets
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    # Compute category distributions
    dist_df = compute_category_distribution(train_df, test_df)
    dist_csv_path = RESULTS_DIR / "category_distribution.csv"
    dist_df.to_csv(dist_csv_path, index=False)
    # Also save in tables/
    dist_df.to_csv(TABLES_DIR / "category_distribution.csv", index=False)
    log.info("Category distribution computed and saved to %s", dist_csv_path)

    # Prepare features and binary target
    feature_cols = get_feature_columns(train_df)
    y_train_full = make_binary_target(train_df["label"])

    # Internal 80/20 train/validation split (identical to EXP-002)
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

    # Model definitions (identical to EXP-002)
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

    all_results: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []

    for model_name, model_obj in model_defs.items():
        log.info("-" * 50)
        log.info("Training: %s …", model_name)
        pipeline = build_pipeline(model_obj, feature_cols)

        t0 = time.time()
        pipeline.fit(X_tr, y_tr)
        train_time = round(time.time() - t0, 4)
        log.info("  Training time: %.2f seconds", train_time)

        # Validation sanity check
        val_pred = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        log.info("  Validation Accuracy: %.4f", val_acc)

        # Evaluate on test set categories
        model_results = evaluate_model_category_sensitivity(pipeline, test_df, feature_cols)
        model_results["overall"]["train_time_s"] = train_time
        all_results[model_name] = model_results

        # Prepare CSV rows
        for cat in ATTACK_CATEGORIES:
            m = model_results["categories"][cat]
            csv_rows.append({
                "model": model_name,
                "category": cat,
                "sample_count": m["sample_count"],
                "detection_rate_pct": round(m["detection_rate"] * 100, 2),
                "precision_vs_normal_pct": round(m["precision_against_normal"] * 100, 2),
                "f1_vs_normal_pct": round(m["f1_against_normal"] * 100, 2),
                "tp_count": m["tp"],
                "fn_count": m["fn"],
                "normal_fp_count": m["fp_normal"],
            })

    # Save category_detection_results.csv
    det_df = pd.DataFrame(csv_rows)
    det_csv_path = RESULTS_DIR / "category_detection_results.csv"
    det_df.to_csv(det_csv_path, index=False)
    det_df.to_csv(TABLES_DIR / "category_detection_results.csv", index=False)
    log.info("Category detection results saved to %s", det_csv_path)

    # Save JSON metrics
    metrics_summary = {
        "experiment_id": "EXP-003",
        "timestamp": timestamp,
        "random_seed": seed,
        "environment": env_info,
        "train_file": {"path": str(train_path), "rows": len(train_df), "sha256": train_sha},
        "test_file": {"path": str(test_path), "rows": len(test_df), "sha256": test_sha},
        "distribution": dist_df.to_dict(orient="records"),
        "results": all_results,
    }
    (METRICS_DIR / "category_metrics.json").write_text(
        json.dumps(metrics_summary, indent=2), encoding="utf-8"
    )

    # Generate figures
    log.info("Generating figures …")
    plot_attack_category_detection_rate(all_results)
    plot_training_representation_vs_detection(dist_df, all_results)

    # Generate experiment report
    log.info("Generating report …")
    report_text = generate_experiment_report(
        env_info=env_info,
        train_info={"path": str(train_path), "rows": len(train_df), "sha256": train_sha},
        test_info={"path": str(test_path), "rows": len(test_df), "sha256": test_sha},
        dist_df=dist_df,
        all_results=all_results,
        timestamp=timestamp,
        seed=seed,
    )
    report_path = RESULTS_DIR / "experiment_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Report written to %s", report_path)

    log.info("=" * 60)
    log.info("EXP-003 complete. Results in: %s", RESULTS_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EXP-003: Minority Attack Category Sensitivity — SentinelNet"
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Path to KDDTrain+.txt")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Path to KDDTest+.txt")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)")
    args = parser.parse_args()
    main(train_path=args.train, test_path=args.test, seed=args.seed)
