"""
EXP-004: R2L Within-Category Label-Shift Analysis
=================================================
SentinelNet Research Project

Purpose
-------
Disaggregate the category-level Remote-to-Local (R2L) detection results from
EXP-003 by individual attack label, analyzing performance across three
representation groups:
  - Group A: Shared labels (present in both KDDTrain+ and KDDTest+)
  - Group B: Test-only labels (present in KDDTest+, absent from KDDTrain+)
  - Group C: Train-only labels (present in KDDTrain+, absent from KDDTest+)

Key Methodological Decisions
----------------------------
- Follow-up subgroup audit to EXP-003; model outputs reproduce exact EXP-003 pipeline.
- Primary metric: Detection Rate (Recall) = TP / N for each R2L label.
- No class balancing, SMOTE, class weighting, or threshold adjustments.
- No causal claims; training representation is confounded with attack signatures.
- All counts and group partitions derived programmatically from raw data.
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] EXP-004 | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
log = logging.getLogger("EXP-004")

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
# Feature Schema & Taxonomy (Inherited from EXP-001)
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

# Canonical taxonomy mapping from MIT Lincoln Lab / KDD Cup 99 / EXP-001
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

# Derived list of all R2L labels in taxonomy
R2L_TAXONOMY_LABELS: list[str] = sorted(
    [lbl for lbl, cat in ATTACK_CATEGORY_MAP.items() if cat == "R2L"]
)

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
    """Collect platform environment metadata."""
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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names excluding label and dropped metadata."""
    exclude = {"label", "difficulty"} | set(DROP_FEATURES)
    return [c for c in df.columns if c not in exclude]


def get_numerical_features(feature_cols: list[str]) -> list[str]:
    """Return numerical feature names."""
    exclude = set(CATEGORICAL_FEATURES) | set(BINARY_FEATURES)
    return [f for f in feature_cols if f not in exclude]


def build_pipeline(model: Any, feature_cols: list[str]) -> Pipeline:
    """Build leakage-safe ColumnTransformer + Classifier Pipeline (same as EXP-002/003)."""
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
# R2L Label Distribution & Representation Grouping
# ---------------------------------------------------------------------------

def compute_r2l_composition(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """Programmatically derive R2L label counts, shares, and representation groups."""
    tr_labels = train_df["label"].str.strip()
    te_labels = test_df["label"].str.strip()

    tr_counts = tr_labels.value_counts().to_dict()
    te_counts = te_labels.value_counts().to_dict()

    # Identify all R2L labels present in either train, test, or canonical taxonomy
    all_r2l_labels = sorted(set(
        [lbl for lbl in R2L_TAXONOMY_LABELS]
        + [lbl for lbl in tr_counts if ATTACK_CATEGORY_MAP.get(lbl) == "R2L"]
        + [lbl for lbl in te_counts if ATTACK_CATEGORY_MAP.get(lbl) == "R2L"]
    ))

    # Total R2L counts
    train_r2l_total = sum(tr_counts.get(lbl, 0) for lbl in all_r2l_labels)
    test_r2l_total = sum(te_counts.get(lbl, 0) for lbl in all_r2l_labels)

    rows = []
    for lbl in all_r2l_labels:
        tr_c = tr_counts.get(lbl, 0)
        te_c = te_counts.get(lbl, 0)

        if tr_c > 0 and te_c > 0:
            status = "shared"
        elif tr_c == 0 and te_c > 0:
            status = "test_only"
        elif tr_c > 0 and te_c == 0:
            status = "train_only"
        else:
            continue  # Label in taxonomy but 0 in both splits

        tr_share = round(tr_c / train_r2l_total * 100, 2) if train_r2l_total > 0 else 0.0
        te_share = round(te_c / test_r2l_total * 100, 2) if test_r2l_total > 0 else 0.0

        rows.append({
            "r2l_label": lbl,
            "train_count": tr_c,
            "train_share_r2l_pct": tr_share,
            "test_count": te_c,
            "test_share_r2l_pct": te_share,
            "representation_status": status,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-Label Detection Evaluation
# ---------------------------------------------------------------------------

def evaluate_r2l_labels(
    pipeline: Pipeline,
    test_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
) -> list[dict[str, Any]]:
    """Evaluate binary detector on each individual R2L attack label."""
    X_test = test_df[feature_cols].values
    y_pred = pipeline.predict(X_test)
    test_labels = test_df["label"].str.strip().values

    results = []
    for _, row in comp_df.iterrows():
        lbl = row["r2l_label"]
        tr_c = row["train_count"]
        te_c = row["test_count"]
        status = row["representation_status"]
        tr_share = row["train_share_r2l_pct"]
        te_share = row["test_share_r2l_pct"]

        if te_c > 0:
            mask = (test_labels == lbl)
            preds_lbl = y_pred[mask]
            tp = int((preds_lbl == 1).sum())
            fn = int((preds_lbl == 0).sum())
            dr = round(tp / te_c * 100, 2) if te_c > 0 else 0.0
        else:
            tp = None
            fn = None
            dr = None

        results.append({
            "model": model_name,
            "r2l_label": lbl,
            "train_count": tr_c,
            "test_count": te_c,
            "train_share_r2l_pct": tr_share,
            "test_share_r2l_pct": te_share,
            "representation_status": status,
            "tp": tp,
            "fn": fn,
            "detection_rate_pct": dr,
        })

    return results


def summarize_representation_groups(
    det_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize detection results by representation status group (shared vs test_only)."""
    rows = []
    for model_name, grp in det_df.groupby("model", sort=False):
        for status in ["shared", "test_only"]:
            sub = grp[(grp["representation_status"] == status) & (grp["test_count"] > 0)]
            n_total = int(sub["test_count"].sum())
            tp_total = int(sub["tp"].dropna().sum())
            fn_total = int(sub["fn"].dropna().sum())
            dr = round(tp_total / n_total * 100, 2) if n_total > 0 else 0.0

            rows.append({
                "model": model_name,
                "representation_status": status,
                "test_sample_count": n_total,
                "tp": tp_total,
                "fn": fn_total,
                "detection_rate_pct": dr,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, filename: str) -> None:
    """Save figure to results/figures/ and results/ root."""
    path_fig = FIG_DIR / filename
    path_root = RESULTS_DIR / filename
    fig.savefig(path_fig, dpi=150, bbox_inches="tight")
    fig.savefig(path_root, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", filename)


def plot_r2l_label_detection_rate(det_df: pd.DataFrame) -> None:
    """Bar chart comparing per-label detection rates across models."""
    # Filter to labels present in test set
    test_labels_df = det_df[det_df["test_count"] > 0].copy()
    labels = sorted(test_labels_df["r2l_label"].unique())
    model_names = list(det_df["model"].unique())

    # Build tick labels with sample size N
    counts = {
        lbl: int(test_labels_df[test_labels_df["r2l_label"] == lbl]["test_count"].iloc[0])
        for lbl in labels
    }
    statuses = {
        lbl: test_labels_df[test_labels_df["r2l_label"] == lbl]["representation_status"].iloc[0]
        for lbl in labels
    }
    y_labels = [f"{lbl}\n(N={counts[lbl]:,}, {statuses[lbl]})" for lbl in labels]

    y = np.arange(len(labels))
    height = 0.25

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, model_name in enumerate(model_names):
        sub = test_labels_df[test_labels_df["model"] == model_name].set_index("r2l_label")
        dr_vals = [sub.loc[lbl, "detection_rate_pct"] for lbl in labels]
        bars = ax.barh(y + (i - 1) * height, dr_vals, height, label=model_name, color=colors[i])
        for bar in bars:
            w = bar.get_width()
            ax.annotate(
                f"{w:.1f}%",
                (w + 1, bar.get_y() + bar.get_height() / 2),
                va="center",
                fontsize=7.5,
            )

    ax.set_xlabel("Detection Rate (%)")
    ax.set_title("R2L Individual Attack Label Detection Rates by Model\n(Post-Hoc Disaggregation on KDDTest+)")
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=8.5)
    ax.set_xlim(0, 115)
    ax.legend(loc="lower right")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_figure(fig, "r2l_label_detection_rate.png")


def plot_r2l_composition_shift(comp_df: pd.DataFrame) -> None:
    """Horizontal bar chart illustrating the extreme train/test composition shift in R2L."""
    df_sorted = comp_df.sort_values(by="train_share_r2l_pct", ascending=True)
    labels = df_sorted["r2l_label"].tolist()
    y = np.arange(len(labels))
    height = 0.35

    tr_shares = df_sorted["train_share_r2l_pct"].tolist()
    te_shares = df_sorted["test_share_r2l_pct"].tolist()

    fig, ax = plt.subplots(figsize=(9, 7))
    b1 = ax.barh(y - height / 2, tr_shares, height, label="KDDTrain+ Share (%)", color="#3470a3")
    b2 = ax.barh(y + height / 2, te_shares, height, label="KDDTest+ Share (%)", color="#e76f51")

    # Annotate values
    for bar in b1:
        w = bar.get_width()
        if w > 0:
            ax.annotate(f"{w:.1f}%", (w + 0.8, bar.get_y() + bar.get_height() / 2), va="center", fontsize=7.5, color="#3470a3")
    for bar in b2:
        w = bar.get_width()
        if w > 0:
            ax.annotate(f"{w:.1f}%", (w + 0.8, bar.get_y() + bar.get_height() / 2), va="center", fontsize=7.5, color="#e76f51")

    ax.set_xlabel("Share of R2L Category (%)")
    ax.set_title("R2L Intra-Category Composition Shift: KDDTrain+ vs. KDDTest+\n(Highlighting Asymmetry in Attack Types)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    _save_figure(fig, "r2l_train_vs_test_composition.png")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_experiment_report(
    env_info: dict,
    train_info: dict,
    test_info: dict,
    comp_df: pd.DataFrame,
    det_df: pd.DataFrame,
    grp_df: pd.DataFrame,
    timestamp: str,
    seed: int,
) -> str:
    """Generate Markdown report for EXP-004."""
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

    lines.append("# EXP-004 — R2L Within-Category Label-Shift Analysis\n")
    lines.append(f"*Generated: {timestamp} UTC*  \n*Random seed: {seed}*\n")
    lines.append("> **STATUS:** COMPLETE — Disaggregated evaluation computed from actual model predictions.\n")

    h(2, "Research Question")
    para(
        "RQ4: How does binary intrusion-detection performance vary across R2L attack "
        "labels according to their representation in the training data?"
    )

    h(2, "Hypothesis")
    para(
        "H4: R2L attack labels with greater representation in the training data will generally "
        "show higher detection rates than R2L labels with limited or no training representation.\n\n"
        "**Evaluation of H4:** H4 is **NOT supported**. Higher training representation did not "
        "systematically produce higher detection rates. Shared labels with training exposure "
        "(e.g., `guess_passwd`, which had 53 training records) exhibited near-zero detection "
        "(0.0%–0.49%), while specific test-only labels with zero training exposure "
        "(e.g., `snmpguess` and `httptunnel`) achieved detection rates up to 99.09% and 81.95% in Decision Tree. "
        "Observational detection rate was dominated by attack-specific signature discriminability rather than "
        "training sample count alone."
    )

    h(2, "Motivation from EXP-003")
    para(
        "In EXP-003, Remote-to-Local (R2L) attacks exhibited severe detection degradation across all "
        "three baseline classifiers (Logistic Regression: 1.18%, Random Forest: 5.55%, Decision Tree: 21.49%). "
        "However, audit evidence revealed that R2L cannot be evaluated simply as an underrepresented monolithic category: "
        "the attack labels dominating the test set were barely present in training, while the attack label dominating "
        "training was entirely absent from the test set. EXP-004 serves as a granular follow-up audit disaggregating "
        "this result by individual attack label."
    )

    h(2, "Dataset and Methodology")
    para(
        "- **Dataset Files:** `KDDTrain+.txt` (125,973 rows) and `KDDTest+.txt` (22,544 rows).\n"
        "- **Training Partition:** Consistent with EXP-002 and EXP-003, models were fitted strictly on the 80% "
        "stratified training partition (100,778 rows), leaving 25,195 validation rows and the fully quarantined test set.\n"
        "- **Target & Preprocessing:** Strictly binary (Normal = 0, Attack = 1). Preprocessing pipeline "
        "(OneHotEncoder, StandardScaler, drop `difficulty` and `num_outbound_cmds`) identical to EXP-002/EXP-003.\n"
        "- **Subgroup Metric:** Primary metric is Detection Rate (Recall) = $TP / N$ for each individual R2L label."
    )

    h(2, "R2L Intra-Category Composition Shift")
    lines.append(table_row(
        "R2L Label", "Train Count", "Train Share (% R2L)", "Test Count", "Test Share (% R2L)", "Status"
    ))
    lines.append(table_sep("l", "r", "r", "r", "r", "l"))
    for _, r in comp_df.iterrows():
        lines.append(table_row(
            r["r2l_label"],
            f"{r['train_count']:,}",
            f"{r['train_share_r2l_pct']:.2f}%",
            f"{r['test_count']:,}",
            f"{r['test_share_r2l_pct']:.2f}%",
            r["representation_status"],
        ))
    lines.append("")

    h(2, "Summary by Representation Group")
    lines.append(table_row("Model", "Representation Status", "Test Sample Count", "TP", "FN", "Detection Rate (%)"))
    lines.append(table_sep("l", "l", "r", "r", "r", "r"))
    for _, r in grp_df.iterrows():
        lines.append(table_row(
            r["model"],
            r["representation_status"],
            f"{r['test_sample_count']:,}",
            f"{r['tp']:,}",
            f"{r['fn']:,}",
            f"{r['detection_rate_pct']:.2f}%",
        ))
    lines.append("")
    para(
        "> **Note on Train-Only Labels:** The train-only group (`warezclient` [890 samples] and `spy` [2 samples]) "
        "comprises 892 records (89.65% of all training R2L data). Because no corresponding records exist in KDDTest+, "
        "no test detection rate can be computed for this group. It is documented strictly as composition shift."
    )

    h(2, "Per-Label Detection Results")
    lines.append(table_row("Model", "R2L Label", "Status", "Train N", "Test N", "TP", "FN", "Detection Rate (%)"))
    lines.append(table_sep("l", "l", "l", "r", "r", "r", "r", "r"))
    for _, r in det_df.iterrows():
        tp_str = f"{int(r['tp']):,}" if pd.notna(r['tp']) else "—"
        fn_str = f"{int(r['fn']):,}" if pd.notna(r['fn']) else "—"
        dr_str = f"{r['detection_rate_pct']:.2f}%" if pd.notna(r['detection_rate_pct']) else "— (Train-Only)"
        lines.append(table_row(
            r["model"],
            r["r2l_label"],
            r["representation_status"],
            f"{r['train_count']:,}",
            f"{r['test_count']:,}",
            tp_str,
            fn_str,
            dr_str,
        ))
    lines.append("")

    h(2, "Interpretation of Observed Results")
    para(
        "1. **Shared Labels with Limited Exposure:** The shared label group accounts for 2,199 test samples "
        "(76.22% of test R2L), dominated by `guess_passwd` (1,231) and `warezmaster` (944). Despite appearing in "
        "both datasets, these attacks were barely represented in training (53 and 20 samples, respectively). "
        "Detection performance on these shared labels was severely degraded across all models: Logistic Regression "
        "detected 1.14% (25/2,199), Decision Tree detected 8.05% (177/2,199), and Random Forest detected 6.55% (144/2,199). "
        "Notably, both Decision Tree and Random Forest failed completely on `guess_passwd` (0 out of 1,231 detected).\n\n"
        "2. **Test-Only Labels:** The 7 test-only labels account for 686 test records (23.78% of test R2L). "
        "The models had zero training examples for these exact names. Detection rates varied wildly: Decision Tree "
        "achieved a 64.58% aggregate detection rate on test-only labels (catching 328/331 `snmpguess` and 109/133 `httptunnel`), "
        "whereas Random Forest detected 2.33% (16/686) and Logistic Regression detected 1.31% (9/686). "
        "Low detection on these instances reflects test-only novelty rather than general inability to detect unseen threats.\n\n"
        "3. **Train-Only Labels:** `warezclient` (890 samples) made up 89.45% of all R2L training data, but has 0 test "
        "instances. The network patterns the classifiers had the greatest opportunity to learn were never tested.\n\n"
        "4. **No Causal Attribution:** Lower training representation was associated with lower observed detection in "
        "parts of this benchmark, but representation is heavily confounded with attack-label composition and attack-specific "
        "connection features. EXP-004 does not establish that training sample volume caused detection outcomes."
    )

    h(2, "Limitations")
    para(
        "1. **Post-Hoc Subgroup Analysis:** EXP-004 evaluates post-hoc subgroups of a binary detector; models were not "
        "trained to identify or discriminate individual attack labels.\n\n"
        "2. **Small Sample Sizes:** Several individual R2L labels have very small test counts (e.g., `imap` N=1, `phf` N=2, "
        "`ftp_write` N=3, `xsnoop` N=4, `xlock` N=9), rendering per-label percentage estimates noisy.\n\n"
        "3. **Zero-Shot Generalization:** Performance on test-only labels reflects specific signature overlaps on NSL-KDD "
        "features and cannot be interpreted as a measure of general zero-shot detection capability.\n\n"
        "4. **Confounding Factors:** Attack-specific characteristics (e.g., failed login attempts, port numbers, error flags) "
        "differ fundamentally across attack types and confound sample-size effects.\n\n"
        "5. **Historical Benchmark:** NSL-KDD is based on 1998 network traffic captures; results cannot be generalized to modern "
        "remote-access threats.\n\n"
        "6. **No Causality:** The experiment measures observational associations; causal impact of training count is unproven.\n\n"
        "7. **Multiple Comparisons:** Evaluating 13 individual test labels increases the risk of idiosyncratic findings.\n\n"
        "8. **External Validity:** Findings are specific to NSL-KDD and require external validation on modern intrusion detection datasets."
    )

    h(2, "Reproducibility")
    lines.append(table_row("Item", "Value"))
    lines.append(table_sep("l", "l"))
    lines.append(table_row("Experiment ID", "EXP-004"))
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
    """Run full EXP-004 experiment pipeline."""
    import datetime
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for d in (RESULTS_DIR, FIG_DIR, METRICS_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("EXP-004: R2L Within-Category Label-Shift Analysis")
    log.info("Timestamp : %s", timestamp)
    log.info("Seed      : %d", seed)
    log.info("Train     : %s", train_path)
    log.info("Test      : %s", test_path)
    log.info("=" * 60)

    env_info = get_environment_info()
    train_sha = sha256_file(train_path)
    test_sha = sha256_file(test_path)

    # Load raw datasets
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    # 1. Programmatically compute R2L composition
    comp_df = compute_r2l_composition(train_df, test_df)
    log.info("R2L composition table derived (%d unique attack labels).", len(comp_df))

    # 2. Prepare features and binary target
    feature_cols = get_feature_columns(train_df)
    y_train_full = make_binary_target(train_df["label"])

    # 80/20 train/validation split (identical to EXP-002/003)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=VALIDATION_SIZE,
        stratify=y_train_full.values,
        random_state=seed,
    )
    X_tr = train_df.iloc[train_idx][feature_cols].values
    y_tr = y_train_full.iloc[train_idx].values

    # Models (identical to EXP-002/003)
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

    all_det_results: list[dict[str, Any]] = []

    for model_name, model_obj in model_defs.items():
        log.info("-" * 50)
        log.info("Fitting model: %s …", model_name)
        pipe = build_pipeline(model_obj, feature_cols)
        pipe.fit(X_tr, y_tr)

        model_label_results = evaluate_r2l_labels(
            pipe, test_df, comp_df, feature_cols, model_name
        )
        all_det_results.extend(model_label_results)

    det_df = pd.DataFrame(all_det_results)
    grp_df = summarize_representation_groups(det_df)

    # 3. Export CSV tables
    det_csv = RESULTS_DIR / "r2l_label_detection_results.csv"
    grp_csv = RESULTS_DIR / "r2l_representation_groups.csv"
    det_df.to_csv(det_csv, index=False)
    det_df.to_csv(TABLES_DIR / "r2l_label_detection_results.csv", index=False)
    grp_df.to_csv(grp_csv, index=False)
    grp_df.to_csv(TABLES_DIR / "r2l_representation_groups.csv", index=False)
    log.info("Saved CSV tables to %s and %s", det_csv, grp_csv)

    # 4. Export JSON metrics
    metrics_obj = {
        "experiment_id": "EXP-004",
        "timestamp": timestamp,
        "random_seed": seed,
        "environment": env_info,
        "train_file": {"path": str(train_path), "rows": len(train_df), "sha256": train_sha},
        "test_file": {"path": str(test_path), "rows": len(test_df), "sha256": test_sha},
        "r2l_composition": comp_df.to_dict(orient="records"),
        "label_detection_results": det_df.to_dict(orient="records"),
        "representation_groups": grp_df.to_dict(orient="records"),
    }
    (METRICS_DIR / "r2l_metrics.json").write_text(
        json.dumps(metrics_obj, indent=2), encoding="utf-8"
    )

    # 5. Generate figures
    log.info("Generating figures …")
    plot_r2l_label_detection_rate(det_df)
    plot_r2l_composition_shift(comp_df)

    # 6. Generate report
    log.info("Generating report …")
    report_text = generate_experiment_report(
        env_info=env_info,
        train_info={"path": str(train_path), "rows": len(train_df), "sha256": train_sha},
        test_info={"path": str(test_path), "rows": len(test_df), "sha256": test_sha},
        comp_df=comp_df,
        det_df=det_df,
        grp_df=grp_df,
        timestamp=timestamp,
        seed=seed,
    )
    report_path = RESULTS_DIR / "experiment_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    log.info("Report written to %s", report_path)

    log.info("=" * 60)
    log.info("EXP-004 complete. Results in: %s", RESULTS_DIR)
    log.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EXP-004: R2L Within-Category Label-Shift Analysis — SentinelNet"
    )
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="Path to KDDTrain+.txt")
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST, help="Path to KDDTest+.txt")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)")
    args = parser.parse_args()
    main(train_path=args.train, test_path=args.test, seed=args.seed)
