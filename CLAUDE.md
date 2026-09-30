# CLAUDE.md — SentinelNet Research Constitution

> **Audience:** Claude (Anthropic AI assistant)  
> This file governs all Claude-assisted contributions to SentinelNet.  
> Claude must read and follow this constitution before taking any action in this repository.

---

## 1. Project Identity

**SentinelNet** is an **academic AI/ML research project** for network intrusion detection.

- It is **not** a cybersecurity scripting project.
- It is **not** a production security tool.
- It is **not** a simple classification pipeline.

The goal is rigorous, reproducible, and publishable machine-learning research on network traffic anomaly and intrusion detection.

---

## 2. The Research Constitution

### 2.1 Scientific Integrity

**You must never:**
- Fabricate experimental results, metrics, or figures.
- Create placeholder accuracy numbers (e.g., `accuracy = 0.97`).
- Invent dataset statistics that have not been computed from real data.
- Write conclusions that are not supported by actual experiments.
- Pre-fill results tables with made-up values.

**You must always:**
- Mark unrun experiments clearly as `STATUS: PENDING`.
- Use `# TODO: Run experiment` comments instead of fabricated values.
- Distinguish clearly between implemented code and validated results.

### 2.2 Reproducibility

Every experiment must be reproducible. This means:

- All random seeds must be set explicitly and logged in `experiments/registry.csv`.
- All hyperparameters must be tracked in config files under `configs/`.
- Data preprocessing steps must be documented and version-controlled.
- Model checkpoints must reference the exact config and seed used.
- No "it worked on my machine" — use dependency pinning in `requirements.txt`.

### 2.3 Dataset Handling

**You must never:**
- Download datasets without explicit human instruction.
- Commit raw dataset files to the repository.
- Generate synthetic data and present it as real network traffic.
- Modify raw data files — treat `data/raw/` as read-only.

**You must always:**
- Document dataset provenance in `data/README.md`.
- Keep raw data separate from processed data (`data/raw/` vs `data/processed/`).
- Use only datasets explicitly approved by the research lead.

### 2.4 Model Development

**You must never:**
- Hardcode model architectures without config-driven alternatives.
- Train models during repository setup or scaffolding phases.
- Assume a model is correct before validation on held-out data.
- Skip cross-validation or use only train-set metrics.

**You must always:**
- Separate model definition, training logic, and evaluation logic.
- Use `src/models/` for architecture definitions only.
- Use `src/experiments/` for training runs and experiment orchestration.
- Use `src/evaluation/` for metric computation and result analysis.

### 2.5 Code Quality

- Follow PEP 8 for Python code.
- Write docstrings for all public functions and classes.
- Use type hints throughout (`def process(x: np.ndarray) -> pd.DataFrame:`).
- Write unit tests for preprocessing and feature engineering in `tests/`.
- Do not commit broken or untested code to `main`.

### 2.6 Research Methodology

**You must never:**
- Make methodological decisions (model choice, feature selection strategy, evaluation protocol) without explicit human guidance.
- Implement a specific ML algorithm unless the research lead has specified it.
- Choose hyperparameter search spaces arbitrarily.

**You must always:**
- Ask before implementing a novel model variant.
- Document all design decisions in the relevant config or notebook.
- Flag ambiguous requirements as `# RESEARCH DECISION NEEDED:` comments.

---

## 3. Repository Structure Rules

| Directory | Purpose | Rules |
|-----------|---------|-------|
| `data/raw/` | Original, immutable data | **Read-only. Never modify.** |
| `data/interim/` | Partially processed data | Regenerable from raw |
| `data/processed/` | Final ML-ready features | Regenerable from interim |
| `src/data/` | Data loading utilities | No business logic |
| `src/preprocessing/` | Cleaning and transformation | Must be reversible/logged |
| `src/features/` | Feature engineering | All features must be documented |
| `src/models/` | Model architectures | No training code here |
| `src/evaluation/` | Metrics and analysis | No model definitions here |
| `src/experiments/` | Experiment orchestration | References configs only |
| `src/utils/` | Shared utilities | Must be unit-tested |
| `configs/` | YAML/JSON experiment configs | One file per experiment |
| `experiments/` | Run artifacts and logs | Auto-generated, gitignored except registry |
| `results/` | Figures, tables, metrics | Only from real runs |
| `models/` | Saved model checkpoints | gitignored, reference by hash |
| `notebooks/` | Exploratory analysis | Not for final results |
| `paper/` | Manuscript sections | Do not write without data |
| `tests/` | Unit and integration tests | Must pass before merge |

---

## 4. Git Workflow

- `main` — stable, reviewed, reproducible code only.
- `dev` — integration branch for features.
- Feature branches named: `feature/<short-description>`.
- Experiment branches named: `exp/<experiment-id>`.
- Commit messages must reference the experiment ID when applicable.
- Never force-push to `main`.

---

## 5. Claude-Specific Instructions

### 5.1 Before Starting Any Task

Claude must:
1. Re-read this file (CLAUDE.md) to ensure current task is permitted.
2. Check `experiments/registry.csv` for any relevant prior experiments.
3. Check `configs/` for existing configuration schemas before creating new ones.
4. Never assume the state of the codebase — always read relevant files first.

### 5.2 When Writing Code

- Always include the module docstring at the top of every new Python file.
- Always include `if __name__ == "__main__":` guards on runnable scripts.
- Never import `*` from any module.
- Use `pathlib.Path` instead of `os.path` for file handling.
- Use `logging` instead of `print` for informational output in library code.

### 5.3 When Answering Research Questions

- Clearly state when a question requires running an experiment to answer.
- Do not speculate about likely accuracy or performance.
- Cite methodology references when suggesting techniques.

### 5.4 When Uncertain

When requirements are unclear:
1. Insert a comment: `# RESEARCH DECISION NEEDED: <describe the ambiguity>`
2. Create a stub/placeholder that makes the decision point explicit.
3. Explicitly tell the researcher what decision is needed before proceeding.

---

## 6. What Claude Is Permitted To Do

✅ Create repository structure and scaffolding  
✅ Write utility functions with full docstrings and type hints  
✅ Write unit tests for existing utilities  
✅ Implement data loading and preprocessing pipelines  
✅ Implement feature engineering pipelines  
✅ Implement model architectures as specified by the research lead  
✅ Write training/evaluation scripts that follow the config schema  
✅ Refactor existing code without changing behavior  
✅ Update documentation to reflect actual code  
✅ Explain ML concepts and research methodology options  
✅ Review code for correctness and reproducibility issues  

---

## 7. What Claude Is Prohibited From Doing

❌ Downloading or fetching any dataset  
❌ Training any model  
❌ Fabricating or estimating any numeric result  
❌ Writing the research paper or manuscript sections  
❌ Making methodological decisions (algorithm choice, hyperparameters, evaluation protocol)  
❌ Modifying files in `data/raw/`  
❌ Committing model weights or large binary files  
❌ Bypassing the experiment registry  

---

*Last updated: 2026-09-30 | SentinelNet Research Team*
