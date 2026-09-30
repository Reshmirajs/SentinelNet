# SentinelNet

> **Academic AI/ML Research Project — Network Intrusion Detection**

[![Status](https://img.shields.io/badge/status-pre--experiment-yellow)](experiments/registry.csv)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](requirements.txt)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

---

## Project Purpose

SentinelNet is a **research-oriented machine learning project** investigating automated detection of network intrusions and traffic anomalies. The project applies modern supervised, unsupervised, and deep learning techniques to benchmark network intrusion detection datasets, with the goal of producing rigorous, reproducible, and publishable research.

This is **not** a production security tool. It is an academic research codebase designed to support systematic experimentation, ablation studies, and fair benchmarking across model families.

---

## Research-Oriented Nature

SentinelNet follows a strict **Research Constitution** (see [`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md)) that governs all contributions — human and AI alike. Key principles:

- **No fabricated results.** All metrics and figures come from actual experiment runs.
- **Full reproducibility.** Every experiment logs its random seed, config, and environment.
- **Separation of concerns.** Data, preprocessing, features, models, and evaluation are each isolated in their own module.
- **Explicit methodology.** No algorithmic decisions are made without researcher approval.
- **Registered experiments.** All runs are tracked in [`experiments/registry.csv`](experiments/registry.csv).

---

## High-Level Architecture

```
Raw Network Traffic Data
         │
         ▼
   src/data/          ← Data loading and I/O utilities
         │
         ▼
   src/preprocessing/ ← Cleaning, normalization, train/val/test splits
         │
         ▼
   src/features/      ← Feature engineering and selection
         │
         ▼
   src/models/        ← Model architecture definitions
         │
         ▼
   src/experiments/   ← Experiment orchestration (reads from configs/)
         │
         ▼
   src/evaluation/    ← Metric computation, statistical testing
         │
         ▼
   results/           ← Figures, tables, metrics (from real runs only)
```

All experiment configuration is driven by YAML files in `configs/`. Model training and evaluation are never hardcoded — they always reference a config.

---

## Repository Structure

```
sentinelnet/
│
├── AGENTS.md                  # Research Constitution for AI coding agents
├── CLAUDE.md                  # Research Constitution for Claude
├── README.md                  # This file
├── .gitignore                 # Excludes data, models, and outputs
├── requirements.txt           # Python dependencies
│
├── configs/                   # Experiment configuration files (YAML)
│
├── data/
│   ├── README.md              # Dataset provenance and documentation
│   ├── raw/                   # Original data — READ ONLY, never committed
│   ├── interim/               # Partially processed — regenerable
│   └── processed/             # ML-ready features — regenerable
│
├── src/
│   ├── data/                  # Data loading utilities
│   ├── preprocessing/         # Cleaning and transformation pipelines
│   ├── features/              # Feature engineering and selection
│   ├── models/                # Model architecture definitions
│   ├── evaluation/            # Metric computation and analysis
│   ├── experiments/           # Experiment orchestration scripts
│   └── utils/                 # Shared utilities (logging, config, IO)
│
├── experiments/
│   └── registry.csv           # Master experiment tracking log
│
├── results/
│   ├── metrics/               # JSON/CSV metric outputs from runs
│   ├── figures/               # Plots and visualizations
│   └── tables/                # LaTeX/CSV result tables
│
├── models/                    # Saved model checkpoints (gitignored)
│
├── notebooks/                 # Exploratory and analysis notebooks
│
├── paper/
│   ├── literature/            # Literature review notes and references
│   ├── methodology/           # Methodology documentation
│   ├── results/               # Draft results sections
│   └── discussion/            # Discussion and conclusion drafts
│
└── tests/                     # Unit and integration tests
```

---

## Current Project Status

| Phase | Status |
|-------|--------|
| Repository scaffolding | ✅ Complete |
| Research constitution | ✅ Complete |
| Dataset acquisition | ⬜ Not started |
| Data preprocessing pipeline | ⬜ Not started |
| Feature engineering | ⬜ Not started |
| Baseline models | ⬜ Not started |
| Experiments | ⬜ Not started |
| Results analysis | ⬜ Not started |
| Paper writing | ⬜ Not started |

> **No experiments have been run yet. No results exist. All research findings are pending.**

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Reshmirajs/SentinelNet.git
cd SentinelNet

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

> **Dataset setup:** See [`data/README.md`](data/README.md) for dataset provenance and acquisition instructions. Datasets are **not** included in this repository.

---

## Research Constitution

All contributors — human and AI — must read and follow the Research Constitution before making any changes to this repository:

- [`AGENTS.md`](AGENTS.md) — For AI coding agents (Copilot, Cursor, Antigravity, etc.)
- [`CLAUDE.md`](CLAUDE.md) — For Claude (Anthropic)

Key prohibitions: no fabricated results, no dataset downloads without instruction, no model training during scaffolding, no methodological decisions without researcher approval.

---

## Contributing

This is an academic research project. Contributions must follow the Research Constitution. Please open an issue before submitting pull requests that affect experimental methodology.

---

## License

MIT License. See `LICENSE` for details.

---

*SentinelNet Research Team — 2026*
