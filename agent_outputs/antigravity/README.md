# Aviation Delay Prediction Pipeline (Antigravity Agent)

## Project Objective

Build a reproducible machine learning pipeline to predict flight arrival delays (binary: delayed >15 minutes) using the US flight dataset. The goal is to create an audited production-ready system (Task 1-5) with clear Monte Carlo reproducibility (Task 6) and robust documentation (Task 8).

## Architecture Overview

- Data ingestion/cleanup: `prepare_data.py`
- Exploratory data analysis: `task2_eda.py`, `task2_basic_eda.py`, `task2_advanced_stats.py`
- Baseline modeling: `task3_baseline_model.py`
- Optimized modeling with hyperparameter tuning and leakage assessment: `task4_optimized_model.py`
- Audit leakage-safe deployment model: `task5_model_audit.py`
- Central orchestration: `run_pipeline.py`

## Prerequisites

- OS: Windows, macOS, or Linux
- Python 3.10+ recommended
- Data files present in repository root:
  - `data/flights.csv`
  - `airlines.csv`
  - `airports.csv`

## Setup Instructions

1. Clone repository:
   - `git clone https://github.com/kshitij4603/MSIN0097_Aviation_Agents.git`
   - `cd MSIN0097_Aviation_Agents`

2. Create and activate venv:
   - Windows: `python -m venv venv` and `venv\Scripts\Activate.ps1`
   - macOS/Linux: `python3 -m venv venv` and `source venv/bin/activate`

3. Install dependencies:
   - `pip install --upgrade pip`
   - `pip install -r agent_outputs/antigravity/requirements.txt`

## Execution

Run full pipeline:

```bash
python agent_outputs/antigravity/run_pipeline.py
```

Outputs will be written under `agent_outputs/antigravity/`, including:
- `task5_honest_metrics.csv`
- `task5_honest_model.pkl`
- `task5_audit_report.txt`
- `task4_best_model.pkl`
- `task3_baseline.pkl`

## Reproducibility Notes

- All scripts use deterministically set `RANDOM_SEED = 42`.
- `run_pipeline.py` executes scripts in strict order and exits on failure.
- Data paths are dynamic and relative (e.g., `../../data/flights.csv` from agent folder).

## Contact

For questions, reference this repo and the Task 5 audit report in `agent_outputs/antigravity/task5_audit_report.txt`.
