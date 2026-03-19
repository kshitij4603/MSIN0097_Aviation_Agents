# Aviation Delay Prediction Pipeline

## Project Objective

This project builds an end-to-end machine learning pipeline for US flight delay analysis and prediction. The core business goal is to help stakeholders understand delay risk, identify structural drivers of late arrivals, and deploy a leakage-audited classifier that predicts whether a flight will arrive more than 15 minutes late.

## What This Package Contains

The `agent_outputs/codex/` directory contains the Codex delivery for:

- data preparation and merged dataset creation
- exploratory data analysis and statistical diagnostics
- baseline and optimized predictive models
- leakage auditing and honest re-evaluation
- reproducible packaging artifacts for handoff

Key scripts include:

- `preprocess_aviation_data.py`
- `eda_delay_drivers.py`
- `task2_eda.py`
- `task2_basic_eda.py`
- `task2_advanced_stats.py`
- `task3_baseline_model.py`
- `task4_optimized_model.py`
- `task5_model_audit.py`
- `run_pipeline.py`

## Repository Prerequisites

Before running the pipeline, make sure:

- you are in the repository root: `MSIN0097_Aviation_Agents`
- the raw data files exist in the repo `data/` directory:
  - `data/flights.csv`
  - `data/airlines.csv`
  - `data/airports.csv`
- Python 3.12 or a compatible Python 3 environment is installed

## Environment Setup

### Mac / Linux

```bash
cd MSIN0097_Aviation_Agents
python3 -m venv venv
source venv/bin/activate
pip install -r agent_outputs/codex/requirements.txt
```

### Windows PowerShell

```powershell
cd MSIN0097_Aviation_Agents
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r agent_outputs/codex/requirements.txt
```

### Windows Command Prompt

```cmd
cd MSIN0097_Aviation_Agents
py -3 -m venv venv
venv\Scripts\activate.bat
pip install -r agent_outputs\codex\requirements.txt
```

## How To Execute The Pipeline

Run the full pipeline from the repository root:

```bash
python agent_outputs/codex/run_pipeline.py
```

The master runner executes the Codex workflow in chronological order:

1. data preparation
2. exploratory data analysis
3. baseline modeling
4. optimized modeling
5. leakage audit and honest final evaluation

To preview the run order without executing:

```bash
python agent_outputs/codex/run_pipeline.py --dry-run
```

## Main Outputs

Representative outputs created by the pipeline include:

- EDA CSVs and PNGs for descriptive analysis
- `task3_metrics.csv`, `task3_feature_importance.png`, `task3_baseline.pkl`
- `task4_optimized_metrics.csv`, `task4_roc_comparison.png`, `task4_best_model.pkl`
- `task5_honest_metrics.csv`, `task5_honest_feature_importance.png`

## Final Audited Model Summary

The production-facing model is the Task 5 leakage-audited model, not the earlier baseline or optimized models. After removing illegal look-ahead features, the honest model achieved:

- Accuracy: `0.6432`
- Precision: `0.2683`
- Recall: `0.5836`
- F1-score: `0.3676`
- ROC-AUC: `0.6647`

This is the defensible performance level to communicate to stakeholders, because it reflects a true pre-departure prediction setting.

## Important Leakage Note

Earlier model versions used post-departure information such as:

- `DEPARTURE_DELAY`
- `DEPARTURE_DELAY_CLIPPED`
- `DEPARTURE_DELAY_15_PLUS`

These were removed during Task 5 because they are not legally available at prediction time for a true pre-departure use case. The audited model card in `model_card.md` explains the implications in detail.

## Recommended Handoff Sequence

For a new engineering team:

1. set up the environment with the pinned requirements
2. verify the raw data is in `data/`
3. run `python agent_outputs/codex/run_pipeline.py`
4. use the Task 5 outputs as the trusted production reference
5. review `model_card.md` before deployment decisions
