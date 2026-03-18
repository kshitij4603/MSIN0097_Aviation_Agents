# Codex Pipeline Runbook

This folder packages the full Codex aviation workflow so a new engineer can reproduce the data preparation, EDA, baseline modeling, optimization, and leakage-audited final model from a clean clone.

## Files in this package

- `requirements.txt`: pinned Python dependencies required by the Codex pipeline
- `run_pipeline.py`: master script that runs all Codex pipeline steps in chronological order
- `preprocess_aviation_data.py`: merged and sampled dataset preparation
- `eda_delay_drivers.py`, `task2_eda.py`, `task2_basic_eda.py`, `task2_advanced_stats.py`: exploratory analysis scripts
- `task3_baseline_model.py`: baseline classifier
- `task4_optimized_model.py`: optimized classifier
- `task5_model_audit.py`: leakage-audited final model evaluation

## Assumptions

- Run all commands from the repository root: `MSIN0097_Aviation_Agents`
- The raw dataset must exist at `data/flights.csv`, `data/airlines.csv`, and `data/airports.csv`
- The scripts themselves use dynamic relative paths such as `../../data/flights.csv`, so do not move the `agent_outputs/codex/` folder

## Mac / Linux setup

```bash
cd MSIN0097_Aviation_Agents
python3 -m venv venv
source venv/bin/activate
pip install -r agent_outputs/codex/requirements.txt
python agent_outputs/codex/run_pipeline.py
```

## Windows PowerShell setup

```powershell
cd MSIN0097_Aviation_Agents
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r agent_outputs/codex/requirements.txt
python .\agent_outputs\codex\run_pipeline.py
```

## Windows Command Prompt setup

```cmd
cd MSIN0097_Aviation_Agents
py -3 -m venv venv
venv\Scripts\activate.bat
pip install -r agent_outputs\codex\requirements.txt
python agent_outputs\codex\run_pipeline.py
```

## Optional validation

To confirm the execution order without running the pipeline:

```bash
python agent_outputs/codex/run_pipeline.py --dry-run
```

## Reproducibility guarantee

This package is reproducible because every pipeline script uses project-relative paths, the dependencies are pinned in `agent_outputs/codex/requirements.txt`, and the modeling scripts enforce a strict global `RANDOM_SEED = 42` for sampling, splitting, and training. The final honest model in Task 5 removes all identified look-ahead leakage features and writes deterministic outputs from the same cleaned pre-departure feature set each time. If a new user runs the commands above against the same repository contents and raw data files, they should reproduce the same Task 5 honest ROC-AUC of `0.665` and the same leakage-audited metrics.
