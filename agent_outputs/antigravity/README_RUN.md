# Reproducible Packaging: Task 6

## 1) Clone repository

Windows (PowerShell):
```powershell
git clone https://github.com/kshitij4603/MSIN0097_Aviation_Agents.git
cd MSIN0097_Aviation_Agents
```

Mac/Linux:
```bash
git clone https://github.com/kshitij4603/MSIN0097_Aviation_Agents.git
cd MSIN0097_Aviation_Agents
```

## 2) Create and activate Python virtual environment

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```
Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r agent_outputs/antigravity/requirements.txt
```

## 4) Run the full pipeline

```bash
python agent_outputs/antigravity/run_pipeline.py
```

This will execute, in order:
- `agent_outputs/antigravity/prepare_data.py`
- `agent_outputs/antigravity/task2_eda.py`
- `agent_outputs/antigravity/task2_basic_eda.py`
- `agent_outputs/antigravity/task2_advanced_stats.py`
- `agent_outputs/antigravity/task3_baseline_model.py`
- `agent_outputs/antigravity/task4_optimized_model.py`
- `agent_outputs/antigravity/task5_model_audit.py`

## 5) Confirm output files

Check `agent_outputs/antigravity/` for:
- `task5_honest_metrics.csv`
- `task5_honest_feature_importance.png`
- `task5_honest_model.pkl`
- `task5_audit_report.txt`

## Reproducibility guarantee

1. Dependencies are pinned in `agent_outputs/antigravity/requirements.txt` (numpy, pandas, matplotlib, seaborn, scikit-learn, statsmodels, xgboost, joblib, python-dateutil, pytz) and should be installed before running the pipeline.
2. Every modeling script uses `RANDOM_SEED = 42` and fixed `train_test_split(random_state=RANDOM_SEED, stratify=y)` to make train/test split deterministic.
3. All data loads use dynamic relative pathing to the raw dataset as required:
   - `Path(__file__).resolve().parents[2] / 'data' / 'flights.csv'` (i.e., `../../data/flights.csv` from the `agent_outputs/antigravity` folder).
4. `run_pipeline.py` runs scripts in strict sequential order and raises on first failure to avoid partial results.

This ensures any new team can clone the repo, install packages, and run the pipeline to reproduce the same final audited ROC-AUC score and derived artifacts with no pathing, missing dependency, or randomness drift issues.
