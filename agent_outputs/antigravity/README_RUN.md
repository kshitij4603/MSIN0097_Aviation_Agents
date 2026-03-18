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

I have pinned dependencies in `agent_outputs/antigravity/requirements.txt`, used a strict `RANDOM_SEED = 42` in every script, and loader paths are exact relative dynamic locations (`../../data/flights.csv`) in the code. The master script `agent_outputs/antigravity/run_pipeline.py` runs each piece in deterministic order and stops on error, guaranteeing the same ROC-AUC and metrics as the audited run when using the same dataset and environment.
