# Claude Aviation Delay Prediction Pipeline

## Project Objective

This pipeline predicts whether a US domestic flight will arrive more than 15 minutes late, using only information available at least 24 hours before scheduled departure. It was developed and evaluated across six tasks covering ingestion, EDA, baseline modelling, optimisation, leakage auditing, and production packaging.

---

## Repository & Data Vault Architecture

This project uses a **centralised data vault** pattern. Raw source files are never stored inside agent output folders — they live in a single canonical location at the repository root and are read by all agent scripts via dynamic absolute paths.

```
MSIN0097_Aviation_Agents/
├── data/                          ← DATA VAULT (not tracked in git — see .gitignore)
│   ├── flights.csv                ← US DOT 2015 Aviation Dataset (~5.8M rows)
│   ├── airlines.csv               ← IATA airline code reference
│   └── airports.csv               ← Airport metadata reference
├── agent_outputs/
│   ├── claude/                    ← This package
│   │   ├── README.md
│   │   ├── model_card.md
│   │   ├── requirements.txt
│   │   ├── run_pipeline.py
│   │   ├── task2_claude_master_eda.py
│   │   ├── task3_claude_baseline.py
│   │   ├── task4_claude_optimized.py
│   │   └── task5_claude_audit.py
│   ├── codex/
│   └── antigravity/
└── MSIN0097_Comparative_Analysis.ipynb
```

> **Important:** `flights.csv` is excluded from git via `.gitignore` due to its size (~600 MB). Download the US DOT 2015 ATAD dataset and place it at `data/flights.csv` before running. All scripts resolve this path using `Path(__file__).resolve().parents[2] / "data" / "flights.csv"` — no manual path editing required.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ (tested on 3.13.7) |
| Available RAM | ≥ 4 GB (500k-row XGBoost training) |
| `data/flights.csv` | US DOT 2015 Aviation Dataset placed in repo vault |

---

## Mac / Linux Setup

```bash
# 1. Clone
git clone https://github.com/kshitij4603/MSIN0097_Aviation_Agents.git
cd MSIN0097_Aviation_Agents

# 2. Place data file
#    Copy flights.csv into the data/ directory (create it if absent)
mkdir -p data
cp /path/to/flights.csv data/

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install pinned dependencies
pip install --upgrade pip
pip install -r agent_outputs/claude/requirements.txt

# 5. Run pipeline
python agent_outputs/claude/run_pipeline.py
```

---

## Windows PowerShell Setup

```powershell
# 1. Clone
git clone https://github.com/kshitij4603/MSIN0097_Aviation_Agents.git
cd MSIN0097_Aviation_Agents

# 2. Place data (create data\ folder and copy flights.csv into it)

# 3. Create virtual environment
py -3 -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Install pinned dependencies
pip install --upgrade pip
pip install -r agent_outputs\claude\requirements.txt

# 5. Run pipeline
python agent_outputs\claude\run_pipeline.py
```

---

## Windows Command Prompt Setup

```cmd
git clone https://github.com/kshitij4603/MSIN0097_Aviation_Agents.git
cd MSIN0097_Aviation_Agents
py -3 -m venv venv
venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r agent_outputs\claude\requirements.txt
python agent_outputs\claude\run_pipeline.py
```

---

## Pipeline Steps

```bash
# Preview run order without executing anything
python agent_outputs/claude/run_pipeline.py --dry-run

# Run a single step
python agent_outputs/claude/run_pipeline.py --step task3_claude_baseline.py
```

| Step | Script | Key Output |
|---|---|---|
| 1 | `task2_claude_master_eda.py` | `eda_worst_offenders.png`, `eda_multicollinearity.csv`, `eda_heteroscedasticity.png` |
| 2 | `task3_claude_baseline.py` | `task3_metrics.csv` — ROC-AUC **0.7060** |
| 3 | `task4_claude_optimized.py` | `task4_optimized_metrics.csv` — ROC-AUC **0.7108** |
| 4 | `task5_claude_audit.py` | `task5_honest_metrics.csv` — ROC-AUC **0.7108**, Δ = **+0.0000** |

---

## Audited Model Performance

The production model is the Task 5 leakage-verified output. All metrics are from the held-out test set (20% of 500k-row sample, stratified).

| Metric | Value |
|---|---|
| ROC-AUC | **0.7108** |
| PR-AUC | 0.3677 |
| F1-Score | 0.4050 |
| Accuracy | 0.6661 |
| Precision | 0.2974 |
| Recall | 0.6349 |
| Δ vs Task 4 (leakage removed) | **+0.0000 pp** |

Zero AUC drop on leakage removal confirms the model was clean from Task 3 onwards.

---

## Leakage Policy

This pipeline enforces a strict **24-hour pre-departure prediction horizon**. The following columns are categorically excluded from all feature sets:

`DEPARTURE_DELAY`, `TAXI_OUT`, `TAXI_IN`, `WHEELS_OFF`, `WHEELS_ON`, `ELAPSED_TIME`, `AIR_TIME`, `DEPARTURE_TIME`, `CANCELLATION_REASON`, and all delay attribution codes (`AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`).

See `model_card.md` for the full limitations and deployment guidance.
