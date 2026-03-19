# Claude Pipeline — Reproducible Runbook

This folder packages the complete Claude aviation pipeline (Tasks 1–5). A new engineer with only `git` and Python 3.10+ can reproduce every result from a clean clone in under 10 minutes.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Tested on 3.13.7. Earlier 3.x should work. |
| `git` | To clone the repo |
| ~4 GB free RAM | 500k-row XGBoost training; Apple Silicon unified memory handled |
| `data/flights.csv` | US DOT 2015 Aviation dataset (not tracked in git — see below) |

> **Data note:** `flights.csv` (~5.8M rows) is excluded from git via `.gitignore`. Place it at `<repo_root>/data/flights.csv` before running the pipeline.

---

## Mac / Linux Setup

```bash
# 1. Clone and enter repo
git clone https://github.com/kshitij4603/MSIN0097_Aviation_Agents.git
cd MSIN0097_Aviation_Agents

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r agent_outputs/claude/requirements.txt

# 4. Run full pipeline
python agent_outputs/claude/run_pipeline.py
```

---

## Windows PowerShell Setup

```powershell
# 1. Clone and enter repo
git clone https://github.com/kshitij4603/MSIN0097_Aviation_Agents.git
cd MSIN0097_Aviation_Agents

# 2. Create virtual environment
py -3 -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r agent_outputs\claude\requirements.txt

# 4. Run full pipeline
python agent_outputs\claude\run_pipeline.py
```

---

## Windows Command Prompt Setup

```cmd
:: 1. Clone and enter repo
git clone https://github.com/kshitij4603/MSIN0097_Aviation_Agents.git
cd MSIN0097_Aviation_Agents

:: 2. Create virtual environment
py -3 -m venv venv
venv\Scripts\activate.bat

:: 3. Install pinned dependencies
pip install --upgrade pip
pip install -r agent_outputs\claude\requirements.txt

:: 4. Run full pipeline
python agent_outputs\claude\run_pipeline.py
```

---

## Running a Single Step

```bash
# Run only Task 3 baseline
python agent_outputs/claude/run_pipeline.py --step task3_claude_baseline.py

# Validate execution order without running anything
python agent_outputs/claude/run_pipeline.py --dry-run
```

---

## Expected Outputs

| Task | Key Output File | Expected Value |
|---|---|---|
| Task 2 | `eda_multicollinearity.csv` | DISTANCE/SCHEDULED_TIME VIF ≈ 32.3 |
| Task 3 | `task3_baseline_metrics.csv` | ROC-AUC ≈ **0.7060** |
| Task 4 | `task4_optimized_metrics.csv` | ROC-AUC ≈ **0.7108** |
| Task 5 | `task5_honest_metrics.csv` | ROC-AUC ≈ **0.7108**, Δ = **+0.0000** |

---

## Reproducibility Guarantee

1. **Exact version pinning:** every dependency uses `==` — no floating versions that can drift between installation dates.
2. **Deterministic sampling:** all scripts use `random_state=42` for `df.sample()`, `train_test_split()`, and model training.
3. **Cross-platform runner:** `run_pipeline.py` uses `sys.executable` (not a hardcoded path) to invoke the active venv Python on any OS.
4. **Leakage-free feature set:** all scripts load only pre-departure columns from `flights.csv` — no post-departure operational signals included.
5. **Apple Silicon compatibility:** XGBoost uses `tree_method="hist"`, `matplotlib.use("Agg")`, and thread-limiting env vars — safe on unified-memory hardware.
