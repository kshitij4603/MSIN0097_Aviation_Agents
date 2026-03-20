# MSIN0097: Agentic AI in Practice — Comparative Analysis

**Module:** MSIN0097 — Predictive Analytics | UCL School of Management
**Submission:** March 2026

---

## Team

| Name | Role |
|---|---|
| Majid Ahmadi Moghaddam | Agent evaluation, statistical analysis |
| Aljoharah Waleed I Alrajhi | Documentation review, reproducibility testing |
| Mukund Chandak | EDA design, leakage audit methodology |
| Asude Kopuz | Model card compliance, safety/compliance assessment |
| Kshitij Virmani | Lead MLOps engineer, CI/CD pipeline, notebook assembly |

---

## Project Overview

This repository presents a rigorous, end-to-end comparative benchmarking of three AI coding agents — **Claude** (Anthropic), **Codex** (OpenAI), and **Antigravity** — executing an identical eight-task machine learning pipeline on the **2015 US DOT Aviation Dataset** (~5.8 million domestic flight records).

Each agent received the same structured task briefs, worked in isolation on its own hardware environment (Claude on Apple Silicon macOS, Codex on Intel macOS, Antigravity on Windows), and was held to a common gold standard: a binary classifier predicting whether a flight will arrive more than 15 minutes late, trained exclusively on features knowable at least 24 hours before scheduled departure.

The evaluation spans six dimensions drawn from the MSIN0097 rubric — **Correctness, Statistical Validity, Reproducibility, Code Quality, Efficiency, and Safety/Compliance** — and documents both the outputs each agent produced and the failure modes it exhibited, whether self-corrected or left unresolved. The master analysis notebook (`MSIN0097_Comparative_Analysis.ipynb`) and the structured error tracker (`MSIN0097_Agent_Error_Tracker.md`) together constitute a continuous, evidence-based audit trail from the first ingestion script to the final production-grade model card.

---

## Key Findings

- **Claude achieved zero ML errors** across all eight tasks. It was the only agent to never introduce look-ahead data leakage, never require a correction run, and to proactively verify its own model via permutation importance — producing an AUC delta of +0.0000 pp against the Task 4 reference, confirming no inflated signal existed to remove.

- **A critical data leakage trap was identified in Tasks 3 and 4.** Codex included `DEPARTURE_DELAY` (Pearson r ≈ 0.93 with the target) in its feature set, inflating its reported ROC-AUC by **+23.1 percentage points** (0.706 → 0.937). Antigravity went further by including the **target variable itself** (`ARRIVAL_DELAY`, from which `DELAY_15` is derived) as a training feature, producing a spurious ROC-AUC of **1.0000**. Both agents self-corrected in Task 5.

- **The honest performance ceiling is ROC-AUC ≈ 0.67–0.71.** After mandatory leakage correction in Task 5, all three agents converge to this band — confirming it as the true information boundary for 24-hour advance delay prediction on this dataset. Any figure materially above this range in earlier tasks is a leakage artefact, not a modelling achievement.

- **Packaging and documentation revealed a second tier of quality differences.** Claude produced the only `requirements.txt` with zero phantom or transitive dependencies, the only model card to explicitly name the 2015 temporal decay risk (COVID-19, airline consolidations, post-pandemic operational patterns), and the only README covering all three OS environments with a deployable checklist.

- **A critical secrets-handling violation was recorded.** During evaluation, Antigravity hardcoded a `ghp_...` GitHub Personal Access Token directly in a terminal command — a pattern that exposes write-scope credentials in shell history and process logs, and would trigger a mandatory security incident report in any production environment.

- **IDE-integrated agents carry structural availability risks.** Antigravity's `task2_eda.py` triggered an infinite `os.system` loop due to `observed=False` on a categorical `groupby`, consuming all CPU cores until force-killed. A subsequent cloud API outage dropped the agent session mid-execution, requiring a full restart. These incidents illustrate the fragility of IDE-integrated agents that depend on live API connectivity with no offline fallback.

---

## Repository Structure

```
MSIN0097_Aviation_Agents/
│
├── MSIN0097_Comparative_Analysis.ipynb     # Master evaluation notebook
│                                           # Task-by-task comparison matrices, visual
│                                           # evidence, graded verdicts, and the Final
│                                           # Comparative Analysis rubric section
│
├── MSIN0097_Agent_Error_Tracker.md         # Structured error log: all documented
│                                           # failures, severity classifications, and
│                                           # resolution status per agent per task
│
├── MSIN0097_Final_Written_Report.docx      # Formal 1,400-word academic analysis
│                                           # (Sections 1–5 + Appendices)
│
├── requirements.txt                        # Project-level harness dependencies (pinned)
├── generate_notebook.py                    # Notebook scaffolding utility
├── task2_eda_run.log                       # Execution log — Antigravity loop incident
│
├── data/                                   # ⚠ DATA VAULT — not tracked in git
│   ├── flights.csv                         # US DOT 2015 Aviation Dataset (~5.8M rows)
│   ├── airlines.csv                        # Airline IATA code reference
│   └── airports.csv                        # Airport metadata reference
│
└── agent_outputs/                          # One subdirectory per agent
    ├── claude/                             # Claude — Apple Silicon macOS
    │   ├── requirements.txt                # Pinned deps (7 direct, zero phantom)
    │   ├── run_pipeline.py                 # Cross-platform runner (--dry-run, --step)
    │   ├── README_RUN.md                   # 3-OS setup guide
    │   ├── README.md                       # Project README (agent-scoped)
    │   ├── model_card.md                   # Full model card with Task 5 metrics
    │   ├── task2_claude_master_eda.py
    │   ├── task3_claude_baseline.py
    │   ├── task4_claude_optimized.py
    │   └── task5_claude_audit.py
    │
    ├── codex/                              # Codex — Intel macOS
    │   ├── requirements.txt
    │   ├── run_pipeline.py
    │   ├── README_RUN.md
    │   ├── README.md
    │   ├── model_card.md
    │   └── [task scripts Tasks 1–5]
    │
    ├── antigravity/                        # Antigravity — Windows
    │   ├── requirements.txt
    │   ├── run_pipeline.py
    │   ├── README_RUN.md
    │   ├── README.md
    │   ├── model_card.md
    │   └── [task scripts Tasks 1–5]
    │
    └── claude_code/                        # Supplementary EDA output (Task 1.5 plots)
```

---

## Results at a Glance

| Agent | Task 4 AUC (reported) | Task 5 AUC (honest) | AUC Drop | Cumulative Errors | Grade |
|---|:---:|:---:|:---:|:---:|:---:|
| **Claude** | 0.7108 | **0.7108** | ±0.0000 pp | **0** | ✅ Deployment-ready |
| **Codex** | 0.9332 | **0.6647** | −26.85 pp | 2 (self-corrected) | ⚠️ Correctable |
| **Antigravity** | 1.0000 | **0.6741** | −32.59 pp | 5 (2 critical) | ❌ Requires revision |

---

## Reproducibility

Each agent subdirectory contains a `README_RUN.md` with OS-specific virtual environment setup and step-by-step execution instructions. The quickest path to reproducing Claude's full pipeline from a clean clone:

```bash
# Mac/Linux
git clone https://github.com/kshitij4603/MSIN0097_Aviation_Agents.git
cd MSIN0097_Aviation_Agents

# Place flights.csv, airlines.csv, airports.csv into data/
mkdir -p data && cp /path/to/flights.csv data/

# Environment setup
python3 -m venv venv && source venv/bin/activate
pip install -r agent_outputs/claude/requirements.txt

# Validate before running
python agent_outputs/claude/run_pipeline.py --dry-run

# Execute full pipeline
python agent_outputs/claude/run_pipeline.py
```

For Windows PowerShell, substitute `.\venv\Scripts\Activate.ps1`; for Windows CMD, use `venv\Scripts\activate.bat`. Full per-OS instructions are in each agent's `README_RUN.md`.

> **Expected Task 5 output (Claude):** `task5_honest_metrics.csv` — ROC-AUC ≈ **0.7108**, AUC drop ≈ **+0.0000 pp**. Any material deviation from this figure indicates an environment or data inconsistency.

---

## Citation

Dataset: US DOT Bureau of Transportation Statistics — 2015 Airline On-Time Performance Data.
Available at: [https://www.kaggle.com/datasets/usdot/flight-delays](https://www.kaggle.com/datasets/usdot/flight-delays)
