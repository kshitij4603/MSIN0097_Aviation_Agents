# Model Card: Task 5 Honest Delay Prediction Model

## Model Overview

**Model name:** Task5 Honest XGBoost Classifier

**Purpose:** Predict whether a flight will arrive >15 minutes late (`DELAY_15` target) using only pre-departure features, minimizing look-ahead leakage.

**Location:** `agent_outputs/antigravity/task5_honest_model.pkl`

## Intended Use

- Production risk assessment for flight operations and scheduling forecasts.
- Secondary usage: capacity planning and delay mitigation alerts.

### Out-of-scope

- Real-time airborne in-flight delay prediction (post-takeoff features are excluded).
- Airport or route-specific policy decisions without recalibrating for local context.

## Training Data

- Source dataset: `data/flights.csv` (U.S. flight schedule dataset).
- Data filtering: `CANCELLED` and `DIVERTED` flights removed; required fields `ARRIVAL_DELAY`, `DEPARTURE_DELAY`, `DISTANCE` are present.
- Train sample: includes up to 500,000 rows sampled with random seed 42.

## Feature Engineering

- Pre-departure features: `MONTH`, `DAY_OF_WEEK`, `SCHED_DEP_HOUR`, `SCHED_ARR_HOUR`, `DISTANCE`, `SCHEDULED_TIME`
- Categorical one-hot encoding: `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT` (top 20 + OTHER)
- Numeric scaling via `StandardScaler`
- Leaky features removed before Task 5 model (Task 4 leaky features excluded, e.g., `ARRIVAL_DELAY`, `DEPARTURE_DELAY`, `AIR_TIME`, `TAXI_OUT`, `TAXI_IN`).

## Evaluation Data & Metrics

- Train/test split: 80/20 stratified with random seed 42.

### Final audited performance (Task 5)

- Primary metric: ROC-AUC on holdout test set
- Stored in `agent_outputs/antigravity/task5_honest_metrics.csv`

From pipeline evaluation example:
- `roc_auc`: reported in audit script; expected ~[real run value from run output].
- `accuracy`, `precision`, `recall`, `f1_score` also computed and listed in metrics file.

## Limitations and Ethical Considerations

- Distributional shift: model trained on historic flight data may degrade with seasonal changes, route modernization, or fleet updates. Requires regular re-training.
- Bias risk: features like `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT` may encode systemic bias; mitigation via fairness review is advised.
- Missing event data: model assumes complete `SCHEDULED_*` fields; untracked data entry errors may reduce reliability.
- Adverse decisions: should not be sole basis for passenger impact actions (e.g., compensation, crew dispatch). Use as an advisory score.

## Deployment Guidelines

- Use same pre-processing logic from `task5_model_audit.py` as inference pipeline.
- Keep dependency versions aligned with `agent_outputs/antigravity/requirements.txt`.
- Maintain random seed for reproducible baseline comparison.
- Conduct periodic backtesting and fairness audits for model drift.

## Usage example

```python
import joblib
import pandas as pd

model = joblib.load('agent_outputs/antigravity/task5_honest_model.pkl')
# preprocess new rows with same pipeline as Task 5
# model.predict_proba(X_new)[:, 1]
```

## Contact

For further details and assumptions, see `agent_outputs/antigravity/task5_audit_report.txt`.
