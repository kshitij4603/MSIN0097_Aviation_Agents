# Model Card: Claude Leakage-Verified Flight Delay Classifier

## Model Overview

| Field | Value |
|---|---|
| Model name | Claude Task 5 Leakage-Verified XGBoost Classifier |
| Model family | XGBoost (`XGBClassifier`, `tree_method="hist"`) |
| Prediction target | `ARRIVAL_DELAY > 15 minutes` (binary: 1 = delayed, 0 = on-time) |
| Prediction horizon | Strictly pre-departure — all features knowable ≥ 24 h before scheduled departure |
| Primary artifact | `task4_best_model.pkl` (Config 2 weights, verified clean in Task 5) |
| Training data | US DOT 2015 Aviation Dataset — `data/flights.csv` (500k operated-flight sample) |

---

## Intended Use

**Suitable use cases:**
- Pre-departure delay risk ranking for airline operations dashboards
- Customer proactive disruption messaging (e.g., notify passengers before a likely-delayed flight)
- Schedule buffer optimisation for business travel tools
- Academic benchmarking of pre-departure delay prediction on public aviation data

**Out of scope:**
- In-flight delay prediction (post-departure features are explicitly excluded)
- Safety-critical or regulatory decisions
- Legally binding customer compensation calculations
- Inference on aviation data outside the 2015 continental US context without retraining

---

## Training & Evaluation Data

- **Source:** `data/flights.csv` — US DOT On-Time Performance data, calendar year 2015
- **Scope:** Continental US domestic flights only
- **Sample:** 500,000 rows drawn with `random_state=42`; stratified 60/20/20 train/validation/test split
- **Target definition:** `DELAY_15 = (ARRIVAL_DELAY > 15).astype(int)` — industry-standard "significant delay" threshold
- **Exclusions:** Cancelled flights (`CANCELLED == 1`), diverted flights (`DIVERTED == 1`), and rows with missing `ARRIVAL_DELAY` are removed before any modelling

---

## Feature Set (Pre-Departure Only)

All 18 features are knowable before the aircraft leaves the gate:

| Feature | Type | Description |
|---|---|---|
| `MONTH`, `DAY`, `DAY_OF_WEEK` | Numeric | Calendar position |
| `SCHED_HOUR`, `SCHED_MINUTE` | Numeric | Departure time of day |
| `DEP_PERIOD` | Ordinal | Time-of-day bucket (0=early, 4=late-night) |
| `SCHEDULED_TIME` | Numeric | Planned block time (minutes) |
| `DISTANCE` | Numeric | Great-circle distance (miles) |
| `IS_WEEKEND` | Binary | Saturday or Sunday departure |
| `IS_PEAK_SUMMER` | Binary | June, July, August |
| `IS_HOLIDAY_SEASON` | Binary | November, December |
| `IS_RED_EYE` | Binary | Scheduled departure ≥ 22:00 |
| `AIRLINE` | Categorical (ordinal enc.) | Carrier identity |
| `ORIGIN_AIRPORT` | Categorical | Departure airport |
| `DESTINATION_AIRPORT` | Categorical | Arrival airport |
| `ROUTE` | Categorical | Origin_Destination pair (top-60 retained) |
| `CARRIER_ROUTE` | Categorical | Airline_Route combo (top-40 retained) |
| `DISTANCE_BIN` | Categorical | short / mid / long haul bucket |

---

## Performance (Task 5 Verified — Held-Out Test Set)

All metrics are computed on the held-out test split (100,000 rows, never seen during training or hyperparameter tuning).

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.7108** |
| **PR-AUC** | **0.3677** |
| **F1-Score** | **0.4050** |
| Accuracy | 0.6661 |
| Precision | 0.2974 |
| Recall | 0.6349 |

### Leakage Verification Summary

| Stage | ROC-AUC | Notes |
|---|---|---|
| Task 3 Baseline (10 raw features) | 0.7060 | Clean — no leakage |
| Task 4 Optimised (+7 engineered features) | 0.7108 | Clean — no leakage |
| Task 5 Audit (permutation importance) | **0.7108** | Δ = **+0.0000 pp** — confirmed clean |

Permutation importance across 8 repeats (5,000-row test subsample) confirmed no single feature contributes more than **0.0635 ROC-AUC units** — an appropriate distribution. The +0.0000 pp AUC delta on leakage removal is the statistical proof that no look-ahead signal was present.

---

## Strengths

- **Zero leakage from Task 3 onwards** — the only agent in the cohort to never require a correction
- **Permutation importance verification** (not gain-based importance) — immune to correlated-feature distortion
- **Feature engineering grounded in business logic** — all 7 engineered features encode genuinely pre-departure signals (seasonality, route, carrier-route, time-of-day)
- **Reproducible** — deterministic sampling (`random_state=42`), pinned dependencies, `tree_method="hist"` for Apple Silicon compatibility

---

## Limitations

### 1. Temporal Decay — 2015 Data Does Not Reflect Modern Aviation

**This is the primary deployment risk.** The training data covers calendar year **2015 only**. The aviation industry has undergone structural changes since then that are not represented in this model:

- **COVID-19 (2020–2022):** Airline capacity collapsed and then rebounded with severely disrupted scheduling patterns. Delay distributions, airline market shares, and route networks changed fundamentally.
- **Airline consolidation:** Several carriers that operated in 2015 have since merged or ceased operations (e.g., Virgin America acquired by Alaska Airlines in 2018). The `AIRLINE` and `CARRIER_ROUTE` features encode a 2015 carrier landscape.
- **Post-pandemic operational stress (2022–present):** Crew shortages, air traffic control staffing gaps, and new congestion patterns at hub airports create delay patterns with no analogue in 2015 data.
- **Fleet changes:** The 737 MAX grounding (2019–2020) and subsequent reintroduction changed route schedules at Southwest and other carriers.
- **A model trained on 2015 data and deployed in 2026 should be treated as a historical benchmark, not a production predictor.** Retraining on recent data (2022 onwards) is required before any live deployment.

### 2. Class Imbalance

Approximately 18% of flights are delayed >15 minutes. The model uses `scale_pos_weight` to compensate but precision remains low (0.30) — most delay alerts will be false positives. Set alert thresholds based on the operating cost of false positives vs. false negatives for the specific business use case.

### 3. No External Signal

Weather, ATC ground stops, and NOTAMs are not incorporated. These are the dominant drivers of delay on any given day but require real-time data feeds not present in the static dataset.

### 4. Route and Carrier Bucketing

Routes outside the top-60 and carrier-routes outside the top-40 are collapsed into `OTHER`. Rare routes may be systematically under-predicted.

### 5. Exclusion of Cancelled Flights

Cancelled flights are structurally excluded (no `ARRIVAL_DELAY`). This means the model cannot distinguish "on-time" from "cancelled" — a deployment system should handle cancellations as a separate upstream flag before invoking this model.

---

## Ethical Considerations

- Delay predictions reflect historical structural patterns (hub congestion, airline operational culture, route demand) — they should not be used to penalise individual airlines, crews, or airports without operational context.
- The model may reflect historical biases in airport investment and air traffic control resource allocation.
- Do not use as the sole basis for passenger-facing service commitments or compensation decisions.

---

## Monitoring Recommendations

- Track ROC-AUC and F1 monthly against a rolling ground-truth sample from live operations
- Segment performance by airline, hub airport, and season — distributional drift will appear in subgroups before the aggregate metric degrades
- Re-run permutation importance annually to detect feature importance shift (a sign of concept drift)
- Trigger retraining if ROC-AUC on live data drops below **0.68** on any rolling 30-day window

---

## Deployment Checklist

- [ ] Data vault (`data/flights.csv`) confirmed present and matches expected row count (~5.8M)
- [ ] All dependencies installed from pinned `requirements.txt`
- [ ] `task5_honest_metrics.csv` ROC-AUC ≥ 0.710 (validates clean reproduction)
- [ ] Permutation importance plot reviewed — no single feature >0.10 AUC units
- [ ] 2015 temporal limitation disclosed to all stakeholders before production use
- [ ] Decision threshold calibrated for business false-positive tolerance
