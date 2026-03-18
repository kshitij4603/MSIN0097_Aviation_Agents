# MSIN0097: Agent Error Tracker
**Purpose:** Permanent log of confirmed ML errors produced by junior agents during evaluation. Each entry documents the exact error, its impact on reported metrics, and the task in which it will be resolved.
**Auditor:** Claude Code (Lead ML Auditor)

---

## Task 1: The Domain Logic Trap
**Status:** Resolved in Task 1 evaluation.

| Agent | Error | Impact |
|---|---|---|
| **Antigravity** | Applied `SimpleImputer(strategy='median')` to `ARRIVAL_DELAY` for cancelled flights | Silently corrupted the training set — imputed a fictitious arrival delay for flights that never departed. Any downstream model trained on this data would learn from statistically meaningless targets. |
| Claude | None | — |
| Codex | None | — |

---

## Task 3: The Look-Ahead Leakage Trap
**Status:** ⚠️ UNRESOLVED — Inflated metrics remain in agent output files. Will be corrected in **Task 6**.
**Audited:** 2026-03-18

### Definition
Look-ahead leakage occurs when a model is trained on features that are only observable **after** the prediction horizon. For this project, the gold-standard rule is: **no feature may be used that is unknowable 24 hours before a flight's scheduled departure.** The following columns are categorically excluded:

| Column | Reason for Exclusion |
|---|---|
| `DEPARTURE_DELAY` | Measured at gate push-back — unknown until the flight departs |
| `TAXI_OUT` | Measured between gate push-back and wheels-off — post-departure |
| `WHEELS_OFF` | Actual wheels-off timestamp — post-departure |
| `WHEELS_ON` | Actual wheels-on timestamp — post-arrival |
| `TAXI_IN` | Measured between wheels-on and gate arrival — post-arrival |
| `ELAPSED_TIME` | Actual gate-to-gate duration — post-arrival |
| `AIR_TIME` | Actual airborne minutes — post-arrival |
| `DEPARTURE_TIME` | Actual departure time — post-departure |
| `CANCELLATION_REASON` | Only defined after a cancellation decision |
| `AIR_SYSTEM_DELAY`, `SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY` | All post-arrival delay attribution codes |

---

### Error Log — Codex

**File:** `agent_outputs/codex/task3_baseline_model.py`
**Error type:** Severe look-ahead leakage
**Severity:** 🔴 CRITICAL

**Evidence:**
```python
# task3_baseline_model.py — lines 85–98
feature_cols = [
    "MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "FLIGHT_NUMBER",
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE_HOUR", "SCHEDULED_DEPARTURE_MINUTE",
    "DEPARTURE_DELAY",   # ← LEAKAGE: post-departure, measured at gate push-back
    "SCHEDULED_TIME", "DISTANCE",
]
```

**Why the ROC-AUC is artificially inflated:**
`DEPARTURE_DELAY` is the single strongest predictor of `ARRIVAL_DELAY` in this dataset (Pearson r ≈ 0.93, as confirmed in Task 2 EDA). A flight that is already 30 minutes late at departure will almost always be late at arrival. By including `DEPARTURE_DELAY` as a feature, the model is effectively told the answer — it is not predicting whether a flight *will* be delayed; it is confirming that a flight *is already* delayed. This is not a predictive model; it is a lookup table with extra steps.

| Metric | Codex (with leakage) | Claude (clean baseline) | Inflation |
|---|---|---|---|
| ROC-AUC | **0.9375** | 0.7060 | **+23.1 percentage points** |
| F1-Score | 0.7861 | 0.3992 | +38.7 percentage points |

**Resolution:** Codex must retrain with `DEPARTURE_DELAY` removed from the feature set. Expected clean ROC-AUC: ~0.70–0.75. Task 6.

---

### Error Log — Antigravity

**File:** `agent_outputs/antigravity/task3_baseline_model.py`
**Error type:** Moderate look-ahead leakage (multiple post-departure operational features)
**Severity:** 🟠 HIGH

**Evidence:**
```python
# task3_baseline_model.py — lines 34–39
# Initial candidate list — contains multiple post-departure columns:
for col in ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE',
            'DEPARTURE_TIME',
            'TAXI_OUT',      # ← LEAKAGE: post-departure
            'TAXI_IN',       # ← LEAKAGE: post-arrival
            'WHEELS_OFF',    'WHEELS_ON',
            'SCHEDULED_TIME',
            'ELAPSED_TIME',  # ← LEAKAGE: post-arrival (actual gate-to-gate time)
            'AIR_TIME',      # ← LEAKAGE: post-arrival (actual airborne time)
            'DISTANCE', 'AIRLINE']:

# Line 39 removes only a partial set — TAXI_OUT, TAXI_IN, ELAPSED_TIME, AIR_TIME remain:
selected_features = [f for f in features if f not in ['DAY', 'DEPARTURE_TIME',
                                                        'WHEELS_OFF', 'WHEELS_ON']]
```

**Why the ROC-AUC is artificially inflated:**
`ELAPSED_TIME` (actual gate-to-gate duration) is a near-direct mathematical component of `ARRIVAL_DELAY`:
> `ARRIVAL_DELAY ≈ ELAPSED_TIME − SCHEDULED_TIME + DEPARTURE_DELAY`

Including `ELAPSED_TIME` and `AIR_TIME` leaks the inflight delay component directly. `TAXI_OUT` and `TAXI_IN` leak the ground-operation delay component. Together, these features give the model significant post-hoc knowledge of the delay magnitude, not pre-departure prediction signal.

Notably, the inflation is smaller than Codex's because Antigravity excluded `DEPARTURE_DELAY` (the dominant single leak). The residual inflation comes from the operational timing features:

| Metric | Antigravity (with leakage) | Claude (clean baseline) | Inflation |
|---|---|---|---|
| ROC-AUC | **0.7471** | 0.7060 | **+4.1 percentage points** |
| F1-Score | 0.3169 | 0.3992 | −8.2 pp (F1 lower despite higher AUC — precision/recall imbalance) |

**Top feature importances from Antigravity's own output (confirming leakage):**
1. `TAXI_OUT` — rank 1 (post-departure)
2. `ELAPSED_TIME` — rank 3 (post-arrival)
3. `AIR_TIME` — rank 6 (post-arrival)
4. `TAXI_IN` — rank 7 (post-arrival)

Four of the top seven most important features are leakage columns.

**Resolution:** Antigravity must retrain using only `MONTH`, `DAY`, `DAY_OF_WEEK`, `SCHEDULED_DEPARTURE`, `SCHEDULED_TIME`, `DISTANCE`, `AIRLINE`. Expected clean ROC-AUC: ~0.68–0.73. Task 6.

---

## Cumulative Error Summary

| Agent | Task 1 Error | Task 3 Error | Total Critical Errors |
|---|---|---|---|
| **Claude** | None | None | **0** |
| **Codex** | None | DEPARTURE_DELAY leakage (+23pp ROC-AUC inflation) | **1 (Critical)** |
| **Antigravity** | SimpleImputer on cancelled flights | TAXI_OUT, ELAPSED_TIME, AIR_TIME, TAXI_IN leakage (+4pp ROC-AUC inflation) | **2 (1 Critical, 1 High)** |

---

## Task 4: Target Variable Used as Feature (Antigravity)
**Status:** ⚠️ UNRESOLVED — Will be corrected in **Task 6**.
**Audited:** 2026-03-18

### Error Log — Antigravity

**File:** `agent_outputs/antigravity/task4_optimized_model.py`
**Error type:** Catastrophic target leakage — target variable used directly as a training feature
**Severity:** 🔴🔴 CRITICAL (beyond severe)

**Evidence:**
```python
# task4_optimized_model.py — line 69
candidate_features = [
    ..., 'ARRIVAL_DELAY', ...   # ← TARGET VARIABLE
]
# line 34
df['DELAY_15'] = (df['ARRIVAL_DELAY'] > 15).astype(int)   # ← TARGET DERIVED FROM ARRIVAL_DELAY
```

Additionally:
```python
# line 55
df['DELAY_DIFF'] = df['ARRIVAL_DELAY'] - df['DEPARTURE_DELAY']  # encodes ARRIVAL_DELAY algebraically
```

**Observed impact:** All metrics — baseline RF and optimized XGBoost alike — achieved perfect scores:
- ROC-AUC = 1.0000
- F1 = 1.0000
- Accuracy = 1.0000
- Precision = 1.0000, Recall = 1.0000

**Why:** The model can trivially infer `DELAY_15 = (ARRIVAL_DELAY > 15)` by directly inspecting the `ARRIVAL_DELAY` feature. No learning occurs. This is not a model — it is a deterministic rule lookup masquerading as a classifier.

**Resolution:** Remove `ARRIVAL_DELAY`, `DELAY_DIFF`, and all other post-arrival columns from the feature set. Task 6.

---

## Task 4: Leakage Persists — Codex Adds Derivatives
**Status:** ⚠️ UNRESOLVED — Will be corrected in **Task 6**.
**Audited:** 2026-03-18

**File:** `agent_outputs/codex/task4_optimized_model.py`

Codex not only retained `DEPARTURE_DELAY` from Task 3 but added two additional derivatives:
- `DEPARTURE_DELAY_CLIPPED = DEPARTURE_DELAY.clip(-30, 180)`
- `DEPARTURE_DELAY_15_PLUS = (DEPARTURE_DELAY > 15).astype(int)`

Three correlated representations of the same leakage signal are now in the feature set. Paradoxically, this caused **metric degradation** (ROC-AUC −0.43pp, F1 −1.59pp) because additional noisy encodings of an already-saturating feature provide no marginal value.

---

## Cumulative Error Summary (Updated Task 4)

| Agent | Task 1 Error | Task 3 Error | Task 4 Error | Total Critical |
|---|---|---|---|---|
| **Claude** | None | None | None | **0** |
| **Codex** | None | DEPARTURE_DELAY leakage | DEPARTURE_DELAY + 2 derivatives | **1 Critical (escalating)** |
| **Antigravity** | SimpleImputer on cancelled flights | Operational leakage (TAXI_OUT, etc.) | **TARGET VARIABLE as feature (ROC-AUC=1.0)** | **3 (2 Critical, 1 High)** |

---
*Last updated: 2026-03-18 by Claude Code Auditor*
