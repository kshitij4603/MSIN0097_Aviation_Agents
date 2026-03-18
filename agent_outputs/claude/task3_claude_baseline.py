"""task3_claude_baseline.py
Claude (Apple Silicon) — Task 3 Clean Baseline
Model: XGBoost binary classifier (TARGET: ARRIVAL_DELAY > 15 min)

Gold Standard Leakage Exclusion List — columns dropped BEFORE feature selection:
  DEPARTURE_DELAY, TAXI_OUT, WHEELS_OFF, WHEELS_ON, TAXI_IN,
  CANCELLATION_REASON (unavailable pre-departure; would cause look-ahead leakage)
Only features knowable ≥24 hours before departure are retained.
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "2")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
DATA_PATH   = REPO_ROOT / "data" / "flights.csv"
OUT_DIR     = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE  = 500_000

# ── Gold Standard: columns that constitute look-ahead leakage ─────────────────
# These are only observable AFTER the flight departs — unknowable at booking time.
LEAKAGE_COLS = {
    "DEPARTURE_DELAY",    # measured at gate push-back — post-departure
    "TAXI_OUT",           # measured wheel-off minus push-back — post-departure
    "WHEELS_OFF",         # actual wheels-off timestamp — post-departure
    "WHEELS_ON",          # actual wheels-on timestamp — post-departure
    "TAXI_IN",            # measured at gate arrival — post-arrival
    "CANCELLATION_REASON",# only defined after a cancellation decision — post-departure
    "DEPARTURE_TIME",     # actual departure time — post-departure
    "ELAPSED_TIME",       # actual elapsed flight time — post-arrival
    "AIR_TIME",           # actual airborne time — post-arrival
    "ARRIVAL_TIME",       # actual arrival time — post-arrival
    "AIR_SYSTEM_DELAY",   # delay breakdown — only known post-arrival
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
}

# ── Pre-departure features retained for modelling ─────────────────────────────
LOAD_COLS = [
    "MONTH", "DAY", "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",    # HHMM int — hour extracted below
    "SCHEDULED_TIME",         # scheduled flight duration (minutes)
    "DISTANCE",               # great-circle distance (miles)
    "CANCELLED", "DIVERTED",
    "ARRIVAL_DELAY",          # target construction only
]

CAT_COLS = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
NUM_COLS = [
    "MONTH", "DAY", "DAY_OF_WEEK",
    "SCHED_HOUR",             # derived from SCHEDULED_DEPARTURE
    "SCHED_MINUTE",
    "SCHEDULED_TIME", "DISTANCE",
]
FEATURE_COLS = NUM_COLS + CAT_COLS

# ── 1. Load & sample ───────────────────────────────────────────────────────────
print("[1/5] Loading data …")
df_raw = pd.read_csv(DATA_PATH, usecols=LOAD_COLS, low_memory=False)
print(f"      Raw rows: {len(df_raw):,}")

# Keep only operated, non-diverted flights with a valid arrival delay
df = df_raw.loc[
    (df_raw["CANCELLED"] == 0)
    & (df_raw["DIVERTED"] == 0)
    & df_raw["ARRIVAL_DELAY"].notna()
].copy()

# 500k random sample — consistent seed
df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
print(f"      Operated rows in 500k sample: {len(df):,}")

# ── 2. Feature engineering (pre-departure only) ───────────────────────────────
print("[2/5] Feature engineering …")

# Decompose SCHEDULED_DEPARTURE HHMM → hour + minute (circular proxy for time-of-day)
sched = df["SCHEDULED_DEPARTURE"].fillna(0).astype(int).astype(str).str.zfill(4)
df["SCHED_HOUR"]   = sched.str[:2].astype(int)
df["SCHED_MINUTE"] = sched.str[2:].astype(int)

# Binary target: delayed by more than 15 minutes
df["TARGET"] = (df["ARRIVAL_DELAY"] > 15).astype(int)
class_balance = df["TARGET"].mean()
print(f"      Class balance (delayed >15 min): {class_balance:.3%}")

# Ordinal-encode categoricals (XGBoost tree splits work natively on integer codes;
# faster and more memory-efficient than OHE for high-cardinality airport codes)
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[CAT_COLS] = enc.fit_transform(df[CAT_COLS].astype(str))

X = df[FEATURE_COLS].astype(float)
y = df["TARGET"]

# ── 3. Train / test split ─────────────────────────────────────────────────────
print("[3/5] Splitting and training XGBoost …")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"      Train: {len(X_train):,}  Test: {len(X_test):,}")

# scale_pos_weight corrects class imbalance without resampling
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos
print(f"      scale_pos_weight = {spw:.2f}  (neg={neg:,}  pos={pos:,})")

model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric="logloss",
    tree_method="hist",       # Apple Silicon: histogram-based (no GPU needed)
    random_state=RANDOM_STATE,
    n_jobs=2,
    verbosity=0,
)
model.fit(X_train, y_train)

# ── 4. Evaluation ─────────────────────────────────────────────────────────────
print("[4/5] Evaluating …")

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

roc_auc  = roc_auc_score(y_test, y_prob)
pr_auc   = average_precision_score(y_test, y_prob)
acc      = accuracy_score(y_test, y_pred)
prec     = precision_score(y_test, y_pred, zero_division=0)
rec      = recall_score(y_test, y_pred, zero_division=0)
f1       = f1_score(y_test, y_pred, zero_division=0)

metrics = pd.DataFrame({
    "metric": ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1_score"],
    "value":  [roc_auc, pr_auc, acc, prec, rec, f1],
})
metrics["value"] = metrics["value"].round(6)
metrics.to_csv(OUT_DIR / "task3_metrics.csv", index=False)
print(f"      ROC-AUC : {roc_auc:.4f}")
print(f"      PR-AUC  : {pr_auc:.4f}")
print(f"      F1      : {f1:.4f}")
print("      ✓ task3_metrics.csv")

# ── 5. Feature importance ──────────────────────────────────────────────────────
print("[5/5] Plotting feature importance …")

importance_df = (
    pd.DataFrame({"feature": FEATURE_COLS, "importance": model.feature_importances_})
    .sort_values("importance", ascending=False)
    .head(10)
    .sort_values("importance", ascending=True)
    .reset_index(drop=True)
)

sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(13, 7))
bars = ax.barh(
    importance_df["feature"],
    importance_df["importance"],
    color=sns.color_palette("crest", len(importance_df)),
    edgecolor="white",
)
for bar, val in zip(bars, importance_df["importance"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)

ax.set_xlabel("XGBoost Feature Importance (gain)")
ax.set_title(
    "Task 3 — Claude Clean Baseline: Top-10 Feature Importances\n"
    "(DEPARTURE_DELAY, TAXI_OUT, TAXI_IN, WHEELS_OFF/ON excluded — no leakage)"
)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

plt.tight_layout()
plt.savefig(OUT_DIR / "task3_feature_importance.png", dpi=200, bbox_inches="tight")
plt.close()
print("      ✓ task3_feature_importance.png")

# Save model
joblib.dump({"model": model, "encoder": enc, "features": FEATURE_COLS},
            OUT_DIR / "task3_baseline.pkl")
print("      ✓ task3_baseline.pkl")

# ── Summary ────────────────────────────────────────────────────────────────────
top3 = importance_df["feature"].tail(3).tolist()[::-1]

print("\n" + "=" * 65)
print("  CLAUDE TASK 3 CLEAN BASELINE — COMPLETE")
print("=" * 65)
print(f"  ROC-AUC : {roc_auc:.4f}   PR-AUC : {pr_auc:.4f}")
print(f"  F1-Score: {f1:.4f}   Accuracy: {acc:.4f}")
print(f"  Top features: {', '.join(top3)}")
print(f"  Leakage cols excluded: {len(LEAKAGE_COLS)}")
print("=" * 65)
print(
    f"\nBusiness interpretation: A clean, deployment-safe XGBoost model achieves "
    f"ROC-AUC={roc_auc:.3f} using only information available at booking time. "
    f"The most predictive pre-departure signals are {top3[0]} and {top3[1]}, "
    f"confirming that route structure and scheduled timing carry real predictive "
    f"signal independent of any operational data. Any model reporting ROC-AUC >0.90 "
    f"on this task should be audited immediately for look-ahead leakage."
)
