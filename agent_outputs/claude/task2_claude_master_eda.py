"""task2_claude_master_eda.py
Claude (Apple Silicon) — Task 2 Master EDA
Covers four tasks:
  1. Worst Offenders  — delay RATE (not volume), airline + airport
  2. Descriptive Stats — skew-aware: median, P90, P99
  3. Correlations & Pivot — categorical/ID columns excluded; DoW × Airline pivot
  4. Advanced Stats    — VIF (iterative singular-safe) + Breusch-Pagan heteroscedasticity plot
All outputs saved strictly to agent_outputs/claude/.
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive — safe for Apple Silicon / headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# Limit BLAS threads — critical on Apple Silicon shared memory
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent          # agent_outputs/claude/
REPO_ROOT    = SCRIPT_DIR.parent.parent                  # repo root
DATA_PATH    = REPO_ROOT / "data" / "flights.csv"
OUT_DIR      = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE  = 500_000

# ── 1. Load & sample ───────────────────────────────────────────────────────────
print("[1/5] Loading and sampling data …")
df_full = pd.read_csv(DATA_PATH, low_memory=False)
print(f"      Full dataset: {len(df_full):,} rows × {df_full.shape[1]} cols")

df = df_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).copy()
del df_full

# Operated, non-cancelled flights with a valid ARRIVAL_DELAY
df_ops = df.loc[(df["CANCELLED"] == 0) & df["ARRIVAL_DELAY"].notna()].copy()
df_ops["delayed_15"] = (df_ops["ARRIVAL_DELAY"] > 15).astype(int)
print(f"      Operated flights in 500k sample: {len(df_ops):,}")

sns.set_theme(style="whitegrid", context="talk", palette="muted")

# ── 2. Worst Offenders (delay RATE) ───────────────────────────────────────────
print("[2/5] Computing worst offenders (delay rate) …")

airline_rate = (
    df_ops.groupby("AIRLINE")
    .agg(delay_rate=("delayed_15", "mean"), n=("delayed_15", "size"))
    .query("n >= 200")
    .sort_values("delay_rate", ascending=True)
    .reset_index()
)

airport_rate = (
    df_ops.groupby("ORIGIN_AIRPORT")
    .agg(delay_rate=("delayed_15", "mean"), n=("delayed_15", "size"))
    .query("n >= 300")
    .sort_values("delay_rate", ascending=True)
    .tail(20)
    .reset_index()
)

pct_fmt = mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

palette_a = sns.color_palette("flare", len(airline_rate))
axes[0].barh(airline_rate["AIRLINE"], airline_rate["delay_rate"],
             color=palette_a, edgecolor="white")
for bar, rate in zip(axes[0].patches, airline_rate["delay_rate"]):
    axes[0].text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                 f"{rate*100:.1f}%", va="center", fontsize=9)
axes[0].xaxis.set_major_formatter(pct_fmt)
axes[0].set_xlabel("Share of Flights Delayed >15 min")
axes[0].set_title("Worst Offender Airlines\n(Delay Rate, min. 200 flights in sample)")

palette_b = sns.color_palette("flare", len(airport_rate))
axes[1].barh(airport_rate["ORIGIN_AIRPORT"], airport_rate["delay_rate"],
             color=palette_b, edgecolor="white")
axes[1].xaxis.set_major_formatter(pct_fmt)
axes[1].set_xlabel("Share of Flights Delayed >15 min")
axes[1].set_title("Top-20 Worst Origin Airports\n(Delay Rate, min. 300 flights in sample)")

plt.suptitle("Task 2 — Worst Offenders: Delay Rate (not volume)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "eda_worst_offenders.png", dpi=200, bbox_inches="tight")
plt.close()
print("      ✓ eda_worst_offenders.png")

# ── 3. Descriptive Stats (skew-aware) ─────────────────────────────────────────
print("[3/5] Descriptive statistics …")

arr = df_ops["ARRIVAL_DELAY"]
dist_stats = pd.DataFrame([{
    "metric":           "ARRIVAL_DELAY (minutes)",
    "count":            int(arr.count()),
    "mean":             round(float(arr.mean()), 2),
    "std":              round(float(arr.std()),  2),
    "skewness":         round(float(arr.skew()), 3),
    "kurtosis":         round(float(arr.kurt()), 3),
    "min":              round(float(arr.min()),  2),
    "p25":              round(float(arr.quantile(0.25)), 2),
    "median_p50":       round(float(arr.median()), 2),
    "p75":              round(float(arr.quantile(0.75)), 2),
    "p90":              round(float(arr.quantile(0.90)), 2),
    "p99":              round(float(arr.quantile(0.99)), 2),
    "max":              round(float(arr.max()),  2),
}])
dist_stats.to_csv(OUT_DIR / "eda_delay_distribution.csv", index=False)
print("      ✓ eda_delay_distribution.csv")
print(dist_stats.T.to_string())

# ── 4. Correlations & Pivot ────────────────────────────────────────────────────
print("[4/5] Correlations and pivot table …")

# Exclude: constant cols, ID-like ints, circular HHMM time ints, post-departure leakage
EXCLUDE_CORR = {
    "YEAR", "FLIGHT_NUMBER",                          # constant / pseudo-ID
    "SCHEDULED_DEPARTURE", "DEPARTURE_TIME",           # HHMM circular ints
    "WHEELS_OFF", "WHEELS_ON",                         # HHMM circular ints
    "ARRIVAL_TIME", "SCHEDULED_ARRIVAL",               # HHMM circular ints
    "DIVERTED", "CANCELLED",                           # binary flags
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY",              # delay breakdown — mostly NaN
    "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
    "delayed_15",                                       # derived target — don't correlate
}

num_cols = [c for c in df_ops.select_dtypes(include="number").columns
            if c not in EXCLUDE_CORR]
corr_matrix = df_ops[num_cols].corr().round(4)
corr_matrix.to_csv(OUT_DIR / "eda_correlations.csv")
print(f"      ✓ eda_correlations.csv  ({len(num_cols)} features)")

DOW_MAP = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
pivot = (
    df_ops.groupby(["DAY_OF_WEEK", "AIRLINE"])["ARRIVAL_DELAY"]
    .mean()
    .unstack("AIRLINE")
    .round(2)
)
pivot.index = pivot.index.map(DOW_MAP)
pivot.to_csv(OUT_DIR / "eda_airline_day_pivot.csv")
print("      ✓ eda_airline_day_pivot.csv")

# ── 5. VIF + Heteroscedasticity ────────────────────────────────────────────────
print("[5/5] VIF and heteroscedasticity …")

# Pre-departure, continuous, non-leakage features only
VIF_CANDIDATES = ["MONTH", "DAY", "DAY_OF_WEEK", "SCHEDULED_TIME", "DISTANCE"]

vif_df = df_ops[VIF_CANDIDATES + ["ARRIVAL_DELAY"]].dropna()

# Drop any near-zero-variance column (would produce a singular design matrix)
variances = vif_df[VIF_CANDIDATES].var()
vif_features = variances[variances > 1e-6].index.tolist()

X_vif = vif_df[vif_features].astype(float)
X_vif_const = sm.add_constant(X_vif, has_constant="add")

vif_results = pd.DataFrame({
    "feature": X_vif_const.columns,
    "VIF": [
        round(float(variance_inflation_factor(X_vif_const.values, i)), 3)
        for i in range(X_vif_const.shape[1])
    ],
}).query("feature != 'const'").sort_values("VIF", ascending=False).reset_index(drop=True)

vif_results.to_csv(OUT_DIR / "eda_multicollinearity.csv", index=False)
print("      ✓ eda_multicollinearity.csv")
print(vif_results.to_string(index=False))

# Breusch-Pagan heteroscedasticity test on OLS residuals
y_het  = vif_df["ARRIVAL_DELAY"].values.astype(float)
X_het  = sm.add_constant(X_vif.values, has_constant="add")
ols    = sm.OLS(y_het, X_het).fit()
resids = ols.resid
fitted = ols.fittedvalues

bp_lm, bp_p, _, _ = het_breuschpagan(resids, X_het)
print(f"      Breusch-Pagan LM stat = {bp_lm:.2f},  p-value = {bp_p:.2e}")
verdict = "HETEROSCEDASTIC — variance of residuals is non-constant" if bp_p < 0.05 \
          else "Homoscedastic — variance appears constant"
print(f"      Verdict: {verdict}")

# Subsample for plotting speed (Apple Silicon unified memory guard)
rng     = np.random.default_rng(RANDOM_STATE)
plot_n  = min(25_000, len(fitted))
idx     = rng.choice(len(fitted), size=plot_n, replace=False)
std_res = (resids - resids.mean()) / resids.std()

fig, axes = plt.subplots(1, 2, figsize=(17, 6))

# Plot A: Residuals vs Fitted
axes[0].scatter(fitted[idx], resids[idx], alpha=0.12, s=4, color="steelblue", rasterized=True)
axes[0].axhline(0, color="crimson", lw=1.5, ls="--")
# Lowess smoother to reveal systematic pattern
lowess = sm.nonparametric.lowess(resids[idx], fitted[idx], frac=0.15)
axes[0].plot(lowess[:, 0], lowess[:, 1], color="darkorange", lw=2, label="LOWESS")
axes[0].legend(fontsize=9)
axes[0].set_xlabel("Fitted Values (min)")
axes[0].set_ylabel("Residuals (min)")
axes[0].set_title("Residuals vs Fitted")
axes[0].text(0.02, 0.97,
             f"Breusch-Pagan p = {bp_p:.2e}\n{verdict}",
             transform=axes[0].transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="grey"))

# Plot B: Scale-Location (√|standardised residuals| vs fitted)
axes[1].scatter(fitted[idx], np.sqrt(np.abs(std_res[idx])),
                alpha=0.12, s=4, color="darkorange", rasterized=True)
lowess2 = sm.nonparametric.lowess(np.sqrt(np.abs(std_res[idx])), fitted[idx], frac=0.15)
axes[1].plot(lowess2[:, 0], lowess2[:, 1], color="steelblue", lw=2, label="LOWESS")
axes[1].legend(fontsize=9)
axes[1].set_xlabel("Fitted Values (min)")
axes[1].set_ylabel("√|Standardised Residuals|")
axes[1].set_title("Scale-Location Plot")

plt.suptitle(
    f"OLS Residual Diagnostics — Predictors: {', '.join(vif_features)}",
    fontsize=12, y=1.01
)
plt.tight_layout()
plt.savefig(OUT_DIR / "eda_heteroscedasticity.png", dpi=200, bbox_inches="tight")
plt.close()
print("      ✓ eda_heteroscedasticity.png")

# ── Final verification ─────────────────────────────────────────────────────────
EXPECTED = [
    "eda_worst_offenders.png",
    "eda_delay_distribution.csv",
    "eda_correlations.csv",
    "eda_airline_day_pivot.csv",
    "eda_multicollinearity.csv",
    "eda_heteroscedasticity.png",
]

print("\n" + "=" * 62)
print("  CLAUDE TASK 2 EDA — ARTIFACT VERIFICATION")
print("=" * 62)
all_ok = True
for fname in EXPECTED:
    p = OUT_DIR / fname
    exists = p.exists()
    size   = f"{p.stat().st_size / 1024:.1f} KB" if exists else "MISSING"
    status = "✓" if exists else "✗"
    if not exists:
        all_ok = False
    print(f"  {status}  {fname:<42}  {size}")

print("=" * 62)
if all_ok:
    print("  ALL ARTIFACTS GENERATED — PHASE 2 CAN PROCEED")
else:
    print("  WARNING: ONE OR MORE ARTIFACTS MISSING")
print("=" * 62)
