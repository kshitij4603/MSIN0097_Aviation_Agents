"""
Aviation Dataset — Load, Join, Missing-Value Handling, Memory Optimisation
==========================================================================
Data source : Predictive_group_coursework_data/
  - flights.csv          (~5.8 M rows)
  - airlines.csv         (IATA code → airline name)
  - airports.csv         (IATA code → airport metadata)

Output      : agent_outputs/claude_code/flights_cleaned.parquet
"""

import os
import sys
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(BASE_DIR, "Predictive_group_coursework_data")
OUT_DIR   = os.path.join(BASE_DIR, "agent_outputs", "claude_code")
os.makedirs(OUT_DIR, exist_ok=True)

FLIGHTS_PATH  = os.path.join(DATA_DIR, "flights.csv")
AIRLINES_PATH = os.path.join(DATA_DIR, "airlines.csv")
AIRPORTS_PATH = os.path.join(DATA_DIR, "airports.csv")

# ── 1. Load reference tables ───────────────────────────────────────────────────
print("=" * 70)
print("STEP 1 — Loading reference tables")
print("=" * 70)

airlines_df = pd.read_csv(AIRLINES_PATH)
airports_df = pd.read_csv(AIRPORTS_PATH)

print(f"airlines shape : {airlines_df.shape}")
print(f"airports shape : {airports_df.shape}")

# ── 2. Load main flights table with selective dtypes ──────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — Loading flights.csv with optimised dtypes")
print("=" * 70)

# Explicitly cast integer-like columns to nullable Int16/Int32 to handle NaNs
# without upcasting to float64, keeping memory footprint minimal.
DTYPE_MAP = {
    "YEAR"               : "Int16",
    "MONTH"              : "Int8",
    "DAY"                : "Int8",
    "DAY_OF_WEEK"        : "Int8",
    "FLIGHT_NUMBER"      : "Int32",
    "SCHEDULED_DEPARTURE": "Int32",
    "DEPARTURE_TIME"     : "Int32",
    "DEPARTURE_DELAY"    : "Int16",
    "TAXI_OUT"           : "Int16",
    "WHEELS_OFF"         : "Int32",
    "SCHEDULED_TIME"     : "Int16",
    "ELAPSED_TIME"       : "Int16",
    "AIR_TIME"           : "Int16",
    "DISTANCE"           : "Int32",
    "WHEELS_ON"          : "Int32",
    "TAXI_IN"            : "Int16",
    "SCHEDULED_ARRIVAL"  : "Int32",
    "ARRIVAL_TIME"       : "Int32",
    "ARRIVAL_DELAY"      : "float32",   # target; keep float for imputation
    "DIVERTED"           : "Int8",
    "CANCELLED"          : "Int8",
    "AIR_SYSTEM_DELAY"   : "float32",
    "SECURITY_DELAY"     : "float32",
    "AIRLINE_DELAY"      : "float32",
    "LATE_AIRCRAFT_DELAY": "float32",
    "WEATHER_DELAY"      : "float32",
}

flights_df = pd.read_csv(
    FLIGHTS_PATH,
    dtype=DTYPE_MAP,
    low_memory=False,
)

print(f"flights shape  : {flights_df.shape}")

# ── 3. LEFT JOINs — map IATA codes to descriptive labels ─────────────────────
print("\n" + "=" * 70)
print("STEP 3 — LEFT JOINs (airline name + origin / destination airports)")
print("=" * 70)

# 3a. Airline name
flights_df = flights_df.merge(
    airlines_df.rename(columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"}),
    on="AIRLINE",
    how="left",
)

# 3b. Origin airport metadata
origin_cols = {
    "IATA_CODE": "ORIGIN_AIRPORT",
    "AIRPORT"  : "ORIGIN_AIRPORT_NAME",
    "CITY"     : "ORIGIN_CITY",
    "STATE"    : "ORIGIN_STATE",
    "LATITUDE" : "ORIGIN_LAT",
    "LONGITUDE": "ORIGIN_LON",
}
flights_df = flights_df.merge(
    airports_df.rename(columns=origin_cols)[list(origin_cols.values())],
    on="ORIGIN_AIRPORT",
    how="left",
)

# 3c. Destination airport metadata
dest_cols = {
    "IATA_CODE": "DESTINATION_AIRPORT",
    "AIRPORT"  : "DEST_AIRPORT_NAME",
    "CITY"     : "DEST_CITY",
    "STATE"    : "DEST_STATE",
    "LATITUDE" : "DEST_LAT",
    "LONGITUDE": "DEST_LON",
}
flights_df = flights_df.merge(
    airports_df.rename(columns=dest_cols)[list(dest_cols.values())],
    on="DESTINATION_AIRPORT",
    how="left",
)

print(f"Post-join shape : {flights_df.shape}")

# Verify join quality — unmatched IATA codes
unmatched_airlines  = flights_df["AIRLINE_NAME"].isna().sum()
unmatched_origin    = flights_df["ORIGIN_AIRPORT_NAME"].isna().sum()
unmatched_dest      = flights_df["DEST_AIRPORT_NAME"].isna().sum()
print(f"Unmatched airline codes   : {unmatched_airlines:,}")
print(f"Unmatched origin airports : {unmatched_origin:,}")
print(f"Unmatched dest airports   : {unmatched_dest:,}")

# ── 4. Analyse ARRIVAL_DELAY missingness ─────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4 — ARRIVAL_DELAY missingness analysis")
print("=" * 70)

total_rows       = len(flights_df)
total_missing    = flights_df["ARRIVAL_DELAY"].isna().sum()
cancelled_mask   = flights_df["CANCELLED"] == 1
diverted_mask    = flights_df["DIVERTED"]  == 1

n_cancelled      = cancelled_mask.sum()
n_diverted       = diverted_mask.sum()
n_missing_cancel = flights_df.loc[cancelled_mask, "ARRIVAL_DELAY"].isna().sum()
n_missing_divert = flights_df.loc[diverted_mask,  "ARRIVAL_DELAY"].isna().sum()
n_missing_other  = (
    flights_df["ARRIVAL_DELAY"].isna()
    & ~cancelled_mask
    & ~diverted_mask
).sum()

print(f"Total rows                       : {total_rows:>10,}")
print(f"ARRIVAL_DELAY missing (total)    : {total_missing:>10,}  "
      f"({100*total_missing/total_rows:.2f}%)")
print(f"  ↳ from CANCELLED == 1          : {n_missing_cancel:>10,}  "
      f"(of {n_cancelled:,} cancelled)")
print(f"  ↳ from DIVERTED  == 1          : {n_missing_divert:>10,}  "
      f"(of {n_diverted:,} diverted)")
print(f"  ↳ other (unexplained)          : {n_missing_other:>10,}")

# ── 5. Statistically sound missing-value strategy ────────────────────────────
print("\n" + "=" * 70)
print("STEP 5 — Missing-value strategy")
print("=" * 70)

# Rationale
# ---------
# CANCELLED flights: ARRIVAL_DELAY is structurally undefined — the flight
#   never landed.  These rows are NOT suitable targets for a regression model
#   predicting delay minutes.  They are removed (listwise deletion is valid
#   because missingness is NOT AT RANDOM; it is caused by the CANCELLED flag
#   itself — a deterministic, observable mechanism).
#
# DIVERTED flights with missing ARRIVAL_DELAY: A diverted flight landed at an
#   unplanned airport; its official ARRIVAL_DELAY is sometimes unreported.
#   Since diversion is rare and the delay value is not structurally absent
#   (the plane did land), we impute with the **median ARRIVAL_DELAY of
#   non-cancelled, non-diverted flights on the same route (ORIGIN→DEST)**.
#   Route-level median is preferred over global median because delay
#   distributions are strongly route-dependent (distance, hub congestion).
#   Where a route has no observations (< 5), we fall back to the global median.
#
# Residual unexplained NaNs: treated identically to diverted NaNs.

# ── 5a. Drop cancelled flights ───────────────────────────────────────────────
n_before = len(flights_df)
flights_df = flights_df[~cancelled_mask].copy()
n_after  = len(flights_df)
print(f"Rows dropped (cancelled)   : {n_before - n_after:,}")
print(f"Rows remaining             : {n_after:,}")

# ── 5b. Route-level median imputation for remaining NaNs ────────────────────
still_missing_mask = flights_df["ARRIVAL_DELAY"].isna()
n_still_missing    = still_missing_mask.sum()
print(f"ARRIVAL_DELAY still missing after cancellation drop : {n_still_missing:,}")

if n_still_missing > 0:
    # Compute route-level median on complete cases (min 5 observations)
    MIN_OBS = 5
    route_median = (
        flights_df
        .dropna(subset=["ARRIVAL_DELAY"])
        .groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])["ARRIVAL_DELAY"]
        .median()
    )
    # Keep only routes with sufficient support
    route_counts = (
        flights_df
        .dropna(subset=["ARRIVAL_DELAY"])
        .groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])["ARRIVAL_DELAY"]
        .count()
    )
    reliable_routes = route_counts[route_counts >= MIN_OBS].index
    route_median    = route_median.loc[route_median.index.isin(reliable_routes)]

    global_median = float(flights_df["ARRIVAL_DELAY"].median())
    print(f"Global median ARRIVAL_DELAY        : {global_median:.2f} min")
    print(f"Routes with >= {MIN_OBS} obs used for imputation : {len(route_median):,}")

    def _impute(row):
        if pd.isna(row["ARRIVAL_DELAY"]):
            key = (row["ORIGIN_AIRPORT"], row["DESTINATION_AIRPORT"])
            return route_median.get(key, global_median)
        return row["ARRIVAL_DELAY"]

    # Vectorised lookup via map on MultiIndex
    idx = pd.MultiIndex.from_arrays(
        [flights_df["ORIGIN_AIRPORT"], flights_df["DESTINATION_AIRPORT"]]
    )
    imputed_values = route_median.reindex(idx).values

    flights_df["ARRIVAL_DELAY"] = np.where(
        still_missing_mask,
        np.where(np.isnan(imputed_values.astype(float)), global_median, imputed_values),
        flights_df["ARRIVAL_DELAY"].values,
    ).astype("float32")

    remaining_na = flights_df["ARRIVAL_DELAY"].isna().sum()
    print(f"ARRIVAL_DELAY missing after imputation : {remaining_na:,}")

# ── 6. Cast string columns to category ───────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 6 — Cast object/string columns to category dtype")
print("=" * 70)

STRING_COLS = [
    "AIRLINE", "TAIL_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    "CANCELLATION_REASON", "AIRLINE_NAME",
    "ORIGIN_AIRPORT_NAME", "ORIGIN_CITY", "ORIGIN_STATE",
    "DEST_AIRPORT_NAME",   "DEST_CITY",   "DEST_STATE",
]

# Only cast columns that actually exist in the frame
string_cols_present = [c for c in STRING_COLS if c in flights_df.columns]
mem_before = flights_df.memory_usage(deep=True).sum() / 1024**2

for col in string_cols_present:
    flights_df[col] = flights_df[col].astype("category")

mem_after = flights_df.memory_usage(deep=True).sum() / 1024**2
print(f"Memory before category cast : {mem_before:>8.1f} MB")
print(f"Memory after  category cast : {mem_after:>8.1f} MB")
print(f"Memory saved                : {mem_before - mem_after:>8.1f} MB  "
      f"({100*(mem_before-mem_after)/mem_before:.1f}% reduction)")

# ── 7. DataFrame info ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 7 — DataFrame .info() (memory optimisation confirmation)")
print("=" * 70)
flights_df.info(verbose=True, memory_usage="deep", show_counts=True)

# ── 8. Persist cleaned dataset ────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "flights_cleaned.parquet")
flights_df.to_parquet(out_path, index=False)
print(f"\nCleaned dataset saved → {out_path}")
print("=" * 70)
