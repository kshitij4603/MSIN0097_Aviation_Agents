import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


RANDOM_SEED = 42
SAMPLE_SIZE = 500_000
CHUNK_SIZE = 100_000
# Resolve repo root dynamically: agent_outputs/codex/ → up 2 levels → repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "agent_outputs" / "codex"


def reservoir_sample_csv(csv_path: Path, sample_size: int, chunk_size: int, random_seed: int) -> pd.DataFrame:
    """
    Simple random sample without replacement.

    The original row-by-row Vitter Algorithm R implementation had two bugs:
      1. IndexError: reservoir was never grown to sample_size before iloc replacements
         began (chunk_size=100k < sample_size=500k, so fill phase never completed).
      2. Performance: the replacement phase looped over every remaining row in Python,
         making it O(5.8M) Python iterations — prohibitively slow on Apple Silicon.

    Replacement strategy: read the full CSV via chunks (preserves the chunked-read
    pattern for memory visibility), concatenate into a single DataFrame, then draw
    a simple random sample with pandas — a single vectorised C operation equivalent
    to Algorithm R asymptotically and identical in statistical properties.
    """
    rng = check_random_state(random_seed)
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    if len(df) < sample_size:
        raise ValueError(
            f"Requested {sample_size:,} rows, but only {len(df):,} rows are available in {csv_path.name}."
        )

    return df.sample(n=sample_size, random_state=rng).reset_index(drop=True)


def cast_object_columns_to_category(df: pd.DataFrame) -> pd.DataFrame:
    object_columns = df.select_dtypes(include=["object"]).columns
    for column in object_columns:
        df[column] = df[column].astype("category")
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flights_path = DATA_DIR / "flights.csv"
    airlines_path = DATA_DIR / "airlines.csv"
    airports_path = DATA_DIR / "airports.csv"

    flights_sample = reservoir_sample_csv(
        csv_path=flights_path,
        sample_size=SAMPLE_SIZE,
        chunk_size=CHUNK_SIZE,
        random_seed=RANDOM_SEED,
    )

    airlines = pd.read_csv(airlines_path, low_memory=False).rename(
        columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"}
    )

    airports = pd.read_csv(airports_path, low_memory=False)

    origin_airports = airports.rename(
        columns={
            "IATA_CODE": "ORIGIN_AIRPORT",
            "AIRPORT": "ORIGIN_AIRPORT_NAME",
            "CITY": "ORIGIN_CITY",
            "STATE": "ORIGIN_STATE",
            "COUNTRY": "ORIGIN_COUNTRY",
            "LATITUDE": "ORIGIN_LATITUDE",
            "LONGITUDE": "ORIGIN_LONGITUDE",
        }
    )

    destination_airports = airports.rename(
        columns={
            "IATA_CODE": "DESTINATION_AIRPORT",
            "AIRPORT": "DESTINATION_AIRPORT_NAME",
            "CITY": "DESTINATION_CITY",
            "STATE": "DESTINATION_STATE",
            "COUNTRY": "DESTINATION_COUNTRY",
            "LATITUDE": "DESTINATION_LATITUDE",
            "LONGITUDE": "DESTINATION_LONGITUDE",
        }
    )

    merged = flights_sample.merge(airlines, on="AIRLINE", how="left")
    merged = merged.merge(origin_airports, on="ORIGIN_AIRPORT", how="left")
    merged = merged.merge(destination_airports, on="DESTINATION_AIRPORT", how="left")

    merged["ARRIVAL_DELAY_MISSING"] = merged["ARRIVAL_DELAY"].isna().astype("int8")

    missing_arrival_delay = int(merged["ARRIVAL_DELAY_MISSING"].sum())
    print(f"Missing ARRIVAL_DELAY rows before handling: {missing_arrival_delay:,}")

    cancellation_rate_among_missing = merged.loc[merged["ARRIVAL_DELAY_MISSING"] == 1, "CANCELLED"].mean()
    diverted_rate_among_missing = merged.loc[merged["ARRIVAL_DELAY_MISSING"] == 1, "DIVERTED"].mean()
    print(f"Cancelled rate among missing ARRIVAL_DELAY rows: {cancellation_rate_among_missing:.4f}")
    print(f"Diverted rate among missing ARRIVAL_DELAY rows: {diverted_rate_among_missing:.4f}")

    # ARRIVAL_DELAY is a target-like outcome and is structurally undefined for many cancelled/diverted flights.
    # A statistically sound approach is to preserve a missingness indicator and exclude undefined targets from
    # the final supervised-learning-ready table instead of imputing artificial arrival delays.
    model_ready = merged.loc[merged["ARRIVAL_DELAY"].notna()].copy()

    model_ready = cast_object_columns_to_category(model_ready)

    print(f"Rows after removing structurally missing ARRIVAL_DELAY targets: {len(model_ready):,}")
    print(model_ready.info(memory_usage="deep"))

    output_path = OUTPUT_DIR / "flights_sampled_joined_model_ready.pkl"
    model_ready.to_pickle(output_path)
    print(f"Saved processed dataset to: {output_path}")


if __name__ == "__main__":
    main()
