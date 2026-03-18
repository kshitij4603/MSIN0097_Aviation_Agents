import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "../../data/flights.csv"
OUTPUT_DIR = SCRIPT_DIR


def load_completed_flights() -> pd.DataFrame:
    usecols = [
        "MONTH",
        "DAY_OF_WEEK",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE",
        "DISTANCE",
        "ARRIVAL_DELAY",
        "CANCELLED",
        "DIVERTED",
    ]
    flights = pd.read_csv(DATA_PATH, usecols=usecols, low_memory=False)
    completed = flights.loc[
        (flights["CANCELLED"] == 0)
        & (flights["DIVERTED"] == 0)
        & (flights["ARRIVAL_DELAY"].notna())
    ].copy()
    completed["delay_15_plus"] = (completed["ARRIVAL_DELAY"] >= 15).astype(int)
    completed["departure_hour"] = (
        completed["SCHEDULED_DEPARTURE"].fillna(0).astype(int).astype(str).str.zfill(4).str[:2].astype(int)
    )
    return completed


def save_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overview = pd.DataFrame(
        {
            "metric": [
                "completed_flights",
                "mean_arrival_delay",
                "median_arrival_delay",
                "p90_arrival_delay",
                "share_15_plus_delay_pct",
            ],
            "value": [
                len(df),
                df["ARRIVAL_DELAY"].mean(),
                df["ARRIVAL_DELAY"].median(),
                df["ARRIVAL_DELAY"].quantile(0.90),
                df["delay_15_plus"].mean() * 100,
            ],
        }
    )
    overview.to_csv(OUTPUT_DIR / "basic_eda_overview.csv", index=False)

    airline_table = (
        df.groupby("AIRLINE", observed=False)
        .agg(
            completed_flights=("ARRIVAL_DELAY", "size"),
            mean_arrival_delay=("ARRIVAL_DELAY", "mean"),
            median_arrival_delay=("ARRIVAL_DELAY", "median"),
            delay_15_plus_rate_pct=("delay_15_plus", lambda s: s.mean() * 100),
        )
        .sort_values(["delay_15_plus_rate_pct", "mean_arrival_delay"], ascending=[False, False])
        .round(2)
        .reset_index()
    )
    airline_table.to_csv(OUTPUT_DIR / "basic_eda_airline_summary.csv", index=False)

    airport_table = (
        df.groupby("ORIGIN_AIRPORT", observed=False)
        .agg(
            completed_flights=("ARRIVAL_DELAY", "size"),
            mean_arrival_delay=("ARRIVAL_DELAY", "mean"),
            delay_15_plus_rate_pct=("delay_15_plus", lambda s: s.mean() * 100),
        )
        .query("completed_flights >= 5000")
        .sort_values(["delay_15_plus_rate_pct", "completed_flights"], ascending=[False, False])
        .head(20)
        .round(2)
        .reset_index()
    )
    airport_table.to_csv(OUTPUT_DIR / "basic_eda_origin_airport_summary.csv", index=False)

    return overview, airline_table, airport_table


def save_plots(df: pd.DataFrame, airline_table: pd.DataFrame, airport_table: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    hourly = (
        df.groupby("departure_hour", observed=False)
        .agg(delay_15_plus_rate=("delay_15_plus", "mean"), mean_arrival_delay=("ARRIVAL_DELAY", "mean"))
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=hourly, x="departure_hour", y="delay_15_plus_rate", marker="o", linewidth=2.5, ax=ax1, color="#b22222")
    ax1.set_title("Delay Risk Rises Across the Departure Day")
    ax1.set_xlabel("Scheduled departure hour")
    ax1.set_ylabel("Share delayed 15+ minutes")
    ax1.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "basic_eda_departure_hour.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    top_airlines = airline_table.head(10).sort_values("delay_15_plus_rate_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=top_airlines, x="delay_15_plus_rate_pct", y="AIRLINE", palette="Reds_r", ax=ax)
    ax.set_title("Highest-Risk Airlines by Share of 15+ Minute Arrival Delays")
    ax.set_xlabel("Flights delayed 15+ minutes (%)")
    ax.set_ylabel("Airline")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "basic_eda_airline_risk.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    top_airports = airport_table.head(10).sort_values("delay_15_plus_rate_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=top_airports, x="delay_15_plus_rate_pct", y="ORIGIN_AIRPORT", palette="flare", ax=ax)
    ax.set_title("Origin Airports with the Highest Delay Risk")
    ax.set_xlabel("Flights delayed 15+ minutes (%)")
    ax.set_ylabel("Origin airport")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "basic_eda_origin_airport_risk.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / ".mplconfig").mkdir(exist_ok=True)

    completed = load_completed_flights()
    overview, airline_table, airport_table = save_tables(completed)
    save_plots(completed, airline_table, airport_table)

    top_airline = airline_table.iloc[0]["AIRLINE"]
    top_airport = airport_table.iloc[0]["ORIGIN_AIRPORT"]
    high_risk_hour = (
        completed.groupby("departure_hour", observed=False)["delay_15_plus"].mean().sort_values(ascending=False).index[0]
    )
    print(
        f"The data show three clear business patterns: late-day departures carry the highest delay risk, with the worst hour in this sample occurring around {high_risk_hour}:00; airline performance is uneven, with {top_airline} sitting at the top of the risk ranking; and a small set of origin airports led by {top_airport} contributes disproportionately to late arrivals. "
        "For the travel agency, that means the safest booking strategy is to favor earlier departures, avoid the highest-risk carriers for time-sensitive trips, and treat itineraries leaving from the worst-performing origin airports as needing extra schedule buffer."
    )


if __name__ == "__main__":
    main()
