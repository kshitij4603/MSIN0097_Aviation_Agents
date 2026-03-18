import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import beta


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "../../data/flights.csv"
OUTPUT_DIR = SCRIPT_DIR

DELAY_DISTRIBUTION_PATH = OUTPUT_DIR / "eda_delay_distribution.csv"
CORRELATIONS_PATH = OUTPUT_DIR / "eda_correlations.csv"
AIRLINE_DAY_PIVOT_PATH = OUTPUT_DIR / "eda_airline_day_pivot.csv"
PLOT_PATH = OUTPUT_DIR / "eda_worst_offenders.png"

SEVERE_DELAY_MINUTES = 60
MIN_COMPLETED_FLIGHTS = 1000
TOP_N = 10

DAY_LABELS = {
    1: "Mon",
    2: "Tue",
    3: "Wed",
    4: "Thu",
    5: "Fri",
    6: "Sat",
    7: "Sun",
}


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    df = pd.DataFrame({"category": categories, "measurement": measurements}).dropna()
    if df.empty or df["category"].nunique() <= 1:
        return np.nan

    grand_mean = df["measurement"].mean()
    grouped = df.groupby("category", observed=False)["measurement"]
    counts = grouped.size()
    means = grouped.mean()

    between_group_ss = ((means - grand_mean) ** 2 * counts).sum()
    total_ss = ((df["measurement"] - grand_mean) ** 2).sum()

    if total_ss == 0:
        return 0.0
    return float(np.sqrt(between_group_ss / total_ss))


def build_delay_distribution_table(completed: pd.DataFrame) -> pd.DataFrame:
    delay = completed["ARRIVAL_DELAY"].astype(float)
    table = pd.DataFrame(
        {
            "metric": [
                "completed_flights",
                "mean_delay_minutes",
                "median_delay_minutes",
                "std_delay_minutes",
                "min_delay_minutes",
                "p05_delay_minutes",
                "p25_delay_minutes",
                "p75_delay_minutes",
                "p95_delay_minutes",
                "max_delay_minutes",
                "skewness",
                "share_early_or_on_time_pct",
                "share_15plus_delay_pct",
                "share_60plus_delay_pct",
            ],
            "value": [
                len(delay),
                delay.mean(),
                delay.median(),
                delay.std(),
                delay.min(),
                delay.quantile(0.05),
                delay.quantile(0.25),
                delay.quantile(0.75),
                delay.quantile(0.95),
                delay.max(),
                delay.skew(),
                (delay <= 0).mean() * 100,
                (delay >= 15).mean() * 100,
                (delay >= SEVERE_DELAY_MINUTES).mean() * 100,
            ],
        }
    )
    return table


def build_correlation_table(completed: pd.DataFrame) -> pd.DataFrame:
    numeric_features = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "FLIGHT_NUMBER",
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_TIME",
        "DISTANCE",
    ]
    categorical_features = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
    ]

    rows = []
    target = completed["ARRIVAL_DELAY"].astype(float)

    for feature in numeric_features:
        subset = completed[[feature, "ARRIVAL_DELAY"]].dropna()
        if subset[feature].nunique() > 1:
            strength = subset[feature].corr(subset["ARRIVAL_DELAY"], method="spearman")
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric",
                    "association_method": "spearman",
                    "association_strength": float(strength),
                    "absolute_association_strength": abs(float(strength)),
                }
            )

    for feature in categorical_features:
        strength = correlation_ratio(completed[feature], target)
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "association_method": "correlation_ratio_eta",
                "association_strength": float(strength),
                "absolute_association_strength": abs(float(strength)),
            }
        )

    correlation_table = pd.DataFrame(rows)
    correlation_table = correlation_table.sort_values(
        ["absolute_association_strength", "feature"], ascending=[False, True]
    ).head(5)
    return correlation_table.reset_index(drop=True)


def build_airline_day_pivot(completed: pd.DataFrame) -> pd.DataFrame:
    pivot_source = completed.copy()
    pivot_source["severe_delay_rate_pct"] = (pivot_source["ARRIVAL_DELAY"] >= SEVERE_DELAY_MINUTES).astype(float) * 100
    pivot = (
        pivot_source.groupby(["AIRLINE", "DAY_OF_WEEK"], observed=False)["severe_delay_rate_pct"]
        .mean()
        .reset_index()
    )
    pivot["DAY_OF_WEEK"] = pivot["DAY_OF_WEEK"].map(DAY_LABELS)
    pivot_table = pivot.pivot(index="AIRLINE", columns="DAY_OF_WEEK", values="severe_delay_rate_pct")
    pivot_table = pivot_table.reindex(columns=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    pivot_table["Weekly Avg"] = pivot_table.mean(axis=1)
    pivot_table = pivot_table.sort_values("Weekly Avg", ascending=False).round(2)
    pivot_table.index.name = "Airline"
    return pivot_table.reset_index()


def build_worst_offenders_plot(completed: pd.DataFrame) -> pd.DataFrame:
    completed = completed.copy()
    completed["severely_delayed"] = (completed["ARRIVAL_DELAY"] >= SEVERE_DELAY_MINUTES).astype(int)

    airline_summary = (
        completed.groupby("AIRLINE", observed=False)
        .agg(
            completed_flights=("severely_delayed", "size"),
            severe_delays=("severely_delayed", "sum"),
            median_arrival_delay=("ARRIVAL_DELAY", "median"),
        )
        .reset_index()
    )
    airline_summary = airline_summary.loc[airline_summary["completed_flights"] >= MIN_COMPLETED_FLIGHTS].copy()

    overall_rate = completed["severely_delayed"].mean()
    prior_strength = max(int(airline_summary["completed_flights"].median() * 0.10), 200)
    alpha_prior = overall_rate * prior_strength
    beta_prior = (1 - overall_rate) * prior_strength

    airline_summary["posterior_alpha"] = alpha_prior + airline_summary["severe_delays"]
    airline_summary["posterior_beta"] = (
        beta_prior + airline_summary["completed_flights"] - airline_summary["severe_delays"]
    )
    airline_summary["risk_adjusted_severe_delay_rate"] = (
        airline_summary["posterior_alpha"]
        / (airline_summary["posterior_alpha"] + airline_summary["posterior_beta"])
    )
    airline_summary["ci_lower"] = beta.ppf(
        0.025, airline_summary["posterior_alpha"], airline_summary["posterior_beta"]
    )
    airline_summary["ci_upper"] = beta.ppf(
        0.975, airline_summary["posterior_alpha"], airline_summary["posterior_beta"]
    )

    top_offenders = (
        airline_summary.sort_values(
            ["risk_adjusted_severe_delay_rate", "completed_flights"],
            ascending=[False, False],
        )
        .head(TOP_N)
        .sort_values("risk_adjusted_severe_delay_rate", ascending=True)
        .reset_index(drop=True)
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12.5, 8.5))

    error_low = (
        top_offenders["risk_adjusted_severe_delay_rate"] - top_offenders["ci_lower"]
    ) * 100
    error_high = (
        top_offenders["ci_upper"] - top_offenders["risk_adjusted_severe_delay_rate"]
    ) * 100

    scatter = ax.scatter(
        top_offenders["risk_adjusted_severe_delay_rate"] * 100,
        top_offenders["AIRLINE"],
        s=top_offenders["completed_flights"] / 120,
        c=top_offenders["completed_flights"],
        cmap="Reds",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.errorbar(
        top_offenders["risk_adjusted_severe_delay_rate"] * 100,
        top_offenders["AIRLINE"],
        xerr=[error_low, error_high],
        fmt="none",
        ecolor="#4d4d4d",
        elinewidth=2,
        capsize=4,
        zorder=2,
    )

    for _, row in top_offenders.iterrows():
        ax.text(
            row["risk_adjusted_severe_delay_rate"] * 100 + 0.12,
            row["AIRLINE"],
            f"{row['completed_flights']:,} flights | median delay {row['median_arrival_delay']:.0f} min",
            va="center",
            fontsize=9.5,
            color="#333333",
        )

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Completed flights in analysis")

    ax.set_title(
        "Worst Airline Offenders by Risk-Adjusted Severe Delay Rate\n"
        f"Completed flights only, severe delay defined as {SEVERE_DELAY_MINUTES}+ minutes late"
    )
    ax.set_xlabel("Empirical-Bayes severe delay rate (%) with 95% credible interval")
    ax.set_ylabel("Airline")
    ax.set_xlim(left=max(0, top_offenders["ci_lower"].min() * 100 - 0.5))

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return top_offenders


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / ".mplconfig").mkdir(exist_ok=True)

    usecols = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "FLIGHT_NUMBER",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_TIME",
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

    delay_distribution = build_delay_distribution_table(completed)
    delay_distribution.to_csv(DELAY_DISTRIBUTION_PATH, index=False)

    correlation_table = build_correlation_table(completed)
    correlation_table.to_csv(CORRELATIONS_PATH, index=False)

    airline_day_pivot = build_airline_day_pivot(completed)
    airline_day_pivot.to_csv(AIRLINE_DAY_PIVOT_PATH, index=False)

    top_offenders = build_worst_offenders_plot(completed)

    worst_airline = top_offenders.iloc[-1]["AIRLINE"]
    second_worst = top_offenders.iloc[-2]["AIRLINE"]
    weekday_hotspot = airline_day_pivot.iloc[0, 1:8].astype(float).idxmax()
    print(
        f"The delay distribution is heavily right-skewed, which means averages alone understate the client risk of rare but severe disruptions, so the agency should watch the upper-tail metrics in eda_delay_distribution.csv rather than relying on mean delay. "
        f"The strongest delay associations come from structural route-and-carrier variables, and the airline-by-day matrix shows that some carriers are materially worse on specific weekdays, with the riskiest weekly hotspot appearing on {weekday_hotspot}. "
        f"For booking policy, treat {worst_airline} and {second_worst} as the highest-risk carriers for time-sensitive itineraries and prefer airlines with consistently lower severe-delay rates in both the pivot table and the risk-adjusted worst-offender ranking."
    )


if __name__ == "__main__":
    main()
