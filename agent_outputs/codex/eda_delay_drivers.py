from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path.cwd()
OUTPUT_DIR = BASE_DIR / "agent_outputs" / "codex"
DATA_PATH = OUTPUT_DIR / "flights_sampled_joined_model_ready.pkl"


def hhmm_to_hour(value):
    if pd.isna(value):
        return pd.NA
    value_str = str(value).split(".")[0].zfill(4)
    return int(value_str[:2])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_pickle(DATA_PATH)

    df = df.copy()
    df["delayed_15"] = (df["ARRIVAL_DELAY"] > 15).astype("int8")
    df["scheduled_departure_hour"] = df["SCHEDULED_DEPARTURE"].apply(hhmm_to_hour).astype("Int64")
    df["route"] = df["ORIGIN_AIRPORT"].astype(str) + "->" + df["DESTINATION_AIRPORT"].astype(str)

    advance_features = [
        "YEAR",
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "AIRLINE_NAME",
        "FLIGHT_NUMBER",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "ORIGIN_CITY",
        "DESTINATION_CITY",
        "ORIGIN_STATE",
        "DESTINATION_STATE",
        "DISTANCE",
        "SCHEDULED_DEPARTURE",
        "scheduled_departure_hour",
        "SCHEDULED_TIME",
        "route",
        "delayed_15",
    ]
    analysis_df = df[advance_features].copy()

    sns.set_theme(style="whitegrid", context="talk")

    hour_day = (
        analysis_df.dropna(subset=["scheduled_departure_hour"])
        .groupby(["DAY_OF_WEEK", "scheduled_departure_hour"], observed=False)["delayed_15"]
        .mean()
        .reset_index()
    )
    heatmap_data = hour_day.pivot(index="DAY_OF_WEEK", columns="scheduled_departure_hour", values="delayed_15")
    plt.figure(figsize=(16, 6))
    ax = sns.heatmap(heatmap_data, cmap="mako", linewidths=0.3, cbar_kws={"label": "Share delayed >15 min"})
    ax.set_title("Delay Risk by Scheduled Departure Hour and Day of Week")
    ax.set_xlabel("Scheduled Departure Hour")
    ax.set_ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_1_delay_heatmap_hour_day.png", dpi=300, bbox_inches="tight")
    plt.close()

    route_summary = (
        analysis_df.groupby(["route", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], observed=False)
        .agg(
            flights=("delayed_15", "size"),
            delay_rate=("delayed_15", "mean"),
            median_distance=("DISTANCE", "median"),
            median_scheduled_time=("SCHEDULED_TIME", "median"),
        )
        .reset_index()
    )
    route_summary = route_summary.loc[route_summary["flights"] >= 250].copy()
    route_summary = route_summary.nlargest(40, "delay_rate").sort_values("flights", ascending=False)
    plt.figure(figsize=(15, 10))
    ax = sns.scatterplot(
        data=route_summary,
        x="flights",
        y="delay_rate",
        size="median_distance",
        hue="median_scheduled_time",
        palette="viridis",
        sizes=(80, 800),
        alpha=0.8,
        edgecolor="black",
    )
    for _, row in route_summary.iterrows():
        ax.text(row["flights"] + 5, row["delay_rate"] + 0.001, row["route"], fontsize=8)
    ax.set_title("High-Risk Routes: Delay Rate vs Volume")
    ax.set_xlabel("Sample Flight Count on Route")
    ax.set_ylabel("Share Delayed >15 min")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_2_route_risk_frontier.png", dpi=300, bbox_inches="tight")
    plt.close()

    top_airlines = (
        analysis_df.groupby("AIRLINE_NAME", observed=False)["delayed_15"]
        .size()
        .sort_values(ascending=False)
        .head(8)
        .index
    )
    airline_month = (
        analysis_df.loc[analysis_df["AIRLINE_NAME"].isin(top_airlines)]
        .groupby(["MONTH", "AIRLINE_NAME"], observed=False)["delayed_15"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(15, 8))
    ax = sns.lineplot(
        data=airline_month,
        x="MONTH",
        y="delayed_15",
        hue="AIRLINE_NAME",
        marker="o",
        linewidth=2.5,
    )
    ax.set_title("Seasonal Delay Exposure for the Largest Airlines")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share Delayed >15 min")
    ax.set_xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plot_3_airline_seasonality.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

'''
Top business insight:
The strongest 24-hour-ahead delay signals are structural rather than operational: scheduled departure bank,
route, and carrier-season combination. Delay risk concentrates in later departure waves, on a small set of
high-volume routes with elevated baseline congestion, and in airline-month pairings that show consistent
seasonal stress. That means the most actionable pre-departure intervention is not flight-by-flight reactive
triage using day-of-operation fields, but proactive capacity planning and customer messaging targeted at the
highest-risk schedule windows and route-carrier combinations before the travel day begins.
'''
