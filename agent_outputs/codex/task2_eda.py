from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import beta


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "../../data/flights.csv"
OUTPUT_DIR = SCRIPT_DIR
PLOT_PATH = OUTPUT_DIR / "eda_worst_offenders.png"

SEVERE_DELAY_MINUTES = 60
MIN_COMPLETED_FLIGHTS = 1000
TOP_N = 10


def main() -> None:
    usecols = ["AIRLINE", "ARRIVAL_DELAY", "CANCELLED", "DIVERTED"]
    flights = pd.read_csv(DATA_PATH, usecols=usecols, low_memory=False)

    completed = flights.loc[
        (flights["CANCELLED"] == 0)
        & (flights["DIVERTED"] == 0)
        & (flights["ARRIVAL_DELAY"].notna())
    ].copy()
    completed["severely_delayed"] = (completed["ARRIVAL_DELAY"] >= SEVERE_DELAY_MINUTES).astype(int)

    airline_summary = (
        completed.groupby("AIRLINE", observed=False)["severely_delayed"]
        .agg(completed_flights="size", severe_delays="sum")
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
    airline_summary["posterior_mean"] = (
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
        airline_summary.sort_values(["posterior_mean", "completed_flights"], ascending=[False, False])
        .head(TOP_N)
        .sort_values("posterior_mean", ascending=True)
        .reset_index(drop=True)
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 8))

    error_low = top_offenders["posterior_mean"] - top_offenders["ci_lower"]
    error_high = top_offenders["ci_upper"] - top_offenders["posterior_mean"]

    scatter = ax.scatter(
        top_offenders["posterior_mean"] * 100,
        top_offenders["AIRLINE"],
        s=top_offenders["completed_flights"] / 120,
        c=top_offenders["completed_flights"],
        cmap="Reds",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.errorbar(
        top_offenders["posterior_mean"] * 100,
        top_offenders["AIRLINE"],
        xerr=[error_low * 100, error_high * 100],
        fmt="none",
        ecolor="#444444",
        elinewidth=2,
        capsize=4,
        zorder=2,
    )

    for _, row in top_offenders.iterrows():
        ax.text(
            row["posterior_mean"] * 100 + 0.15,
            row["AIRLINE"],
            f"{row['completed_flights']:,} flights",
            va="center",
            fontsize=10,
            color="#333333",
        )

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Completed flights in analysis sample")

    ax.set_title(
        "Worst Airline Offenders by Risk-Adjusted Severe Delay Rate\n"
        f"Posterior probability of arriving {SEVERE_DELAY_MINUTES}+ minutes late"
    )
    ax.set_xlabel("Empirical-Bayes severe delay rate (%) with 95% credible interval")
    ax.set_ylabel("Airline")
    ax.set_xlim(left=max(0, top_offenders["ci_lower"].min() * 100 - 0.5))

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    worst = top_offenders.iloc[-1]
    best_of_top = top_offenders.iloc[0]
    print(
        f"Based on the plot, airline {worst['AIRLINE']} is the clearest booking risk because it has the highest "
        f"risk-adjusted probability of severe 60+ minute arrival delays, even after stabilizing for sample size. "
        f"For client itineraries, prioritize carriers below that top-risk tier and especially avoid booking time-sensitive "
        f"trips on airlines clustered near {worst['AIRLINE']} rather than lower-risk carriers such as {best_of_top['AIRLINE']}. "
        "This ranking is volume-aware, so it reflects persistent performance problems rather than one-off bad outcomes."
    )


if __name__ == "__main__":
    main()
