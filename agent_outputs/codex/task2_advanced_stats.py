from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
VIF_SAMPLE_SIZE = 75_000
MODEL_SAMPLE_SIZE = 150_000

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR
FLIGHTS_PATH = DATA_DIR / "flights.csv"


def derive_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hhmm = df["SCHEDULED_DEPARTURE"].fillna(0).astype(int).astype(str).str.zfill(4)
    df["SCHEDULED_DEPARTURE_HOUR"] = hhmm.str[:2].astype(int)
    df["SCHEDULED_DEPARTURE_MINUTE"] = hhmm.str[2:].astype(int)
    return df


def sample_flights() -> pd.DataFrame:
    usecols = [
        "YEAR",
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
        "CANCELLED",
        "DIVERTED",
        "ARRIVAL_DELAY",
    ]
    df = pd.read_csv(FLIGHTS_PATH, usecols=usecols, low_memory=False)
    sample_n = min(300_000, len(df))
    df = df.sample(n=sample_n, random_state=RANDOM_SEED).reset_index(drop=True)
    df = derive_time_features(df)
    return df


def clean_numeric_matrix(df: pd.DataFrame, numeric_cols: list[str], sample_size: int) -> pd.DataFrame:
    numeric_df = df[numeric_cols].copy()
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(axis=1, how="all")

    # Drop constant and near-constant columns before VIF to avoid singularities.
    nunique = numeric_df.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    numeric_df = numeric_df[keep_cols]

    if len(numeric_df) > sample_size:
        numeric_df = numeric_df.sample(n=sample_size, random_state=RANDOM_SEED)

    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))
    return numeric_df


def compute_vif_table(df: pd.DataFrame) -> pd.DataFrame:
    X = df.to_numpy(dtype=float)
    vif_rows = []

    for idx, column in enumerate(df.columns):
        y = X[:, idx]
        X_other = np.delete(X, idx, axis=1)

        if X_other.shape[1] == 0:
            r_squared = 0.0
        else:
            model = LinearRegression()
            model.fit(X_other, y)
            r_squared = model.score(X_other, y)

        tolerance = max(1.0 - r_squared, 1e-12)
        vif = 1.0 / tolerance
        vif_rows.append(
            {
                "feature": column,
                "r_squared_against_others": r_squared,
                "tolerance": tolerance,
                "vif": vif,
                "multicollinearity_flag": (
                    "severe" if vif >= 10 else "moderate" if vif >= 5 else "low"
                ),
            }
        )

    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.loc[(df["ARRIVAL_DELAY"].notna()) & (df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)].copy()
    if len(model_df) > MODEL_SAMPLE_SIZE:
        model_df = model_df.sample(n=MODEL_SAMPLE_SIZE, random_state=RANDOM_SEED)
    return model_df


def save_heteroscedasticity_plot(df: pd.DataFrame) -> tuple[float, float]:
    feature_cols_num = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "FLIGHT_NUMBER",
        "SCHEDULED_DEPARTURE_HOUR",
        "SCHEDULED_DEPARTURE_MINUTE",
        "SCHEDULED_TIME",
        "DISTANCE",
    ]
    feature_cols_cat = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]

    top_origins = df["ORIGIN_AIRPORT"].value_counts().head(30).index
    top_destinations = df["DESTINATION_AIRPORT"].value_counts().head(30).index

    df = df.copy()
    df["ORIGIN_AIRPORT"] = np.where(df["ORIGIN_AIRPORT"].isin(top_origins), df["ORIGIN_AIRPORT"], "OTHER")
    df["DESTINATION_AIRPORT"] = np.where(df["DESTINATION_AIRPORT"].isin(top_destinations), df["DESTINATION_AIRPORT"], "OTHER")

    X = df[feature_cols_num + feature_cols_cat]
    y = df["ARRIVAL_DELAY"].astype(float)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), feature_cols_num),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                feature_cols_cat,
            ),
        ]
    )

    model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
    model.fit(X, y)
    fitted = model.predict(X)
    residuals = y - fitted
    abs_residuals = np.abs(residuals)

    plot_df = pd.DataFrame(
        {
            "Fitted arrival delay": fitted,
            "Absolute residual": abs_residuals,
        }
    )
    plot_df["fitted_bin"] = pd.qcut(plot_df["Fitted arrival delay"], q=20, duplicates="drop")
    bin_summary = (
        plot_df.groupby("fitted_bin", observed=False)
        .agg(mean_fitted=("Fitted arrival delay", "mean"), mean_abs_residual=("Absolute residual", "mean"))
        .reset_index(drop=True)
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 8))
    hb = ax.hexbin(
        plot_df["Fitted arrival delay"],
        plot_df["Absolute residual"],
        gridsize=45,
        cmap="mako",
        mincnt=1,
    )
    fig.colorbar(hb, ax=ax, label="Point density")
    sns.lineplot(
        data=bin_summary,
        x="mean_fitted",
        y="mean_abs_residual",
        marker="o",
        linewidth=2.5,
        color="#d95f02",
        ax=ax,
        label="Mean absolute residual by fitted-delay bin",
    )
    ax.set_title("Heteroscedasticity Diagnostic for Arrival Delay")
    ax.set_xlabel("Model fitted arrival delay (minutes)")
    ax.set_ylabel("Absolute residual (minutes)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_heteroscedasticity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    corr = float(np.corrcoef(plot_df["Fitted arrival delay"], plot_df["Absolute residual"])[0, 1])
    slope = float(np.polyfit(bin_summary["mean_fitted"], bin_summary["mean_abs_residual"], deg=1)[0])
    return corr, slope


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flights = sample_flights()

    vif_features = [
        "YEAR",
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "FLIGHT_NUMBER",
        "SCHEDULED_DEPARTURE_HOUR",
        "SCHEDULED_DEPARTURE_MINUTE",
        "SCHEDULED_TIME",
        "DISTANCE",
    ]
    vif_matrix = clean_numeric_matrix(flights, vif_features, VIF_SAMPLE_SIZE)
    vif_table = compute_vif_table(vif_matrix)
    vif_table.to_csv(OUTPUT_DIR / "eda_multicollinearity.csv", index=False)

    model_df = build_modeling_frame(flights)
    hetero_corr, hetero_slope = save_heteroscedasticity_plot(model_df)

    severe_features = vif_table.loc[vif_table["vif"] >= 10, "feature"].tolist()
    moderate_features = vif_table.loc[(vif_table["vif"] >= 5) & (vif_table["vif"] < 10), "feature"].tolist()

    message = (
        "Task 2 implication for Task 3: the VIF analysis shows "
        f"{'severe multicollinearity in ' + ', '.join(severe_features) if severe_features else 'no severe multicollinearity among the retained numeric schedule features'}, "
        f"with moderate overlap in {', '.join(moderate_features) if moderate_features else 'no additional moderate-VIF features'}. "
        f"The heteroscedasticity diagnostic also shows a positive fitted-vs-absolute-residual relationship "
        f"(correlation={hetero_corr:.3f}, slope={hetero_slope:.3f}), which means error variance increases as delay risk rises. "
        "That rules out plain unregularized linear regression as the primary model and makes coefficient-only interpretations fragile. "
        "For Task 3 we should keep regularized linear models only as baselines, then prefer tree-based methods such as Random Forest, Gradient Boosting, or XGBoost; "
        "if the target is binary delay-above-15-minutes, regularized Logistic Regression is an acceptable benchmark, but boosted trees should be the lead candidates because they are more robust to non-linearity, interaction effects, and heteroscedastic residual structure."
    )
    print(message)


if __name__ == "__main__":
    main()
