import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_SEED = 42
MAX_SAMPLE_SIZE = 50_000

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "../../data/flights.csv"
TASK4_METRICS_PATH = SCRIPT_DIR / "task4_optimized_metrics.csv"
OUTPUT_DIR = SCRIPT_DIR

METRICS_PATH = OUTPUT_DIR / "task5_honest_metrics.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "task5_honest_feature_importance.png"

LEAKY_FEATURES = ["DEPARTURE_DELAY", "DEPARTURE_DELAY_CLIPPED", "DEPARTURE_DELAY_15_PLUS"]


def load_clean_predeparture_data() -> tuple[pd.DataFrame, pd.Series]:
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
        "CANCELLED",
        "DIVERTED",
        "ARRIVAL_DELAY",
    ]
    df = pd.read_csv(DATA_PATH, usecols=usecols, low_memory=False)
    df = df.loc[
        (df["CANCELLED"] == 0)
        & (df["DIVERTED"] == 0)
        & (df["ARRIVAL_DELAY"].notna())
    ].copy()

    if len(df) > MAX_SAMPLE_SIZE:
        df = df.sample(n=MAX_SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

    df["TARGET_DELAY_15"] = (df["ARRIVAL_DELAY"] > 15).astype(int)

    sched = df["SCHEDULED_DEPARTURE"].fillna(0).astype(int).astype(str).str.zfill(4)
    df["SCHEDULED_DEPARTURE_HOUR"] = sched.str[:2].astype(int)
    df["SCHEDULED_DEPARTURE_MINUTE"] = sched.str[2:].astype(int)
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype(int)
    df["IS_PEAK_SUMMER"] = df["MONTH"].isin([6, 7, 8]).astype(int)
    df["DISTANCE_BIN"] = pd.cut(
        df["DISTANCE"],
        bins=[0, 500, 1000, 1500, 2500, np.inf],
        labels=["short", "short_mid", "mid", "long", "ultra_long"],
        include_lowest=True,
    ).astype(str)
    df["ROUTE"] = df["ORIGIN_AIRPORT"].astype(str) + "_" + df["DESTINATION_AIRPORT"].astype(str)
    df["AIRLINE_ROUTE"] = df["AIRLINE"].astype(str) + "_" + df["ROUTE"].astype(str)

    top_origins = df["ORIGIN_AIRPORT"].value_counts().head(25).index
    top_destinations = df["DESTINATION_AIRPORT"].value_counts().head(25).index
    top_routes = df["ROUTE"].value_counts().head(60).index
    top_airline_routes = df["AIRLINE_ROUTE"].value_counts().head(80).index
    top_flight_numbers = df["FLIGHT_NUMBER"].value_counts().head(40).index

    df["ORIGIN_AIRPORT"] = df["ORIGIN_AIRPORT"].where(df["ORIGIN_AIRPORT"].isin(top_origins), "OTHER")
    df["DESTINATION_AIRPORT"] = df["DESTINATION_AIRPORT"].where(df["DESTINATION_AIRPORT"].isin(top_destinations), "OTHER")
    df["ROUTE"] = df["ROUTE"].where(df["ROUTE"].isin(top_routes), "OTHER")
    df["AIRLINE_ROUTE"] = df["AIRLINE_ROUTE"].where(df["AIRLINE_ROUTE"].isin(top_airline_routes), "OTHER")
    df["FLIGHT_NUMBER"] = df["FLIGHT_NUMBER"].where(df["FLIGHT_NUMBER"].isin(top_flight_numbers), -1)

    feature_cols = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "FLIGHT_NUMBER",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "ROUTE",
        "AIRLINE_ROUTE",
        "SCHEDULED_DEPARTURE_HOUR",
        "SCHEDULED_DEPARTURE_MINUTE",
        "IS_WEEKEND",
        "IS_PEAK_SUMMER",
        "SCHEDULED_TIME",
        "DISTANCE",
        "DISTANCE_BIN",
    ]
    return df[feature_cols], df["TARGET_DELAY_15"]


def build_pipeline(params: dict) -> Pipeline:
    numeric_features = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "FLIGHT_NUMBER",
        "SCHEDULED_DEPARTURE_HOUR",
        "SCHEDULED_DEPARTURE_MINUTE",
        "IS_WEEKEND",
        "IS_PEAK_SUMMER",
        "SCHEDULED_TIME",
        "DISTANCE",
    ]
    categorical_features = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "ROUTE",
        "AIRLINE_ROUTE",
        "DISTANCE_BIN",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = HistGradientBoostingClassifier(random_state=RANDOM_SEED, **params)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / ".mplconfig").mkdir(exist_ok=True)

    task4_metrics = pd.read_csv(TASK4_METRICS_PATH)
    task4_auc = float(task4_metrics.loc[task4_metrics["model"] == "optimized_hgb", "roc_auc"].iloc[0])

    X, y = load_clean_predeparture_data()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    train_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    candidate_params = [
        {"learning_rate": 0.05, "max_iter": 120, "max_leaf_nodes": 31, "max_depth": 6, "min_samples_leaf": 20, "l2_regularization": 0.1},
        {"learning_rate": 0.08, "max_iter": 140, "max_leaf_nodes": 31, "max_depth": 8, "min_samples_leaf": 20, "l2_regularization": 0.0},
        {"learning_rate": 0.05, "max_iter": 180, "max_leaf_nodes": 63, "max_depth": None, "min_samples_leaf": 40, "l2_regularization": 0.5},
    ]

    best_f1 = -1.0
    best_params = None
    for params in candidate_params:
        pipeline = build_pipeline(params)
        pipeline.fit(X_train, y_train, model__sample_weight=train_weight)
        valid_prob = pipeline.predict_proba(X_valid)[:, 1]
        valid_pred = (valid_prob >= 0.5).astype(int)
        score = f1_score(y_valid, valid_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_params = params

    final_weight = compute_sample_weight(class_weight="balanced", y=y_train_full)
    best_pipeline = build_pipeline(best_params)
    best_pipeline.fit(X_train_full, y_train_full, model__sample_weight=final_weight)

    test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    honest_auc = roc_auc_score(y_test, test_prob)

    metrics = pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "recall", "f1_score", "roc_auc", "task4_reference_auc", "auc_drop_from_task4"],
            "value": [
                accuracy_score(y_test, test_pred),
                precision_score(y_test, test_pred, zero_division=0),
                recall_score(y_test, test_pred, zero_division=0),
                f1_score(y_test, test_pred, zero_division=0),
                honest_auc,
                task4_auc,
                task4_auc - honest_auc,
            ],
        }
    )
    metrics.to_csv(METRICS_PATH, index=False)

    sample_for_importance = X_test.sample(n=min(4000, len(X_test)), random_state=RANDOM_SEED)
    y_sample = y_test.loc[sample_for_importance.index]
    importance = permutation_importance(
        best_pipeline,
        sample_for_importance,
        y_sample,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    importance_df = (
        pd.DataFrame(
            {
                "feature": sample_for_importance.columns,
                "importance": importance.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="rocket", ax=ax)
    ax.set_title("Top 10 Honest Feature Importances After Leakage Audit")
    ax.set_xlabel("Permutation importance (ROC-AUC decrease)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    auc_drop = task4_auc - honest_auc
    print(
        f"**Audit Report** We dropped {', '.join(LEAKY_FEATURES)} because they are illegal for a pre-departure arrival-delay model: each one requires observing the flight after departure, which creates look-ahead leakage and artificially inflates performance. "
        f"After retraining the same boosted-tree family on a strictly pre-departure feature set, the true ROC-AUC is {honest_auc:.3f} versus the Task 4 ROC-AUC of {task4_auc:.3f}, an exact drop of {auc_drop:.3f}. "
        "That drop is the honest cost of removing leakage, and it gives the business a production-ready estimate of how well the model will perform when predictions must be made before the aircraft actually leaves the gate."
    )


if __name__ == "__main__":
    main()
