import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_SEED = 42
MAX_SAMPLE_SIZE = 70_000

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "../../data/flights.csv"
OUTPUT_DIR = SCRIPT_DIR

METRICS_PATH = OUTPUT_DIR / "task3_metrics.csv"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "task3_feature_importance.png"
MODEL_PATH = OUTPUT_DIR / "task3_baseline.pkl"


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series]:
    usecols = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "FLIGHT_NUMBER",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE",
        "DEPARTURE_DELAY",
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
    df["SCHEDULED_DEPARTURE"] = df["SCHEDULED_DEPARTURE"].fillna(0).astype(int)
    df["SCHEDULED_DEPARTURE_HOUR"] = (
        df["SCHEDULED_DEPARTURE"].astype(str).str.zfill(4).str[:2].astype(int)
    )
    df["SCHEDULED_DEPARTURE_MINUTE"] = (
        df["SCHEDULED_DEPARTURE"].astype(str).str.zfill(4).str[2:].astype(int)
    )

    # Reduce cardinality while preserving useful route structure.
    top_origins = df["ORIGIN_AIRPORT"].value_counts().head(20).index
    top_destinations = df["DESTINATION_AIRPORT"].value_counts().head(20).index
    top_flight_numbers = df["FLIGHT_NUMBER"].value_counts().head(30).index

    df["ORIGIN_AIRPORT"] = df["ORIGIN_AIRPORT"].where(df["ORIGIN_AIRPORT"].isin(top_origins), "OTHER")
    df["DESTINATION_AIRPORT"] = df["DESTINATION_AIRPORT"].where(df["DESTINATION_AIRPORT"].isin(top_destinations), "OTHER")
    df["FLIGHT_NUMBER"] = df["FLIGHT_NUMBER"].where(df["FLIGHT_NUMBER"].isin(top_flight_numbers), -1)

    feature_cols = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "FLIGHT_NUMBER",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE_HOUR",
        "SCHEDULED_DEPARTURE_MINUTE",
        "DEPARTURE_DELAY",
        "SCHEDULED_TIME",
        "DISTANCE",
    ]
    X = df[feature_cols]
    y = df["TARGET_DELAY_15"]
    return X, y


def build_pipeline() -> Pipeline:
    numeric_features = [
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "FLIGHT_NUMBER",
        "SCHEDULED_DEPARTURE_HOUR",
        "SCHEDULED_DEPARTURE_MINUTE",
        "DEPARTURE_DELAY",
        "SCHEDULED_TIME",
        "DISTANCE",
    ]
    categorical_features = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def save_metrics(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> pd.DataFrame:
    metrics = pd.DataFrame(
        {
            "metric": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
            "value": [
                accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred, zero_division=0),
                recall_score(y_true, y_pred, zero_division=0),
                f1_score(y_true, y_pred, zero_division=0),
                roc_auc_score(y_true, y_prob),
            ],
        }
    )
    metrics.to_csv(METRICS_PATH, index=False)
    return metrics


def save_feature_importance_plot(pipeline: Pipeline) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(10)
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=importance, x="importance", y="feature", palette="crest", ax=ax)
    ax.set_title("Top 10 Feature Importances for Flight Delay Prediction")
    ax.set_xlabel("Random Forest feature importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return importance


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / ".mplconfig").mkdir(exist_ok=True)

    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = save_metrics(y_test, y_pred, y_prob)
    top_features = save_feature_importance_plot(pipeline)
    joblib.dump(pipeline, MODEL_PATH)

    print(
        f"The baseline Random Forest model delivered an ROC-AUC of {metrics.loc[metrics['metric'] == 'roc_auc', 'value'].iloc[0]:.3f} "
        f"and an F1-score of {metrics.loc[metrics['metric'] == 'f1_score', 'value'].iloc[0]:.3f}, giving us a credible first benchmark for delay prediction. "
        f"The strongest drivers were led by {top_features.iloc[-1]['feature']}, {top_features.iloc[-2]['feature']}, and {top_features.iloc[-3]['feature']}, "
        "which tells the business that route structure, schedule timing, and departure-side operational signals are doing most of the predictive work."
    )


if __name__ == "__main__":
    main()
