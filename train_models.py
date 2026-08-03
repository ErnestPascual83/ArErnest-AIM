"""
train_models.py
----------------
Step 2 of the pipeline: train a classifier (is this restaurant a "High"
popularity performer?) and a regressor (predict its annual demand score),
then save both models + the encoder as .pkl files the Streamlit app loads.

IMPORTANT — leakage check done during EDA:
`annual_demand_proxy_score_0_100` is very likely the source metric that
`popularity_class` was thresholded from (High avg ~88, Medium ~78, Low ~70,
correlation 0.67-0.81 with rating/reviews). So:
  - The CLASSIFIER never sees annual_demand_proxy_score_0_100 as a feature.
  - The REGRESSOR never sees popularity_class / is_high_popularity as a feature.
Both only use: cuisine_group, district, weighted_public_rating, total_public_reviews.

Run:
    python src/train_models.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/restaurants_clean.csv")
MODELS_DIR = Path("models")

NUMERIC_FEATURES = ["weighted_public_rating", "total_public_reviews"]
CATEGORICAL_FEATURES = ["cuisine_group", "district"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="passthrough",  # numeric features pass through unchanged
    )


def train_classifier(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X = df[FEATURES]
    y = df["is_high_popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300, max_depth=5, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "f1": round(f1_score(y_test, preds), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "n_test": len(y_test),
    }
    return pipe, metrics


def train_regressor(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X = df[FEATURES]
    y = df["annual_demand_proxy_score_0_100"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline(
        steps=[
            ("prep", build_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=5, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = {
        "mae": round(mean_absolute_error(y_test, preds), 2),
        "r2": round(r2_score(y_test, preds), 3),
        "n_test": len(y_test),
    }
    return pipe, metrics


def main():
    df = pd.read_csv(DATA_PATH)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    clf_pipe, clf_metrics = train_classifier(df)
    reg_pipe, reg_metrics = train_regressor(df)

    joblib.dump(clf_pipe, MODELS_DIR / "classifier_pipeline.pkl")
    joblib.dump(reg_pipe, MODELS_DIR / "regressor_pipeline.pkl")

    # Save the option lists the Streamlit UI needs for its dropdowns.
    options = {
        "cuisine_group": sorted(df["cuisine_group"].unique().tolist()),
        "district": sorted(df["district"].unique().tolist()),
    }
    with open(MODELS_DIR / "ui_options.json", "w") as f:
        json.dump(options, f, indent=2)

    metrics = {"classifier": clf_metrics, "regressor": reg_metrics}
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Classifier metrics:", clf_metrics)
    print("Regressor metrics:", reg_metrics)
    print(f"\nSaved models + metadata -> {MODELS_DIR}/")


if __name__ == "__main__":
    main()
