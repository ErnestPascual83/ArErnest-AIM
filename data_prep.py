"""
data_prep.py
------------
Step 1 of the pipeline: load the raw scraped dataset, clean it, engineer
the target columns, and save a modeling-ready CSV.

Run:
    python src/data_prep.py
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/Tropang_Foodie_Dataset_1.csv")
PROCESSED_PATH = Path("data/restaurants_clean.csv")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop exact duplicate restaurant records, if any.
    df = df.drop_duplicates(subset="restaurant_id")

    # 2. price_level_1_4 is missing for ~81% of rows (only 24/127 populated).
    #    That's too sparse to impute reliably or use as a model feature, so
    #    we keep it in the cleaned file for reference/reporting but exclude
    #    it from the feature set used to train models (see train_models.py).
    df["price_level_1_4"] = df["price_level_1_4"].astype("Int64")  # nullable int

    # 3. Normalize text fields.
    df["cuisine_type"] = df["cuisine_type"].str.strip()
    df["district"] = df["district"].str.strip()
    df["popularity_class"] = df["popularity_class"].str.strip()

    # 4. Collapse rare cuisines into "Other" so one-hot encoding doesn't
    #    explode into dozens of near-empty columns.
    cuisine_counts = df["cuisine_type"].value_counts()
    rare_cuisines = cuisine_counts[cuisine_counts < 4].index
    df["cuisine_group"] = df["cuisine_type"].where(
        ~df["cuisine_type"].isin(rare_cuisines), "Other"
    )

    # 5. Binary popularity target for classification.
    #    popularity_class has only 3 "Low" rows out of 127 — far too few
    #    for a model to learn a 3rd class reliably. We reframe the business
    #    question as "Is this restaurant a High performer?" (High vs Not-High)
    #    which is both more robust to train and more decision-useful.
    df["is_high_popularity"] = (df["popularity_class"] == "High").astype(int)

    return df


def main():
    df = load_raw()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns from {RAW_PATH}")

    clean_df = clean(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned dataset -> {PROCESSED_PATH} ({len(clean_df)} rows)")
    print("\nClass balance (is_high_popularity):")
    print(clean_df["is_high_popularity"].value_counts())


if __name__ == "__main__":
    main()
