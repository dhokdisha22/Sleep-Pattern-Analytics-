"""
data_cleaning.py
Sleep Pattern Analysis — Data Cleaning Module

Implements FR-01, FR-02, FR-03, FR-04 from the FRD:
- Data loading
- Data validation (required columns, dtypes, missing/duplicate/invalid values)
- Data cleaning (missing values, duplicates)
- Sleep duration calculation from bedtime/wake_time
"""

import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "date", "bedtime", "wake_time", "sleep_duration",
    "age", "stress_level", "physical_activity_min", "sleep_quality",
]


def load_data(path: str) -> pd.DataFrame:
    """Load a sleep dataset from a CSV file."""
    df = pd.read_csv(path)
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Check required columns, dtypes, missing values, duplicates, and
    invalid values. Returns a report dict (does not mutate df).
    """
    report = {}

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report["missing_columns"] = missing_cols

    report["missing_values"] = df.isna().sum().to_dict()
    report["duplicate_rows"] = int(df.duplicated().sum())

    invalid_duration = 0
    if "sleep_duration" in df.columns:
        invalid_duration = int(((df["sleep_duration"] < 0) | (df["sleep_duration"] > 24)).sum())
    report["invalid_sleep_duration_rows"] = invalid_duration

    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Parse date column
    - Remove exact duplicate rows
    - Fix invalid sleep_duration values (negative / >24h) -> NaN, then impute
    - Impute missing numeric values with column median
    - Recalculate sleep_duration from bedtime/wake_time where possible
    """
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.drop_duplicates().reset_index(drop=True)

    if "sleep_duration" in df.columns:
        df.loc[(df["sleep_duration"] < 0) | (df["sleep_duration"] > 24), "sleep_duration"] = np.nan

    # Recalculate duration from bedtime/wake_time when both are present and duration is missing
    if {"bedtime", "wake_time", "sleep_duration"}.issubset(df.columns):
        recalced = df.apply(
            lambda r: calculate_sleep_duration(r["bedtime"], r["wake_time"])
            if pd.isna(r["sleep_duration"]) else r["sleep_duration"],
            axis=1,
        )
        df["sleep_duration"] = recalced

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    if "sleep_quality" in df.columns:
        df = df.dropna(subset=["sleep_quality"])

    return df.reset_index(drop=True)


def calculate_sleep_duration(bedtime: str, wake_time: str) -> float:
    """
    Calculate sleep duration (hours) from HH:MM bedtime and wake_time strings,
    correctly handling overnight sleep (bedtime later than wake_time).
    """
    try:
        b_h, b_m = map(int, str(bedtime).split(":"))
        w_h, w_m = map(int, str(wake_time).split(":"))
        bed_minutes = b_h * 60 + b_m
        wake_minutes = w_h * 60 + w_m
        diff = (wake_minutes - bed_minutes) % (24 * 60)
        return round(diff / 60, 2)
    except Exception:
        return np.nan


if __name__ == "__main__":
    raw = load_data("../data/raw/sleep_dataset.csv")
    print("Validation report:", validate_data(raw))
    cleaned = clean_data(raw)
    cleaned.to_csv("../data/processed/cleaned_sleep_data.csv", index=False)
    print(f"Cleaned data saved: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
