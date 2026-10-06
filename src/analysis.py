"""
analysis.py
Sleep Pattern Analysis — Statistical Analysis Module

Implements FR-05, FR-06, FR-08 from the FRD:
- Sleep statistics (average, min, max, variation)
- Sleep pattern / consistency analysis
- Trend analysis over time
"""

import pandas as pd
import numpy as np


def compute_sleep_statistics(df: pd.DataFrame) -> dict:
    """Return summary statistics for sleep_duration."""
    s = df["sleep_duration"].dropna()
    return {
        "average_sleep_duration": round(s.mean(), 2),
        "min_sleep_duration": round(s.min(), 2),
        "max_sleep_duration": round(s.max(), 2),
        "std_sleep_duration": round(s.std(), 2),
        "median_sleep_duration": round(s.median(), 2),
        "total_records": int(len(df)),
    }


def analyze_consistency(df: pd.DataFrame) -> dict:
    """
    Analyze sleep consistency using the standard deviation of sleep_duration
    and bedtime. Lower std => more consistent schedule.
    """
    duration_std = df["sleep_duration"].std()

    bedtime_minutes = df["bedtime"].apply(_to_minutes)
    bedtime_std = bedtime_minutes.std()

    if duration_std < 0.75:
        consistency_label = "Highly Consistent"
    elif duration_std < 1.5:
        consistency_label = "Moderately Consistent"
    else:
        consistency_label = "Irregular"

    return {
        "sleep_duration_std": round(duration_std, 2),
        "bedtime_std_minutes": round(bedtime_std, 1),
        "consistency_label": consistency_label,
    }


def analyze_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a date-sorted dataframe with a rolling 7-day average of sleep
    duration for trend visualization.
    """
    trend_df = df.sort_values("date").copy()
    trend_df["rolling_avg_7d"] = (
        trend_df["sleep_duration"].rolling(window=7, min_periods=1).mean().round(2)
    )
    return trend_df[["date", "sleep_duration", "rolling_avg_7d"]]


def identify_irregular_days(df: pd.DataFrame, threshold_hours: float = 1.5) -> pd.DataFrame:
    """
    Flag days where sleep_duration deviates from the individual's mean by
    more than `threshold_hours`.
    """
    mean_duration = df["sleep_duration"].mean()
    df = df.copy()
    df["deviation"] = (df["sleep_duration"] - mean_duration).abs()
    df["is_irregular"] = df["deviation"] > threshold_hours
    return df[df["is_irregular"]][["date", "sleep_duration", "deviation"]]


def _to_minutes(t: str) -> float:
    try:
        h, m = map(int, str(t).split(":"))
        return h * 60 + m
    except Exception:
        return np.nan


if __name__ == "__main__":
    df = pd.read_csv("../data/processed/cleaned_sleep_data.csv", parse_dates=["date"])
    print(compute_sleep_statistics(df))
    print(analyze_consistency(df))
    print(identify_irregular_days(df).head())
