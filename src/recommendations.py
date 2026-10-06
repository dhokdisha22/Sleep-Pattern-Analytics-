"""
recommendations.py
Sleep Pattern Analysis — Insights & Recommendations Module

Implements FR-11, FR-12 from the FRD:
- Simple data-based insights (low duration, inconsistent schedule, etc.)
- General, non-medical recommendations

Per TRD Section 13, this module must not present medical/diagnostic claims.
"""

import pandas as pd


def generate_insights(df: pd.DataFrame, stats: dict, consistency: dict) -> list:
    """Return a list of plain-language, data-based insight strings."""
    insights = []

    avg = stats["average_sleep_duration"]
    if avg < 6:
        insights.append(f"Average sleep duration is low ({avg} hrs) — below the commonly cited 7-hour range.")
    elif avg > 9:
        insights.append(f"Average sleep duration is on the higher side ({avg} hrs).")
    else:
        insights.append(f"Average sleep duration ({avg} hrs) falls within a typical range.")

    if consistency["consistency_label"] == "Irregular":
        insights.append("Sleep schedule shows high variability — bedtime and duration change significantly day to day.")
    elif consistency["consistency_label"] == "Moderately Consistent":
        insights.append("Sleep schedule is moderately consistent, with some fluctuation across days.")
    else:
        insights.append("Sleep schedule is highly consistent, which is generally associated with better sleep hygiene.")

    if "sleep_quality" in df.columns:
        poor_pct = (df["sleep_quality"] == "Poor").mean() * 100
        if poor_pct > 25:
            insights.append(f"{poor_pct:.1f}% of recorded nights were classified as 'Poor' sleep quality.")

    if "stress_level" in df.columns:
        high_stress_avg = df.loc[df["stress_level"] >= 7, "sleep_duration"].mean()
        low_stress_avg = df.loc[df["stress_level"] <= 4, "sleep_duration"].mean()
        if pd.notna(high_stress_avg) and pd.notna(low_stress_avg) and high_stress_avg < low_stress_avg:
            diff = round(low_stress_avg - high_stress_avg, 2)
            insights.append(f"Higher stress days show {diff} hrs less sleep on average than lower stress days.")

    return insights


def generate_recommendations(stats: dict, consistency: dict) -> list:
    """Return general, non-medical, actionable recommendations."""
    recs = []

    if stats["average_sleep_duration"] < 7:
        recs.append("Consider gradually shifting bedtime earlier to work toward 7-9 hours of sleep per night.")

    if consistency["consistency_label"] != "Highly Consistent":
        recs.append("Try keeping a consistent bedtime and wake-up time, including on weekends, to stabilize your sleep schedule.")

    recs.append("Limit screen time and stimulating activity in the hour before bed.")
    recs.append("Maintain regular physical activity, but avoid intense exercise close to bedtime.")
    recs.append("These are general wellness suggestions, not medical advice. Consult a healthcare professional for persistent sleep issues.")

    return recs


if __name__ == "__main__":
    from analysis import compute_sleep_statistics, analyze_consistency

    df = pd.read_csv("../data/processed/cleaned_sleep_data.csv")
    stats = compute_sleep_statistics(df)
    consistency = analyze_consistency(df)

    for i in generate_insights(df, stats, consistency):
        print("- ", i)
    print()
    for r in generate_recommendations(stats, consistency):
        print("* ", r)
