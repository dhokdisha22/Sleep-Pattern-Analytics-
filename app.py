"""
app.py
Sleep Pattern Analysis — Streamlit Dashboard

Implements FR-13, FR-14, FR-15 from the FRD:
- Interactive dashboard: stats, charts, ML results, insights, recommendations
- Filtering by available fields (date range, age, stress level)
- Clear, user-friendly result display
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from data_cleaning import load_data, validate_data, clean_data
from analysis import compute_sleep_statistics, analyze_consistency, analyze_trend, identify_irregular_days
from visualization import (
    plot_sleep_duration_trend, plot_duration_histogram, plot_quality_distribution,
    plot_bedtime_waketime, plot_stress_vs_duration, plot_confusion_matrix,
)
from ml_model import prepare_features, train_model, evaluate_model
from recommendations import generate_insights, generate_recommendations

st.set_page_config(page_title="Sleep Pattern Analysis", page_icon="😴", layout="wide")

st.title("😴 Sleep Pattern Analysis Dashboard")
st.caption("TY Individual College Project — Academic analysis tool, not a medical diagnostic system.")

# ---------------- Sidebar: Data Input ----------------
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload sleep dataset (CSV)", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset", value=uploaded_file is None)

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sleep_dataset.csv")

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
elif use_sample:
    raw_df = load_data(DEFAULT_PATH)
else:
    st.info("Upload a CSV file or check 'Use sample dataset' to continue.")
    st.stop()

with st.expander("Data Validation Report"):
    st.json(validate_data(raw_df))


df = clean_data(raw_df)

# Normalize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# Rename dataset columns to match the project code
df = df.rename(columns={
    "quality_of_sleep": "sleep_quality",
    "wake_up_time": "wake_time",
    "physical_activity_level": "physical_activity_min"
})

# Convert date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")
# ---------------- Sidebar: Filters ----------------
st.sidebar.header("2. Filters")
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range", (min_date, max_date))
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

if "age" in df.columns and df["age"].notna().any():
    age_range = st.sidebar.slider("Age range", int(df["age"].min()), int(df["age"].max()),
                                   (int(df["age"].min()), int(df["age"].max())))
    df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

if df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ---------------- Summary Statistics ----------------
st.header("2. Summary Statistics")
stats = compute_sleep_statistics(df)
consistency = analyze_consistency(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average Sleep", f"{stats['average_sleep_duration']} hrs")
c2.metric("Min Sleep", f"{stats['min_sleep_duration']} hrs")
c3.metric("Max Sleep", f"{stats['max_sleep_duration']} hrs")
c4.metric("Consistency", consistency["consistency_label"])

# ---------------- Charts ----------------
st.header("3. Visualizations")
trend_df = analyze_trend(df)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_sleep_duration_trend(trend_df), use_container_width=True)
    st.plotly_chart(plot_quality_distribution(df), use_container_width=True)
with col2:
    st.plotly_chart(plot_duration_histogram(df), use_container_width=True)
    st.plotly_chart(plot_bedtime_waketime(df), use_container_width=True)

st.plotly_chart(plot_stress_vs_duration(df), width="stretch")

irregular = identify_irregular_days(df)
with st.expander(f"Irregular Sleep Days ({len(irregular)} found)"):
    st.dataframe(irregular)

# ---------------- ML Model ----------------
st.header("4. Sleep Quality Classification (Random Forest)")

st.caption(
    "Sleep quality is classified into three categories: "
    "Poor, Average, and Good."
)

# Check whether enough data is available
quality_counts = df["sleep_quality"].dropna().value_counts()

if (
    df["sleep_quality"].nunique() >= 2
    and len(df) >= 20
):
    try:
        # Prepare features and target
        X, y, encoder = prepare_features(df)

        # Train Random Forest model
        model, X_test, y_test = train_model(X, y)

        # Evaluate model
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            encoder
        )

        # Display metrics
        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        m2.metric(
            "Precision",
            f"{metrics['precision'] * 100:.2f}%"
        )

        m3.metric(
            "Recall",
            f"{metrics['recall'] * 100:.2f}%"
        )

        m4.metric(
            "F1-score",
            f"{metrics['f1_score'] * 100:.2f}%"
        )

        # Model status
        if metrics["accuracy"] >= 0.70:
            st.success(
                "The model shows good classification performance."
            )
        elif metrics["accuracy"] >= 0.50:
            st.info(
                "The model shows moderate classification performance."
            )
        else:
            st.warning(
                "The model has limited predictive performance. "
                "Results should be interpreted as an academic experiment."
            )

        # Confusion Matrix
        st.subheader("Confusion Matrix")

        st.plotly_chart(
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                metrics["labels"]
            ),
            use_container_width=True,
        )

        # Classification Report
        with st.expander("Full Classification Report"):
            st.text(
                metrics["classification_report"]
            )

    except Exception as e:
        st.error(
            f"Unable to train the classification model: {e}"
        )

else:
    st.info(
        "Not enough labeled data to train the classifier. "
        "At least 20 records and 2 sleep-quality classes are required."
    )
# ---------------- Insights & Recommendations ----------------
st.header("5. Insights & Recommendations")
insights = generate_insights(df, stats, consistency)
recs = generate_recommendations(stats, consistency)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Insights")
    for i in insights:
        st.write(f"- {i}")
with col_b:
    st.subheader("Recommendations")
    for r in recs:
        st.write(f"- {r}")

st.divider()
st.caption("This dashboard provides general wellness insights based on statistical/ML analysis. "
           "It is not a medical diagnostic tool.")
