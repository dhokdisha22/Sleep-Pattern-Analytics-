"""
visualization.py
Sleep Pattern Analysis — Visualization Module

Implements FR-07 from the FRD:
- Bar charts, line charts, histograms and other relevant visualizations
Uses Plotly so charts render natively inside the Streamlit dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_sleep_duration_trend(df: pd.DataFrame):
    """Line chart: sleep duration over time with a 7-day rolling average."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["sleep_duration"],
                              mode="lines+markers", name="Sleep Duration",
                              line=dict(color="#6C63FF")))
    if "rolling_avg_7d" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_avg_7d"],
                                  mode="lines", name="7-Day Avg",
                                  line=dict(color="#FF6584", dash="dash")))
    fig.update_layout(title="Sleep Duration Trend", xaxis_title="Date",
                       yaxis_title="Hours", template="plotly_white")
    return fig


def plot_duration_histogram(df: pd.DataFrame):
    """Histogram of sleep duration distribution."""
    fig = px.histogram(df, x="sleep_duration", nbins=20,
                        title="Sleep Duration Distribution",
                        color_discrete_sequence=["#6C63FF"])
    fig.update_layout(template="plotly_white", xaxis_title="Sleep Duration (hrs)",
                       yaxis_title="Count")
    return fig


def plot_quality_distribution(df: pd.DataFrame):
    """Bar chart of sleep quality category counts."""
    counts = df["sleep_quality"].value_counts().reset_index()
    counts.columns = ["sleep_quality", "count"]
    fig = px.bar(counts, x="sleep_quality", y="count", color="sleep_quality",
                 title="Sleep Quality Distribution",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_bedtime_waketime(df: pd.DataFrame):
    """Scatter plot comparing bedtime vs wake-up time (in minutes past midnight)."""
    tmp = df.copy()
    tmp["bedtime_min"] = tmp["bedtime"].apply(_to_minutes)
    tmp["waketime_min"] = tmp["wake_time"].apply(_to_minutes)
    fig = px.scatter(tmp, x="bedtime_min", y="waketime_min", color="sleep_quality",
                      title="Bedtime vs Wake-up Time",
                      labels={"bedtime_min": "Bedtime (min past midnight)",
                              "waketime_min": "Wake-up Time (min past midnight)"})
    fig.update_layout(template="plotly_white")
    return fig


def plot_stress_vs_duration(df: pd.DataFrame):
    """Box plot of sleep duration grouped by stress level."""
    fig = px.box(
    df,
    x="stress_level",
    y="sleep_duration",
    color="stress_level",
    title="Sleep Duration by Stress Level",
    color_discrete_sequence=px.colors.sequential.Purples
)
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def plot_confusion_matrix(cm, labels):
    """Heatmap for the ML model confusion matrix."""
    fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                     labels=dict(x="Predicted", y="Actual", color="Count"),
                     title="Confusion Matrix", color_continuous_scale="Blues")
    return fig


def _to_minutes(t: str):
    try:
        h, m = map(int, str(t).split(":"))
        return h * 60 + m
    except Exception:
        return None
