"""
ml_model.py
Sleep Pattern Analysis — Machine Learning Module

Random Forest classification for sleep quality.
Sleep quality scores are grouped into 3 classes:
- Poor
- Average
- Good
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

FEATURE_COLUMNS = [
    "sleep_duration",
    "age",
    "stress_level",
    "physical_activity_min",
]

TARGET_COLUMN = "sleep_quality"


def convert_quality_to_class(value):
    """
    Convert numeric sleep quality score into 3 classes.
    1-4  -> Poor
    5-7  -> Average
    8-10 -> Good
    """
    if pd.isna(value):
        return np.nan

    value = float(value)

    if value <= 4:
        return "Poor"
    elif value <= 7:
        return "Average"
    else:
        return "Good"


def prepare_features(df: pd.DataFrame):
    """Prepare features and convert sleep quality into 3 classes."""

    data = df.copy()

    # Convert target to meaningful categories
    data[TARGET_COLUMN] = data[TARGET_COLUMN].apply(
        convert_quality_to_class
    )

    # Keep only required columns
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    data = data[required_columns].copy()

    # Convert feature columns to numeric
    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    # Remove rows containing missing values
    data = data.dropna()

    X = data[FEATURE_COLUMNS].copy()
    y_raw = data[TARGET_COLUMN].copy()

    # Encode:
    # Average -> 0
    # Good    -> 1
    # Poor    -> 2
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    return X, y, encoder


def train_model(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Split data and train Random Forest classifier."""

    # Check class distribution
    class_counts = pd.Series(y).value_counts()

    # Stratification is possible only when every class
    # has at least 2 records.
    if len(class_counts) >= 2 and class_counts.min() >= 2:
        stratify_value = y
    else:
        stratify_value = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_value,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test


def evaluate_model(
    model,
    X_test,
    y_test,
    encoder: LabelEncoder
) -> dict:
    """Calculate classification metrics."""

    y_pred = model.predict(X_test)

    y_test_clean = np.asarray(y_test, dtype=int)
    y_pred_clean = np.asarray(y_pred, dtype=int)

    accuracy = accuracy_score(
        y_test_clean,
        y_pred_clean
    )

    precision = precision_score(
        y_test_clean,
        y_pred_clean,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test_clean,
        y_pred_clean,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test_clean,
        y_pred_clean,
        average="weighted",
        zero_division=0
    )

    num_classes = len(encoder.classes_)

    all_labels = list(range(num_classes))

    target_names = [
        str(label)
        for label in encoder.classes_
    ]

    matrix = confusion_matrix(
        y_test_clean,
        y_pred_clean,
        labels=all_labels
    )

    report = classification_report(
        y_test_clean,
        y_pred_clean,
        labels=all_labels,
        target_names=target_names,
        zero_division=0
    )

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": matrix.tolist(),
        "labels": target_names,
        "classification_report": report,
    }

    return metrics


def save_model(
    model,
    encoder,
    path: str = "../models/sleep_quality_model.pkl"
):
    """Save trained model and encoder."""

    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "features": FEATURE_COLUMNS,
        },
        path
    )


def load_model(
    path: str = "../models/sleep_quality_model.pkl"
):
    """Load saved model."""

    return joblib.load(path)


def predict_quality(
    model_bundle,
    input_row: dict
) -> str:
    """Predict sleep quality for one new record."""

    model = model_bundle["model"]
    encoder = model_bundle["encoder"]
    features = model_bundle["features"]

    X_new = pd.DataFrame([input_row])[features]

    pred_encoded = model.predict(X_new)[0]

    return encoder.inverse_transform(
        [pred_encoded]
    )[0]


if __name__ == "__main__":

    df = pd.read_csv(
        "../data/processed/cleaned_sleep_data.csv"
    )

    X, y, encoder = prepare_features(df)

    model, X_test, y_test = train_model(
        X,
        y
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        encoder
    )

    print("\nSleep Quality Classification")
    print("----------------------------")
    print(metrics["classification_report"])

    print(
        f"Accuracy: {metrics['accuracy']}"
    )

    save_model(
        model,
        encoder
    )

    print(
        "\nModel saved successfully."
    )