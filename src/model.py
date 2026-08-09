"""
model.py
--------
Predicts the AQI_Bucket category (Good / Satisfactory / Moderate / Poor /
Very Poor / Severe) for a given day and city from pollutant readings.

This is the piece that turns the project from pure EDA into an applied
ML task: a classifier with a train/test split, evaluation metrics, and
feature importances -- not just descriptive charts.
"""

import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
    "Month",
]
TARGET = "AQI_Bucket"


def prepare_model_data(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with a known AQI_Bucket label and no missing features."""
    return df_clean.dropna(subset=FEATURES + [TARGET]).copy()


def train(df_clean: pd.DataFrame, model_out="outputs/aqi_model.joblib",
          report_out="outputs/model_report.json"):
    data = prepare_model_data(df_clean)

    le = LabelEncoder()
    y = le.fit_transform(data[TARGET])
    X = data[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # max_depth is capped and n_estimators kept modest deliberately: an
    # unbounded RandomForest on this data balloons to 60MB+ on disk for a
    # negligible accuracy gain, which is unnecessary bloat for a project
    # meant to be committed to git.
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=14, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    importances = dict(sorted(
        zip(FEATURES, clf.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True
    ))

    joblib.dump({"model": clf, "label_encoder": le, "features": FEATURES}, model_out)

    summary = {
        "accuracy": round(acc, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": le.classes_.tolist(),
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": importances,
    }
    with open(report_out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Accuracy: {acc:.4f}  (train={len(X_train)}, test={len(X_test)})")
    print("Top 5 features by importance:")
    for feat, imp in list(importances.items())[:5]:
        print(f"  {feat}: {imp:.3f}")

    return clf, le, summary


def predict_bucket(model_bundle: dict, readings: dict) -> str:
    """
    readings: dict with keys matching FEATURES (missing keys default to 0).
    Returns the predicted AQI_Bucket label as a string.
    """
    clf = model_bundle["model"]
    le = model_bundle["label_encoder"]
    row = [[readings.get(f, 0.0) for f in FEATURES]]
    pred = clf.predict(row)[0]
    return le.inverse_transform([pred])[0]


def predict_bucket_with_confidence(model_bundle: dict, readings: dict) -> dict:
    """
    Same as predict_bucket, but also returns per-class probabilities.
    Used by the API so clients can see how confident the model is,
    not just the single winning label.
    """
    clf = model_bundle["model"]
    le = model_bundle["label_encoder"]
    row = [[readings.get(f, 0.0) for f in FEATURES]]
    proba = clf.predict_proba(row)[0]
    pred_idx = proba.argmax()
    return {
        "predicted_bucket": le.inverse_transform([pred_idx])[0],
        "confidence": round(float(proba[pred_idx]), 4),
        "probabilities": {
            label: round(float(p), 4) for label, p in zip(le.classes_, proba)
        },
    }


if __name__ == "__main__":
    from data_pipeline import load_and_clean
    df_clean = load_and_clean()
    train(df_clean)