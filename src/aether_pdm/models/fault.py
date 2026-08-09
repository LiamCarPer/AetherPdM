"""
Fault classification model for bearing faults.

Trains a RandomForest on all fault classes (inner_race, outer_race, ball, normal).
Outputs class prediction with confidence scores.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from aether_pdm.eval.metrics import classification_report_dict

FAULT_LABELS = ["normal", "inner_race", "outer_race", "ball"]


META_COLS = {
    "window_id", "window_start", "window_end", "asset_id", "file_id",
    "channel", "fault_type", "fault_diameter", "severity", "split",
    "load_hp", "feature_version", "rpm", "sampling_rate", "waveform",
}


def train_fault_classifier(
    features_path: Path,
    n_estimators: int = 300,
    max_depth: int = 12,
    min_samples_leaf: int = 4,
    class_weight: str = "balanced",
    random_state: int = 42,
    mlflow_uri: str | None = None,
    feature_cols: list[str] | None = None,
    split: str | None = None,
) -> tuple[RandomForestClassifier, LabelEncoder]:
    """
    Train RandomForest fault classifier on all labeled data.

    Features should already be windowed and computed.
    Uses 'fault_type' column as the target.

    Parameters
    ----------
    feature_cols : optional subset of columns to use as features.
        When None, all non-meta columns are used.
    split : optional data split filter (e.g. 'train', 'val').
        When set, only rows with matching ``split`` column are used.
    """
    df = pd.read_parquet(features_path)
    if split:
        df = df[df["split"] == split]
    df = df[df["fault_type"].isin(FAULT_LABELS)].copy()
    if df.empty:
        raise ValueError("No labeled fault samples found")

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found in data: {missing}")
        x = df[feature_cols].values
    else:
        feature_cols = [c for c in df.columns if c not in META_COLS]
        x = df[feature_cols].values

    le = LabelEncoder()
    y = le.fit_transform(df["fault_type"])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x, y)

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri or "mlruns")
    with mlflow.start_run(run_name="fault_train") as run:
        classes_list = le.classes_.tolist()
        mlflow.log_params({
            "model_type": "RandomForest",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "n_train_samples": len(x),
            "classes": ",".join(classes_list),
        })
        model_info = mlflow.sklearn.log_model(
            model, "model", registered_model_name="aether-fault-clf"
        )
        if model_info.registered_model_version is not None:
            mlflow.tracking.MlflowClient().set_registered_model_alias(
                "aether-fault-clf",
                "staging",
                str(model_info.registered_model_version),
            )
        mlflow.log_artifact(str(features_path), artifact_path="data")

        # Log baseline metrics on training data
        y_pred = model.predict(x)
        metrics = classification_report_dict(y, y_pred, labels=le.classes_.tolist())
        mlflow.log_metrics(metrics)

        run_id = run.info.run_id
        print(f"Fault classifier trained. MLflow run: {run_id}")
        print(f"  Train samples: {len(x)}, Classes: {list(le.classes_)}")
        print(f"  Metrics: {metrics}")

    return model, le


def predict_fault(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    features: np.ndarray,
) -> tuple[list[str], list[float]]:
    """
    Predict fault classes and confidence scores.

    Returns:
        classes: predicted class labels
        confidences: probability scores for predicted class
    """
    probs = model.predict_proba(features)
    pred_indices = model.predict(features)
    classes = label_encoder.inverse_transform(pred_indices).tolist()
    confidences = [float(probs[i, pred_indices[i]]) for i in range(len(features))]
    return classes, confidences


def predict_fault_full(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    features: np.ndarray,
) -> list[dict]:
    """
    Predict fault classes with full probability distribution.

    Returns a list of dicts with class -> probability for each sample.
    """
    probs = model.predict_proba(features)
    results = []
    for i in range(len(features)):
        probs_dict = {
            label_encoder.inverse_transform([j])[0]: float(probs[i, j])
            for j in range(probs.shape[1])
        }
        results.append(probs_dict)
    return results
