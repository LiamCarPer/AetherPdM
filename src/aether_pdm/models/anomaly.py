"""
Anomaly detection model for bearing vibration.

Trains an IsolationForest on healthy-only windows.
Outputs anomaly score and binary prediction.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from aether_pdm.eval.metrics import compute_anomaly_metrics

META_COLS = {
    "window_id", "window_start", "window_end", "asset_id", "file_id",
    "channel", "fault_type", "fault_diameter", "severity", "split",
    "load_hp", "feature_version", "rpm", "sampling_rate", "waveform",
}


def train_anomaly(
    features_path: Path,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
    mlflow_uri: str | None = None,
    feature_cols: list[str] | None = None,
    split: str | None = None,
    strict_boundary: bool = False,
) -> IsolationForest:
    """
    Train IsolationForest anomaly detector on healthy-only data.

    Features should already be windowed and computed.
    Filters to rows where fault_type == 'normal'.

    Parameters
    ----------
    feature_cols : optional subset of columns to use as features.
        When None, all non-meta columns are used.
    split : optional data split filter (e.g. 'train', 'val').
        When set, only rows with matching ``split`` column are used.
    strict_boundary : bool
        When True, override the model's decision boundary so that the worst
        healthy training window scores exactly 0 and every healthy window
        scores >= 0 (FAR = 0 on the healthy training distribution). This is
        the correct configuration when training on PURE healthy data:
        ``contamination`` > 0 would otherwise force ~contamination of the
        healthy rows below the boundary, guaranteeing a false-alarm floor on
        any healthy validation set. Fault windows (far outside the healthy
        hull) still score negative and are flagged. Default False preserves
        the sklearn default boundary.
    """
    df = pd.read_parquet(features_path)
    if split:
        df = df[df["split"] == split]
    healthy = df[df["fault_type"] == "normal"].copy()
    if healthy.empty:
        raise ValueError("No healthy (normal) samples found in the dataset")

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in healthy.columns]
        if missing:
            raise ValueError(f"Feature columns not found in data: {missing}")
        x_train = healthy[feature_cols].values
    else:
        feature_cols = [c for c in healthy.columns if c not in META_COLS]
        x_train = healthy[feature_cols].values

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train)

    if strict_boundary:
        # decision_function(X) = score_samples(X) - offset_. Setting offset_ to
        # the minimum healthy score puts the boundary at the worst healthy
        # window: all healthy windows score >= 0, only deviations beyond the
        # healthy hull (true faults) score negative.
        model.offset_ = float(np.min(model.score_samples(x_train)))

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri or "mlruns")
    with mlflow.start_run(run_name="anomaly_train") as run:
        mlflow.log_params({
            "model_type": "IsolationForest",
            "n_estimators": n_estimators,
            "contamination": contamination,
            "random_state": random_state,
            "n_train_samples": len(x_train),
            "strict_boundary": strict_boundary,
        })
        model_info = mlflow.sklearn.log_model(
            model, "model", registered_model_name="aether-anomaly"
        )
        if model_info.registered_model_version is not None:
            mlflow.tracking.MlflowClient().set_registered_model_alias(
                "aether-anomaly",
                "staging",
                str(model_info.registered_model_version),
            )
        mlflow.log_artifact(str(features_path), artifact_path="data")

        # Log baseline metrics on training data
        scores = model.decision_function(x_train)
        preds = np.where(scores < 0, 1, 0)  # negative score = anomaly
        y_true = np.zeros(len(x_train))  # all healthy
        metrics = compute_anomaly_metrics(y_true, preds)
        mlflow.log_metrics(metrics)

        run_id = run.info.run_id
        print(f"Anomaly model trained. MLflow run: {run_id}")
        print(f"  Train samples: {len(x_train)}, Contamination target: {contamination:.2f}")
        print(f"  Metrics: {metrics}")

    return model


def predict_anomaly(
    model: IsolationForest,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict anomaly scores and binary labels.

    Returns:
        scores: anomaly scores (higher = more anomalous)
        predictions: 0 = normal, 1 = anomaly
    """
    scores = model.decision_function(features)
    predictions = np.where(scores < 0, 1, 0)
    return scores, predictions
