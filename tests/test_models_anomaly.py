"""Tests for anomaly detection model."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from aether_pdm.models.anomaly import train_anomaly, predict_anomaly
from aether_pdm.data.synthetic import generate_dataset
from aether_pdm.signal.pipeline import process_dataset


def _prepare_features(tmp_path: Path, n_normal: int = 5, n_faulty: int = 5) -> Path:
    """Generate synthetic data and run through the signal pipeline."""
    data_path = generate_dataset(tmp_path / "synth", n_normal=n_normal, n_faulty=n_faulty, seed=42)
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    result = process_dataset(data_path, output_dir=feat_dir, window_size=1024, overlap=0.5)
    return feat_dir / "features_v1.parquet"


def test_train_anomaly_creates_model(tmp_path):
    """Training should produce an IsolationForest model."""
    features_path = _prepare_features(tmp_path, n_normal=5, n_faulty=5)
    model = train_anomaly(features_path, mlflow_uri="sqlite:///" + str(tmp_path / "mlruns.db").replace("\\", "/"))
    assert isinstance(model, IsolationForest)


def test_predict_anomaly_shapes():
    """Predict should return consistent score and prediction arrays."""
    model = IsolationForest(contamination=0.1, random_state=42).fit(np.random.randn(50, 5))
    features = np.random.randn(10, 5)
    scores, preds = predict_anomaly(model, features)
    assert scores.shape == (10,)
    assert preds.shape == (10,)
    assert set(preds).issubset({0, 1})


def test_anomaly_detects_outliers():
    """Clear outliers should be detected as anomalous."""
    normal = np.random.randn(90, 3) * 0.5
    outliers = np.random.randn(10, 3) * 5 + 10
    X = np.vstack([normal, outliers])
    model = IsolationForest(contamination=0.1, random_state=42).fit(X)
    _, preds = predict_anomaly(model, X)
    assert preds[:90].sum() < preds[90:].sum()
