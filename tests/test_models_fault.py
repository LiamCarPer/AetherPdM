"""Tests for fault classification model."""

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from aether_pdm.data.synthetic import generate_dataset
from aether_pdm.models.fault import (
    FAULT_LABELS,
    predict_fault,
    predict_fault_full,
    train_fault_classifier,
)
from aether_pdm.signal.pipeline import FEATURE_VERSION, process_dataset


def _prepare_features(tmp_path: Path, n_normal: int = 5, n_faulty: int = 10) -> Path:
    """Generate synthetic data and run through the signal pipeline."""
    data_path = generate_dataset(tmp_path / "synth", n_normal=n_normal, n_faulty=n_faulty, seed=42)
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    result = process_dataset(data_path, output_dir=feat_dir, window_size=1024, overlap=0.5)
    return feat_dir / f"features_{FEATURE_VERSION}.parquet"


def test_train_fault_creates_model(tmp_path):
    """Training should produce a RandomForest + LabelEncoder."""
    features_path = _prepare_features(tmp_path, n_normal=5, n_faulty=10)
    model, le = train_fault_classifier(features_path, mlflow_uri="sqlite:///" + str(tmp_path / "mlruns.db").replace("\\", "/"))
    assert isinstance(model, RandomForestClassifier)
    assert isinstance(le, LabelEncoder)


def test_predict_fault_returns_labels():
    """Predict should return class labels from the trained encoder."""
    X = np.random.randn(20, 4)
    y = np.array([0, 1, 2, 3] * 5)
    le = LabelEncoder().fit(FAULT_LABELS)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    classes, confs = predict_fault(model, le, np.random.randn(5, 4))
    assert len(classes) == 5
    assert len(confs) == 5
    for c in classes:
        assert c in FAULT_LABELS


def test_predict_fault_full_probabilities():
    """Full prediction should return probability dict for each sample."""
    X = np.random.randn(20, 4)
    y = np.array([0, 1, 2, 3] * 5)
    le = LabelEncoder().fit(FAULT_LABELS)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    results = predict_fault_full(model, le, np.random.randn(3, 4))
    assert len(results) == 3
    for r in results:
        assert set(r.keys()) == set(FAULT_LABELS)
        assert abs(sum(r.values()) - 1.0) < 1e-6
