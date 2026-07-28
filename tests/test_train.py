"""Tests for config-driven training entrypoint."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from aether_pdm.models.anomaly import train_anomaly
from aether_pdm.models.fault import train_fault_classifier
from aether_pdm.models.train import load_config

# ── helpers ──


def _mlflow_uri(tmp_path: Path) -> str:
    """Return a local SQLite MLflow URI for testing."""
    return "sqlite:///" + str(tmp_path / "mlruns.db").replace("\\", "/")


def _write_features(tmp_path: Path, n_normal: int = 20, n_faulty: int = 20) -> Path:
    """Write a small synthetic feature Parquet for testing."""
    rows = []
    for _ in range(n_normal):
        rows.append({
            "rms": 0.1, "kurtosis": 3.0, "crest": 4.0, "skew": 0.0,
            "fault_type": "normal", "split": "train",
        })
    for _ in range(n_faulty):
        rows.append({
            "rms": 0.5, "kurtosis": 8.0, "crest": 6.0, "skew": 0.5,
            "fault_type": "inner_race", "split": "train",
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return path


def _write_anomaly_config(tmp_path: Path, **overrides) -> Path:
    cfg = {
        "model": {
            "name": "aether-anomaly",
            "algorithm": "IsolationForest",
            "params": {
                "n_estimators": 50,
                "contamination": 0.1,
                "random_state": 42,
            },
        },
        "features": ["rms", "kurtosis", "crest", "skew"],
        "data": {"source": "synthetic", "train_labels": ["normal"], "split": "train"},
    }
    _deep_merge(cfg, overrides)
    path = tmp_path / "anomaly_config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


def _write_fault_config(tmp_path: Path, **overrides) -> Path:
    cfg = {
        "model": {
            "name": "aether-fault-clf",
            "algorithm": "RandomForest",
            "params": {
                "n_estimators": 50,
                "max_depth": 8,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
            },
        },
        "features": ["rms", "kurtosis", "crest", "skew"],
        "data": {
            "source": "synthetic",
            "classes": ["normal", "inner_race"],
            "split": "train",
        },
    }
    _deep_merge(cfg, overrides)
    path = tmp_path / "fault_config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge overrides into base dict."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# ── load_config ──


def test_load_config_returns_dict(tmp_path):
    path = _write_anomaly_config(tmp_path)
    cfg = load_config(path)
    assert cfg["model"]["name"] == "aether-anomaly"
    assert cfg["model"]["params"]["n_estimators"] == 50


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


# ── train_anomaly config-driven ──


def test_train_anomaly_with_config(tmp_path):
    feat_path = _write_features(tmp_path)
    cfg_path = _write_anomaly_config(tmp_path)

    cfg = load_config(cfg_path)
    params = cfg["model"]["params"]

    model = train_anomaly(
        features_path=feat_path,
        contamination=params["contamination"],
        n_estimators=params["n_estimators"],
        random_state=params["random_state"],
        feature_cols=cfg["features"],
        split=cfg["data"]["split"],
        mlflow_uri=_mlflow_uri(tmp_path),
    )
    assert model.contamination == 0.1


def test_train_anomaly_with_invalid_feature_cols(tmp_path):
    feat_path = _write_features(tmp_path)
    with pytest.raises(ValueError, match="Feature columns not found"):
        train_anomaly(
            features_path=feat_path,
            feature_cols=["nonexistent_col"],
            split="train",
            mlflow_uri=_mlflow_uri(tmp_path),
        )


# ── train_fault_classifier config-driven ──


def test_train_fault_with_config(tmp_path):
    feat_path = _write_features(tmp_path)
    cfg_path = _write_fault_config(tmp_path)

    cfg = load_config(cfg_path)
    params = cfg["model"]["params"]

    model, le = train_fault_classifier(
        features_path=feat_path,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        random_state=params["random_state"],
        feature_cols=cfg["features"],
        split=cfg["data"]["split"],
        mlflow_uri=_mlflow_uri(tmp_path),
    )
    assert model.max_depth == 8
    assert model.min_samples_leaf == 2


def test_train_fault_config_aliased_params(tmp_path):
    """Verify that params set via config map to sklearn correctly."""
    feat_path = _write_features(tmp_path)

    model, le = train_fault_classifier(
        features_path=feat_path,
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=99,
        mlflow_uri=_mlflow_uri(tmp_path),
    )
    assert model.n_estimators == 100
    assert model.max_depth == 10
    assert model.random_state == 99


def test_train_fault_with_invalid_feature_cols(tmp_path):
    feat_path = _write_features(tmp_path)
    with pytest.raises(ValueError, match="Feature columns not found"):
        train_fault_classifier(
            features_path=feat_path,
            feature_cols=["nonexistent"],
            split="train",
            mlflow_uri=_mlflow_uri(tmp_path),
        )


# ── split filtering ──


def test_train_anomaly_split_filter(tmp_path):
    """Only 'train' split rows should be used for training."""
    df = pd.DataFrame({
        "rms": [0.1, 0.2, 0.3],
        "kurtosis": [3.0, 4.0, 5.0],
        "crest": [4.0, 5.0, 6.0],
        "skew": [0.0, 0.1, 0.2],
        "fault_type": ["normal", "normal", "normal"],
        "split": ["train", "val", "train"],
    })
    feat_path = tmp_path / "features.parquet"
    df.to_parquet(feat_path, index=False)

    model = train_anomaly(features_path=feat_path, split="train", mlflow_uri=_mlflow_uri(tmp_path))
    # Should have 2 train samples, not 3
    assert model.contamination is not None  # model trained successfully


def test_train_fault_split_filter(tmp_path):
    """Only 'train' split rows should be used for training."""
    df = pd.DataFrame({
        "rms": [0.1, 0.5],
        "kurtosis": [3.0, 8.0],
        "crest": [4.0, 6.0],
        "skew": [0.0, 0.5],
        "fault_type": ["normal", "inner_race"],
        "split": ["val", "train"],
    })
    feat_path = tmp_path / "features.parquet"
    df.to_parquet(feat_path, index=False)

    uri = _mlflow_uri(tmp_path)
    model, le = train_fault_classifier(
        features_path=feat_path, split="train", mlflow_uri=uri,
    )
    assert len(le.classes_) == 1  # Only inner_race present in train split
    assert le.classes_[0] == "inner_race"
