"""Tests for the CWRU -> Paderborn domain shift study."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import aether_pdm.ops.promote as promote_mod
from aether_pdm.ops.domain_shift import (
    compute_domain_shift,
    evaluate_on_target,
    write_domain_shift_report,
)


def _write_features(path, n_normal=60, n_faulty=60, seed=0, shift=0.0):
    """Write a synthetic feature Parquet with a 'split' column."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_normal):
        rows.append({
            "rms": 0.1 + shift + rng.normal(0, 0.02),
            "kurtosis": 3.0 + rng.normal(0, 0.2),
            "crest": 4.0 + rng.normal(0, 0.2),
            "fault_type": "normal",
            "split": "test",
        })
    for _ in range(n_faulty):
        rows.append({
            "rms": 0.5 + shift + rng.normal(0, 0.05),
            "kurtosis": 8.0 + rng.normal(0, 0.4),
            "crest": 6.0 + rng.normal(0, 0.3),
            "fault_type": "inner_race",
            "split": "test",
        })
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _mock_mlflow(monkeypatch):
    """Make mlflow client/tracking inert so studies never hit a real store."""
    import aether_pdm.ops.domain_shift as ds_mod

    monkeypatch.setattr(ds_mod.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(ds_mod.mlflow.tracking, "MlflowClient", lambda: object())


def test_compute_domain_shift_returns_summary(tmp_path):
    """Shifted distributions -> dict with summary keys + worst feature."""
    cwru = _write_features(tmp_path / "cwru" / "features.parquet", n_normal=300, n_faulty=300)
    paderborn = _write_features(
        tmp_path / "paderborn" / "features.parquet",
        n_normal=300,
        n_faulty=300,
        seed=7,
        shift=0.5,
    )

    result = compute_domain_shift(cwru, paderborn)

    assert "feature_drift_report" in result
    assert "mean_psi" in result
    assert "n_features_drifted" in result
    assert "worst_feature" in result
    assert result["worst_feature"] == "rms"  # shifted by 0.5
    assert result["worst_psi"] > 0.25
    assert result["mean_psi"] > 0.1
    assert result["n_features_drifted"] >= 1


def test_compute_domain_shift_no_shift(tmp_path):
    """Identical distributions -> no drifted features, mean PSI ~ 0."""
    cwru = _write_features(tmp_path / "cwru" / "features.parquet", n_normal=300, n_faulty=300)
    paderborn = _write_features(
        tmp_path / "paderborn" / "features.parquet", n_normal=300, n_faulty=300
    )

    result = compute_domain_shift(cwru, paderborn)

    assert result["n_features_drifted"] == 0
    assert result["mean_psi"] == 0.0
    assert result["worst_feature"] is not None


def test_evaluate_on_target_anomaly(tmp_path):
    """labels=None -> anomaly evaluation metrics."""
    path = _write_features(tmp_path / "features.parquet")
    model = IsolationForest(contamination=0.1, random_state=42).fit(np.random.randn(50, 3))

    result = evaluate_on_target(model, path, "test")

    assert "detection_rate" in result
    assert "false_alarm_rate" in result
    assert "n_samples" in result
    assert result["n_samples"] == 120


def test_evaluate_on_target_fault(tmp_path):
    """LabelEncoder labels -> fault classification metrics."""
    path = _write_features(tmp_path / "features.parquet")
    x = np.random.randn(40, 3)
    y = np.array([0] * 20 + [1] * 20)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(x, y)
    le = LabelEncoder().fit(["inner_race", "normal"])

    result = evaluate_on_target(model, path, "test", labels=le)

    assert "f1_macro" in result
    assert "balanced_accuracy" in result
    assert result["n_samples"] == 120


def test_evaluate_on_target_captures_error(tmp_path):
    """Empty target split -> dict with error key, no crash."""
    path = _write_features(tmp_path / "features.parquet")
    df = pd.read_parquet(path)
    df["split"] = "train"  # no 'test' rows anymore
    df.to_parquet(path, index=False)
    model = IsolationForest(contamination=0.1, random_state=42).fit(np.random.randn(10, 3))

    result = evaluate_on_target(model, path, "test")

    assert "error" in result
    assert result["split"] == "test"


def test_run_domain_shift_study_missing_models(tmp_path, monkeypatch):
    """No MLflow models -> dict with models_missing=True and notes (no crash)."""
    import aether_pdm.ops.domain_shift as ds_mod

    cwru = _write_features(tmp_path / "cwru" / "features.parquet")
    paderborn = _write_features(tmp_path / "paderborn" / "features.parquet", seed=3)
    _mock_mlflow(monkeypatch)

    def raise_missing(name, client):
        raise ValueError(f"No model versions found for '{name}'.")

    monkeypatch.setattr(promote_mod, "_load_candidate_model", raise_missing)

    result = ds_mod.run_domain_shift_study(cwru, paderborn)

    assert result["models_missing"] is True
    assert len(result["notes"]) >= 2
    assert "error" in result["anomaly"]["source"]
    assert "error" in result["anomaly"]["target"]
    assert "error" in result["fault"]["source"]
    assert "error" in result["fault"]["target"]
    assert "drift" in result
    assert "performance_delta" in result


def test_run_domain_shift_study_mocked_models(tmp_path, monkeypatch):
    """Models loaded + evaluate fns mocked -> source/target metrics assembled."""
    import aether_pdm.ops.domain_shift as ds_mod

    cwru = _write_features(tmp_path / "cwru" / "features.parquet")
    paderborn = _write_features(tmp_path / "paderborn" / "features.parquet", seed=5)
    _mock_mlflow(monkeypatch)

    fake_le = LabelEncoder().fit(["inner_race", "normal"])

    def fake_load(name, client):
        return object(), 3

    def fake_anomaly(model, features_path, split="val"):
        return {
            "n_samples": 120,
            "n_faults": 60,
            "n_normal": 60,
            "false_alarm_rate": 0.03,
            "detection_rate": 0.95,
            "best_threshold": -0.1,
            "best_far": 0.02,
            "best_detection_rate": 0.97,
        }

    def fake_fault(model, le, features_path, split="val"):
        return {
            "n_samples": 120,
            "f1_macro": 0.91,
            "balanced_accuracy": 0.89,
            "classes": ["inner_race", "normal"],
        }

    monkeypatch.setattr(promote_mod, "_load_candidate_model", fake_load)
    monkeypatch.setattr(
        promote_mod, "_load_fault_label_encoder", lambda client, name, version: fake_le
    )
    monkeypatch.setattr(promote_mod, "evaluate_anomaly_candidate", fake_anomaly)
    monkeypatch.setattr(promote_mod, "evaluate_fault_candidate", fake_fault)

    result = ds_mod.run_domain_shift_study(cwru, paderborn)

    assert result["models_missing"] is False
    assert result["anomaly"]["source"]["detection_rate"] == 0.95
    assert result["anomaly"]["target"]["detection_rate"] == 0.95
    assert result["fault"]["source"]["f1_macro"] == 0.91
    assert result["fault"]["target"]["balanced_accuracy"] == 0.89
    assert result["performance_delta"]["anomaly"]["detection_rate"] == 0.0
    assert result["performance_delta"]["fault"]["f1_macro"] == 0.0
    assert "drift" in result


def test_run_domain_shift_study_partial_models(tmp_path, monkeypatch):
    """Only the anomaly model available -> fault entries report errors."""
    import aether_pdm.ops.domain_shift as ds_mod

    cwru = _write_features(tmp_path / "cwru" / "features.parquet")
    paderborn = _write_features(tmp_path / "paderborn" / "features.parquet", seed=11)
    _mock_mlflow(monkeypatch)

    def fake_load(name, client):
        if name == ds_mod.ANOMALY_MODEL_NAME:
            return object(), 2
        raise ValueError(f"No model versions found for '{name}'.")

    def fake_anomaly(model, features_path, split="val"):
        return {
            "n_samples": 120,
            "n_faults": 60,
            "n_normal": 60,
            "false_alarm_rate": 0.05,
            "detection_rate": 0.88,
            "best_threshold": -0.1,
            "best_far": 0.04,
            "best_detection_rate": 0.90,
        }

    monkeypatch.setattr(promote_mod, "_load_candidate_model", fake_load)
    monkeypatch.setattr(promote_mod, "evaluate_anomaly_candidate", fake_anomaly)

    result = ds_mod.run_domain_shift_study(cwru, paderborn)

    assert result["models_missing"] is False  # anomaly present
    assert result["anomaly"]["source"]["detection_rate"] == 0.88
    assert "error" in result["fault"]["source"]
    assert "error" in result["fault"]["target"]


def test_write_domain_shift_report(tmp_path):
    """Report writer -> Markdown file with the expected sections."""
    cwru = _write_features(tmp_path / "cwru" / "features.parquet", n_normal=200, n_faulty=200)
    paderborn = _write_features(
        tmp_path / "paderborn" / "features.parquet",
        n_normal=200,
        n_faulty=200,
        seed=9,
        shift=0.4,
    )

    drift = compute_domain_shift(cwru, paderborn)
    result = {
        "source_features": str(cwru),
        "target_features": str(paderborn),
        "models_missing": True,
        "notes": ["No trained models found in registry."],
        "anomaly": {
            "source": {"error": "no model", "split": "test"},
            "target": {"error": "no model", "split": "test"},
            "delta": {
                "false_alarm_rate": None,
                "detection_rate": None,
                "best_detection_rate": None,
            },
        },
        "fault": {
            "source": {"error": "no model", "split": "test"},
            "target": {"error": "no model", "split": "test"},
            "delta": {"f1_macro": None, "balanced_accuracy": None},
        },
        "drift": drift,
        "performance_delta": {
            "anomaly": {
                "false_alarm_rate": None,
                "detection_rate": None,
                "best_detection_rate": None,
            },
            "fault": {"f1_macro": None, "balanced_accuracy": None},
        },
    }

    out = write_domain_shift_report(result, tmp_path / "report.md")

    assert out == tmp_path / "report.md"
    text = out.read_text(encoding="utf-8")
    assert "Domain Shift Report" in text
    assert "Mean PSI" in text
    assert "Feature Drift" in text
    assert "Interpretation" in text
    assert "No trained models were found" in text
