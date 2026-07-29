"""Tests for threshold optimisation, health score calibration, and fault probability scaling."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from aether_pdm.eval.metrics import find_optimal_threshold, threshold_scan
from aether_pdm.models.calibrate import (
    calibrate_anomaly_model,
    calibrate_fault_model,
    calibrate_health_score,
    optimize_anomaly_threshold,
)
from aether_pdm.models.fault import FAULT_LABELS

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _mlflow_uri(tmp_path: Path) -> str:
    return "sqlite:///" + str(tmp_path / "mlruns.db").replace("\\", "/")


def _make_features_parquet(
    tmp_path: Path,
    split: str = "val",
    n_normal: int = 50,
    n_faulty: int = 50,
) -> Path:
    """Create a minimal features Parquet for calibration testing."""
    rows = []
    rng = np.random.RandomState(42)
    for i in range(n_normal):
        rows.append({
            "rms": float(rng.normal(0.1, 0.02)),
            "kurtosis": float(rng.normal(3.0, 0.2)),
            "crest": float(rng.normal(4.0, 0.3)),
            "skew": float(rng.normal(0.0, 0.1)),
            "peak_freq": float(rng.normal(60.0, 5.0)),
            "spec_energy": float(rng.normal(0.5, 0.1)),
            "fault_type": "normal",
            "split": split,
            "window_id": f"w_n{i}",
            "fault_diameter": 0,
            "severity": 0,
        })
    fault_types = ["inner_race", "outer_race", "ball"]
    for i in range(n_faulty):
        ft = fault_types[i % len(fault_types)]
        rows.append({
            "rms": float(rng.normal(0.4, 0.1)),
            "kurtosis": float(rng.normal(7.0, 1.5)),
            "crest": float(rng.normal(7.0, 1.0)),
            "skew": float(rng.normal(0.3, 0.2)),
            "peak_freq": float(rng.normal(180.0, 10.0)),
            "spec_energy": float(rng.normal(2.0, 0.3)),
            "fault_type": ft,
            "split": split,
            "window_id": f"w_f{i}",
            "fault_diameter": 0,
            "severity": i % 3 + 1,
        })
    df = pd.DataFrame(rows)
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
#  threshold_scan
# ---------------------------------------------------------------------------


def test_threshold_scan_shapes():
    """Should return arrays all of length n_thresholds."""
    scores = np.array([0.1, 0.2, 0.3, -0.5, -0.8])
    y_true = np.array([0, 0, 0, 1, 1])
    result = threshold_scan(scores, y_true, n_thresholds=100)
    assert result["thresholds"].shape == (100,)
    assert result["false_alarm_rates"].shape == (100,)
    assert result["detection_rates"].shape == (100,)


def test_threshold_scan_empty():
    """Empty inputs should return empty arrays without crashing."""
    result = threshold_scan(np.array([]), np.array([]))
    assert result["thresholds"].size == 0


def test_threshold_scan_monotonic():
    """FAR should be non-decreasing with threshold (higher threshold = more predicted anomalies)."""
    scores = np.linspace(-2, 2, 200)
    y_true = np.array([0] * 100 + [1] * 100)
    result = threshold_scan(scores, y_true, n_thresholds=200)
    far = result["false_alarm_rates"]
    assert np.all(np.diff(far) >= -1e-12)  # non-decreasing


def test_threshold_scan_extreme_thresholds():
    """At min threshold everything is normal, at max threshold almost everything is anomalous."""
    scores = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    y_true = np.array([0, 1, 0, 1, 0])
    result = threshold_scan(scores, y_true, n_thresholds=50)
    # At lowest threshold: nothing predicted anomalous → FAR≈0, DR≈0
    assert result["false_alarm_rates"][0] == pytest.approx(0.0, abs=0.01)
    assert result["detection_rates"][0] == pytest.approx(0.0, abs=0.01)
    # At highest threshold: all scores below max are anomalous.
    # The sample with score == max (1.0) is *not* anomalous (scores < threshold, strict inequality)
    # So: 3 normals, 2 faults. At threshold=max: 4 anomalies (2FP + 2TP), 1 TN
    # FAR = 2/3 ≈ 0.667, DR = 2/2 = 1.0
    assert result["false_alarm_rates"][-1] == pytest.approx(2.0 / 3.0, abs=0.01)
    assert result["detection_rates"][-1] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
#  find_optimal_threshold
# ---------------------------------------------------------------------------


def test_find_optimal_threshold_basic():
    """Should find a threshold between the two classes."""
    scores = np.concatenate([
        np.random.normal(0.2, 0.1, size=100),   # normal-like scores
        np.random.normal(-0.8, 0.2, size=50),   # anomalous scores
    ])
    y_true = np.array([0] * 100 + [1] * 50)
    result = find_optimal_threshold(scores, y_true, target_recall=0.80, max_far=0.30)
    assert "best_threshold" in result
    assert -1.0 < result["best_threshold"] < 1.0
    assert result["best_far"] <= 0.30 or result["best_detection_rate"] >= 0.80


def test_find_optimal_threshold_no_valid_candidate():
    """When no threshold satisfies both constraints, fallback should still return something."""
    # Mixed scores situation where it's hard to satisfy both
    scores = np.random.normal(0.0, 0.1, size=100)
    y_true = np.array([0] * 50 + [1] * 50)  # random labels → no separation
    result = find_optimal_threshold(
        scores, y_true, target_recall=0.95, max_far=0.05, n_thresholds=100,
    )
    assert "best_threshold" in result
    assert isinstance(result["best_threshold"], float)
    assert isinstance(result["best_far"], float)


# ---------------------------------------------------------------------------
#  optimize_anomaly_threshold
# ---------------------------------------------------------------------------


def test_optimize_anomaly_threshold_keys():
    """Return dict should contain all expected keys."""
    scores = np.array([0.3, 0.1, 0.4, -0.6, -0.9, 0.0])
    y_true = np.array([0, 0, 0, 1, 1, 0])
    result = optimize_anomaly_threshold(scores, y_true)
    for key in ("threshold", "false_alarm_rate", "detection_rate", "f1",
                "all_thresholds", "all_fars", "all_detection_rates"):
        assert key in result


def test_optimize_anomaly_threshold_perfect_separation():
    """With perfectly separated scores the best threshold should land cleanly."""
    scores = np.array([1.0, 1.1, 0.9, -2.0, -2.1, -1.9])
    y_true = np.array([0, 0, 0, 1, 1, 1])
    result = optimize_anomaly_threshold(scores, y_true, target_recall=0.90, max_far=0.10)
    assert result["false_alarm_rate"] <= 0.10
    assert result["detection_rate"] >= 0.90
    assert result["f1"] > 0.5


def test_optimize_anomaly_threshold_empty():
    """Empty arrays should not crash."""
    result = optimize_anomaly_threshold(np.array([]), np.array([]))
    assert result["threshold"] == 0.0


def test_optimize_anomaly_threshold_fallback_warning(caplog):
    """When constraints are too tight, a warning should be logged."""
    scores = np.random.normal(0.0, 1.0, size=200)
    y_true = np.array([0] * 100 + [1] * 100)
    # target_recall=1.0 is impossible with random labels
    import logging
    with caplog.at_level(logging.WARNING):
        result = optimize_anomaly_threshold(scores, y_true, target_recall=1.0, max_far=0.0)
    assert "No threshold satisfies" in caplog.text
    assert result["threshold"] is not None


# ---------------------------------------------------------------------------
#  calibrate_health_score
# ---------------------------------------------------------------------------


def test_calibrate_health_score_far_above_threshold():
    """Score well above threshold → health ≈ 1."""
    score = calibrate_health_score(10.0, threshold=0.0, steepness=2.0)
    assert score > 0.999


def test_calibrate_health_score_far_below_threshold():
    """Score well below threshold → health ≈ 0."""
    score = calibrate_health_score(-10.0, threshold=0.0, steepness=2.0)
    assert score < 0.001


def test_calibrate_health_score_at_threshold():
    """Score equals threshold → health = 0.5."""
    score = calibrate_health_score(0.0, threshold=0.0, steepness=2.0)
    assert score == pytest.approx(0.5)


def test_calibrate_health_score_steepness():
    """Higher steepness should create a sharper transition."""
    s1 = calibrate_health_score(0.5, threshold=0.0, steepness=1.0)
    s2 = calibrate_health_score(0.5, threshold=0.0, steepness=5.0)
    # With steepness=5, health should be closer to 1 for same +0.5 offset
    assert s2 > s1


def test_calibrate_health_score_range():
    """Health should always be in [0, 1]."""
    for s in [-100.0, -1.0, 0.0, 1.0, 100.0]:
        h = calibrate_health_score(s, threshold=0.0, steepness=2.0)
        assert 0.0 <= h <= 1.0


# ---------------------------------------------------------------------------
#  calibrate_anomaly_model (integration)
# ---------------------------------------------------------------------------


def test_calibrate_anomaly_model_integration(tmp_path):
    """End-to-end: train IsolationForest, calibrate threshold on val split."""
    feat_path = _make_features_parquet(tmp_path, split="val", n_normal=50, n_faulty=50)

    # Train a quick IsolationForest on a different (fake) split, then calibrate on val
    # Simulate: the model was trained on train split, now we have a "val" DataFrame
    from aether_pdm.models.anomaly import META_COLS as ANOMALY_META_COLS

    df = pd.read_parquet(feat_path)
    feature_cols = [c for c in df.columns if c not in ANOMALY_META_COLS]
    x = df[feature_cols].values.astype(np.float64)

    model = IsolationForest(contamination=0.1, random_state=42, n_estimators=50)
    model.fit(x)

    result = calibrate_anomaly_model(
        model,
        features_path=feat_path,
        mlflow_uri=_mlflow_uri(tmp_path),
        split="val",
    )
    assert "threshold" in result
    assert "f1" in result
    assert isinstance(result["threshold"], float)


def test_calibrate_anomaly_model_empty_split(tmp_path):
    """Should raise when no rows match the requested split."""
    feat_path = _make_features_parquet(tmp_path, split="train", n_normal=20, n_faulty=10)
    model = IsolationForest(contamination=0.1, random_state=42, n_estimators=10)
    model.fit(np.random.randn(10, 4))

    with pytest.raises(ValueError, match="No samples found with split='val'"):
        calibrate_anomaly_model(
            model, features_path=feat_path,
            mlflow_uri=_mlflow_uri(tmp_path), split="val",
        )


def test_calibrate_anomaly_model_all_healthy(tmp_path):
    """Should raise when the val split has no fault samples."""
    rows = [
        {"rms": 0.1, "kurtosis": 3.0, "crest": 4.0, "skew": 0.0,
         "peak_freq": 60.0, "spec_energy": 0.5,
         "fault_type": "normal", "split": "val", "window_id": f"w{i}"}
        for i in range(10)
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "healthy_only.parquet"
    df.to_parquet(path, index=False)

    model = IsolationForest(contamination=0.1, random_state=42, n_estimators=10)
    model.fit(np.random.randn(10, 4))

    with pytest.raises(ValueError, match="All val samples are healthy"):
        calibrate_anomaly_model(
            model, features_path=path,
            mlflow_uri=_mlflow_uri(tmp_path), split="val",
        )


# ---------------------------------------------------------------------------
#  calibrate_fault_model (integration)
# ---------------------------------------------------------------------------


def test_calibrate_fault_model_integration(tmp_path):
    """End-to-end: train RandomForest, calibrate probabilities on val split."""
    feat_path = _make_features_parquet(tmp_path, split="val", n_normal=30, n_faulty=60)

    from aether_pdm.models.fault import META_COLS as FAULT_META_COLS

    df = pd.read_parquet(feat_path)
    df = df[df["fault_type"].isin(FAULT_LABELS)]
    feature_cols = [c for c in df.columns if c not in FAULT_META_COLS]
    x = df[feature_cols].values.astype(np.float64)

    le = LabelEncoder()
    y = le.fit_transform(df["fault_type"])

    model = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
    model.fit(x, y)

    calibrated_model, metrics = calibrate_fault_model(
        model, le,
        features_path=feat_path,
        mlflow_uri=_mlflow_uri(tmp_path),
        split="val",
    )
    # After calibration, predict_proba should still work
    probs = calibrated_model.predict_proba(x)
    assert probs.shape == (len(x), len(le.classes_))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert "f1_macro" in metrics


def test_calibrate_fault_model_empty_split(tmp_path):
    """Should raise when no rows match the requested split."""
    feat_path = _make_features_parquet(tmp_path, split="train", n_normal=10, n_faulty=10)
    x = np.random.randn(20, 4)
    y = np.array([0, 1, 2, 3] * 5)
    le = LabelEncoder().fit(FAULT_LABELS)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(x, y)

    with pytest.raises(ValueError, match="No samples found with split='val'"):
        calibrate_fault_model(model, le, features_path=feat_path,
                              mlflow_uri=_mlflow_uri(tmp_path), split="val")


def test_calibrate_fault_model_single_class(tmp_path):
    """Should raise when only one class is present in the calibration split."""
    rows = [
        {"rms": 0.1, "kurtosis": 3.0, "crest": 4.0, "skew": 0.0,
         "peak_freq": 60.0, "spec_energy": 0.5,
         "fault_type": "normal", "split": "val", "window_id": f"w{i}"}
        for i in range(10)
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "single_class.parquet"
    df.to_parquet(path, index=False)

    x = np.random.randn(20, 4)
    y = np.array([0, 1, 2, 3] * 5)
    le = LabelEncoder().fit(FAULT_LABELS)
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(x, y)

    with pytest.raises(ValueError, match="Only one class present"):
        calibrate_fault_model(
            model, le, features_path=path,
            mlflow_uri=_mlflow_uri(tmp_path), split="val",
        )


def test_health_score_scalar_inputs():
    """calibrate_health_score with scalar inputs should return a float."""
    h = calibrate_health_score(0.3, threshold=0.0, steepness=2.0)
    assert isinstance(h, float)
    assert 0.0 < h < 1.0
