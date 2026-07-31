"""Tests for drift monitors."""

import contextlib

import numpy as np
import pandas as pd
import pytest

from aether_pdm.ops.drift import (
    detect_drift,
    drift_status,
    feature_drift_report,
    ks_drift,
    psi,
    score_distribution_monitor,
)


def test_psi_identical_distributions():
    """PSI between identical distributions should be near zero."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 5000)
    prod = rng.normal(0, 1, 5000)
    assert psi(ref, prod) < 0.05


def test_psi_shifted_distributions():
    """PSI between shifted distributions should be high."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 5000)
    prod = rng.normal(3, 1, 5000)  # shifted mean
    assert psi(ref, prod) > 0.25


def test_psi_constant_arrays():
    """Constant arrays should not crash; return finite value."""
    ref = np.ones(100) * 5.0
    prod = np.ones(100) * 5.0
    result = psi(ref, prod)
    assert np.isfinite(result)


def test_psi_empty_raises():
    with pytest.raises(ValueError):
        psi(np.array([]), np.array([1, 2, 3]))


def test_psi_drops_nan_with_warning():
    """NaN values should be dropped with a warning, not crash."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 500)
    prod = rng.normal(0, 1, 500)
    ref[0] = np.nan
    prod[10] = np.nan
    with pytest.warns(UserWarning, match="non-finite"):
        result = psi(ref, prod)
    assert np.isfinite(result)


def test_psi_all_non_finite_raises():
    """Arrays containing only NaN/inf should raise ValueError."""
    with pytest.warns(UserWarning, match="non-finite"):
        with pytest.raises(ValueError, match="non-finite"):
            psi(np.array([np.nan, np.nan, np.nan]), np.array([1.0, 2.0, 3.0]))


def test_ks_drift_identical():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    prod = rng.normal(0, 1, 1000)
    result = ks_drift(ref, prod)
    assert result["drifted"] is False
    assert result["p_value"] > 0.05


def test_ks_drift_shifted():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 1000)
    prod = rng.normal(2, 1, 1000)
    result = ks_drift(ref, prod)
    assert result["drifted"] is True


def test_drift_status():
    assert drift_status(0.05) == "none"
    assert drift_status(0.15) == "moderate"
    assert drift_status(0.30) == "severe"


def test_feature_drift_report_detects_shift(tmp_path):
    """Report should flag a shifted feature as severe."""
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({
        "rms": rng.normal(0.1, 0.02, 500),
        "kurtosis": rng.normal(3.0, 0.5, 500),
        "fault_type": ["normal"] * 500,
    })
    prod = pd.DataFrame({
        "rms": rng.normal(0.4, 0.05, 500),  # shifted
        "kurtosis": rng.normal(3.0, 0.5, 500),
        "fault_type": ["inner_race"] * 500,
    })
    report = feature_drift_report(ref, prod)
    assert "rms" in report["feature"].values
    rms_row = report[report["feature"] == "rms"].iloc[0]
    assert rms_row["status"] == "severe"


def test_score_distribution_monitor_spike():
    rng = np.random.default_rng(42)
    ref = rng.normal(0.1, 0.02, 1000)   # healthy baseline scores
    prod = rng.normal(0.7, 0.1, 1000)   # elevated anomaly scores
    result = score_distribution_monitor(ref, prod)
    assert result["spike_detected"] is True


def test_detect_drift_end_to_end(tmp_path):
    """End-to-end drift detection on a features Parquet."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "rms": np.concatenate([rng.normal(0.1, 0.02, n), rng.normal(0.5, 0.05, n)]),
        "kurtosis": np.concatenate([rng.normal(3.0, 0.5, n), rng.normal(3.0, 0.5, n)]),
        "split": ["train"] * n + ["test"] * n,
    })
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    result = detect_drift(path, reference_split="train", production_split="test")
    assert "feature_drift_report" in result
    assert "drift_fired" in result
    assert "worst_feature" in result


def test_feature_drift_report_skips_non_numeric():
    """String/timestamp columns should be excluded from the report."""
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({
        "rms": rng.normal(0.1, 0.02, 300),
        "sensor_tag": ["sensor-A"] * 300,
        "installed_at": pd.date_range("2024-01-01", periods=300, freq="h"),
    })
    prod = pd.DataFrame({
        "rms": rng.normal(0.4, 0.05, 300),
        "sensor_tag": ["sensor-B"] * 300,
        "installed_at": pd.date_range("2025-01-01", periods=300, freq="h"),
    })
    report = feature_drift_report(ref, prod)
    assert list(report["feature"]) == ["rms"]
    assert "sensor_tag" not in report["feature"].values
    assert "installed_at" not in report["feature"].values


def test_feature_drift_report_different_columns():
    """Only the intersection of columns should be evaluated."""
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({
        "rms": rng.normal(0.1, 0.02, 300),
        "kurtosis": rng.normal(3.0, 0.5, 300),
    })
    prod = pd.DataFrame({
        "kurtosis": rng.normal(3.0, 0.5, 300),
        "crest_factor": rng.normal(4.0, 0.5, 300),
    })
    report = feature_drift_report(ref, prod)
    assert list(report["feature"]) == ["kurtosis"]


def test_feature_drift_report_rejects_non_dataframe():
    """Non-DataFrame inputs should raise TypeError."""
    with pytest.raises(TypeError, match="DataFrames"):
        feature_drift_report({"rms": [1.0]}, pd.DataFrame({"rms": [1.0]}))


def test_detect_drift_no_drift(tmp_path):
    """Both splits from the same distribution -> drift_fired False."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "rms": rng.normal(0.1, 0.02, 2 * n),
        "kurtosis": rng.normal(3.0, 0.5, 2 * n),
        "split": ["train"] * n + ["test"] * n,
    })
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    result = detect_drift(path, reference_split="train", production_split="test")
    assert result["drift_fired"] is False
    assert result["n_features_drifted"] == 0
    assert result["mean_psi"] < 0.25


def test_detect_drift_missing_split(tmp_path):
    """Missing 'split' column should raise ValueError with a clear message."""
    path = tmp_path / "no_split.parquet"
    pd.DataFrame({
        "rms": [0.1, 0.5],
        "kurtosis": [3.0, 8.0],
    }).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="no 'split' column"):
        detect_drift(path)


def test_detect_drift_empty_split_raises(tmp_path):
    """Empty reference or production split should raise ValueError."""
    path = tmp_path / "empty_split.parquet"
    pd.DataFrame({
        "rms": [0.1, 0.5],
        "split": ["train", "train"],
    }).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="split='test'"):
        detect_drift(path, reference_split="train", production_split="test")


def test_detect_drift_empty_reference_split_raises(tmp_path):
    """Empty reference split should raise ValueError."""
    path = tmp_path / "empty_ref.parquet"
    pd.DataFrame({
        "rms": [0.1, 0.5],
        "split": ["test", "test"],
    }).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="split='train'"):
        detect_drift(path, reference_split="train", production_split="test")


def test_detect_drift_missing_file_raises(tmp_path):
    """Missing features file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        detect_drift(tmp_path / "does_not_exist.parquet")


def test_detect_drift_no_common_features(tmp_path):
    """Only metadata columns shared -> graceful empty report, no drift fired."""
    n = 50
    df = pd.DataFrame({
        "fault_type": ["normal"] * n + ["inner_race"] * n,
        "split": ["train"] * n + ["test"] * n,
    })
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)

    with pytest.warns(UserWarning, match="No common numeric features"):
        result = detect_drift(path, reference_split="train", production_split="test")
    assert result["feature_drift_report"].empty
    assert result["mean_psi"] == 0.0
    assert result["worst_feature"] is None
    assert result["n_features_drifted"] == 0
    assert result["drift_fired"] is False


def test_log_drift_to_mlflow_mocked(monkeypatch):
    """Should log drift metrics and JSON feature PSIs to mocked MLflow."""
    import aether_pdm.ops.drift as drift_mod

    logged_metrics: dict = {}
    logged_params: dict = {}
    monkeypatch.setattr(drift_mod.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        drift_mod.mlflow, "start_run", lambda *args, **kwargs: contextlib.nullcontext()
    )
    monkeypatch.setattr(drift_mod.mlflow, "log_metrics", lambda d: logged_metrics.update(d))
    monkeypatch.setattr(drift_mod.mlflow, "log_param", lambda k, v: logged_params.update({k: v}))

    drift_result = {
        "mean_psi": 0.35,
        "worst_psi": 0.6,
        "worst_feature": "rms",
        "n_features_drifted": 2,
        "drift_fired": True,
        "feature_drift_report": pd.DataFrame({
            "feature": ["rms", "kurtosis"],
            "psi": [0.6, 0.1],
        }),
    }
    drift_mod.log_drift_to_mlflow(drift_result)

    assert logged_metrics["mean_psi"] == 0.35
    assert logged_metrics["n_features_drifted"] == 2.0
    assert logged_metrics["worst_psi"] == 0.6
    assert "rms" in logged_params["feature_drift"]
    assert logged_params["drift_fired"] == "True"
