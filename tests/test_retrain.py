"""Tests for retrain pipeline orchestration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from aether_pdm.ops.retrain import retrain_models, run_retrain_pipeline, should_retrain


def _write_features(tmp_path: Path, with_drift: bool = False) -> Path:
    """Write features Parquet; with_drift shifts test split to trigger drift."""
    rng = np.random.default_rng(42)
    n = 200
    if with_drift:
        rms_prod = rng.normal(0.5, 0.05, n)
    else:
        rms_prod = rng.normal(0.1, 0.02, n)
    df = pd.DataFrame(
        {
            "rms": np.concatenate([rng.normal(0.1, 0.02, n), rms_prod]),
            "kurtosis": np.concatenate(
                [rng.normal(3.0, 0.5, n), rng.normal(3.0, 0.5, n)]
            ),
            "split": ["train"] * n + ["test"] * n,
        }
    )
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return path


def _no_drift_decision() -> dict:
    return {
        "retrain": False,
        "reason": "no_drift",
        "drift": {"mean_psi": 0.01, "worst_feature": "none", "n_features_drifted": 0},
    }


def _drift_decision() -> dict:
    return {
        "retrain": True,
        "reason": "drift_fired",
        "drift": {"mean_psi": 0.35, "worst_feature": "rms", "n_features_drifted": 1},
    }


def _retrain_result() -> dict:
    return {
        "anomaly_trained": True,
        "fault_trained": True,
        "anomaly_model_type": "IsolationForest",
        "fault_model_type": "RandomForest",
    }


def _promoted_anomaly() -> dict:
    return {
        "candidate_version": 2,
        "decision": "promoted",
        "reason": "ok",
        "metrics": {"detection_rate": 0.9, "false_alarm_rate": 0.05},
    }


def _promoted_fault() -> dict:
    return {
        "candidate_version": 2,
        "decision": "promoted",
        "reason": "ok",
        "metrics": {"f1_macro": 0.95, "balanced_accuracy": 0.94},
    }


def _rejected_anomaly() -> dict:
    return {
        "candidate_version": 2,
        "decision": "rejected",
        "reason": "far 0.3 > 0.10",
        "metrics": {"detection_rate": 0.9, "false_alarm_rate": 0.3},
    }


def _rejected_fault() -> dict:
    return {
        "candidate_version": 2,
        "decision": "rejected",
        "reason": "f1_macro 0.7 < 0.90",
        "metrics": {"f1_macro": 0.7},
    }


# ---------------------------------------------------------------------------
# should_retrain
# ---------------------------------------------------------------------------


def test_should_retrain_no_drift(tmp_path):
    feat_path = _write_features(tmp_path, with_drift=False)
    result = should_retrain(feat_path)
    assert result["retrain"] is False
    assert result["reason"] == "no_drift"


def test_should_retrain_drift(tmp_path):
    feat_path = _write_features(tmp_path, with_drift=True)
    result = should_retrain(feat_path)
    assert result["retrain"] is True
    assert result["reason"] == "drift_fired"


def test_should_retrain_force(tmp_path):
    feat_path = _write_features(tmp_path, with_drift=False)
    result = should_retrain(feat_path, force=True)
    assert result["retrain"] is True
    assert result["reason"] == "forced"


def test_should_retrain_missing_features_file(tmp_path):
    """Missing features file -> FileNotFoundError propagates."""
    missing = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        should_retrain(missing)


def test_should_retrain_missing_split_column(tmp_path):
    """Features without a split column -> ValueError wrapped with context."""
    df = pd.DataFrame({"rms": [0.1, 0.2, 0.3]})
    path = tmp_path / "no_split.parquet"
    df.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Drift check failed"):
        should_retrain(path)


# ---------------------------------------------------------------------------
# retrain_models
# ---------------------------------------------------------------------------


@patch("aether_pdm.ops.retrain.train_anomaly")
@patch("aether_pdm.ops.retrain.train_fault_classifier")
def test_retrain_models_calls_trainers(mock_fault, mock_anomaly, tmp_path):
    """retrain_models should call both trainers with split=train."""
    feat_path = _write_features(tmp_path)
    mock_anomaly.return_value = MagicMock()
    mock_fault.return_value = (MagicMock(), MagicMock())
    result = retrain_models(feat_path, mlflow_uri="sqlite:///test.db")
    assert result["anomaly_trained"] is True
    assert result["fault_trained"] is True
    mock_anomaly.assert_called_once()
    mock_fault.assert_called_once()
    # Verify split passed to trainers
    _, kwargs = mock_anomaly.call_args
    assert kwargs.get("split") == "train"


@patch("aether_pdm.ops.retrain.train_anomaly")
@patch("aether_pdm.ops.retrain.train_fault_classifier")
def test_retrain_models_propagates_trainer_error(mock_fault, mock_anomaly, tmp_path):
    """train_anomaly raising ValueError should propagate, not be swallowed."""
    feat_path = _write_features(tmp_path)
    mock_anomaly.side_effect = ValueError(
        "No healthy (normal) samples found in the dataset"
    )
    with pytest.raises(ValueError, match="healthy"):
        retrain_models(feat_path, mlflow_uri="sqlite:///test.db")
    mock_fault.assert_not_called()


# ---------------------------------------------------------------------------
# run_retrain_pipeline
# ---------------------------------------------------------------------------


@patch("aether_pdm.ops.retrain.promote_anomaly")
@patch("aether_pdm.ops.retrain.promote_fault")
@patch("aether_pdm.ops.retrain.retrain_models")
@patch("aether_pdm.ops.retrain.should_retrain")
def test_run_pipeline_skips_when_no_drift(
    mock_drift, mock_retrain, mock_promote_f, mock_promote_a, tmp_path
):
    """No drift -> skip, no trainers or promoters called."""
    feat_path = _write_features(tmp_path, with_drift=False)
    mock_drift.return_value = _no_drift_decision()
    result = run_retrain_pipeline(feat_path, mlflow_uri="sqlite:///test.db")
    assert result["skipped"] is True
    assert result["outcome"] == "skipped"
    mock_retrain.assert_not_called()
    mock_promote_a.assert_not_called()
    mock_promote_f.assert_not_called()


@patch("aether_pdm.ops.retrain.promote_anomaly")
@patch("aether_pdm.ops.retrain.promote_fault")
@patch("aether_pdm.ops.retrain.retrain_models")
@patch("aether_pdm.ops.retrain.should_retrain")
def test_run_pipeline_promotes_both(
    mock_drift, mock_retrain, mock_promote_f, mock_promote_a, tmp_path
):
    """Drift fired, both gates pass -> outcome=promoted."""
    feat_path = _write_features(tmp_path, with_drift=True)
    mock_drift.return_value = _drift_decision()
    mock_retrain.return_value = _retrain_result()
    mock_promote_a.return_value = _promoted_anomaly()
    mock_promote_f.return_value = _promoted_fault()
    result = run_retrain_pipeline(feat_path, mlflow_uri="sqlite:///test.db")
    assert result["skipped"] is False
    assert result["retrained"] is True
    assert result["outcome"] == "promoted"
    assert result["anomaly"]["decision"] == "promoted"
    assert result["fault"]["decision"] == "promoted"


@patch("aether_pdm.ops.retrain.promote_anomaly")
@patch("aether_pdm.ops.retrain.promote_fault")
@patch("aether_pdm.ops.retrain.retrain_models")
@patch("aether_pdm.ops.retrain.should_retrain")
def test_run_pipeline_partial_promotion(
    mock_drift, mock_retrain, mock_promote_f, mock_promote_a, tmp_path
):
    """One gate passes, one rejects -> outcome=partial (implicit rollback for rejected)."""
    feat_path = _write_features(tmp_path, with_drift=True)
    mock_drift.return_value = _drift_decision()
    mock_retrain.return_value = _retrain_result()
    mock_promote_a.return_value = _promoted_anomaly()
    mock_promote_f.return_value = _rejected_fault()
    result = run_retrain_pipeline(feat_path, mlflow_uri="sqlite:///test.db")
    assert result["outcome"] == "partial"
    assert result["fault"]["decision"] == "rejected"


@patch("aether_pdm.ops.retrain.promote_anomaly")
@patch("aether_pdm.ops.retrain.promote_fault")
@patch("aether_pdm.ops.retrain.retrain_models")
@patch("aether_pdm.ops.retrain.should_retrain")
def test_run_pipeline_all_rejected(
    mock_drift, mock_retrain, mock_promote_f, mock_promote_a, tmp_path
):
    """Both gates reject -> outcome=rejected (rollback, keep previous production)."""
    feat_path = _write_features(tmp_path, with_drift=True)
    mock_drift.return_value = _drift_decision()
    mock_retrain.return_value = _retrain_result()
    mock_promote_a.return_value = _rejected_anomaly()
    mock_promote_f.return_value = _rejected_fault()
    result = run_retrain_pipeline(feat_path, mlflow_uri="sqlite:///test.db")
    assert result["outcome"] == "rejected"


@patch("aether_pdm.ops.retrain.promote_anomaly")
@patch("aether_pdm.ops.retrain.promote_fault")
@patch("aether_pdm.ops.retrain.retrain_models")
@patch("aether_pdm.ops.retrain.should_retrain")
def test_run_pipeline_force_retrains(
    mock_drift, mock_retrain, mock_promote_f, mock_promote_a, tmp_path
):
    """force=True triggers retrain even without drift."""
    feat_path = _write_features(tmp_path, with_drift=False)
    mock_drift.return_value = {
        "retrain": True,
        "reason": "forced",
        "drift": {"mean_psi": 0.02, "worst_feature": "none", "n_features_drifted": 0},
    }
    mock_retrain.return_value = _retrain_result()
    mock_promote_a.return_value = _promoted_anomaly()
    mock_promote_f.return_value = _promoted_fault()
    result = run_retrain_pipeline(feat_path, mlflow_uri="sqlite:///test.db", force=True)
    assert result["skipped"] is False
    assert result["outcome"] == "promoted"
    mock_retrain.assert_called_once()


@patch("aether_pdm.ops.retrain.promote_anomaly")
@patch("aether_pdm.ops.retrain.promote_fault")
@patch("aether_pdm.ops.retrain.retrain_models")
@patch("aether_pdm.ops.retrain.should_retrain")
def test_run_pipeline_handles_promote_error(
    mock_drift, mock_retrain, mock_promote_f, mock_promote_a, tmp_path
):
    """promote_anomaly raising ValueError is recorded, not fatal."""
    feat_path = _write_features(tmp_path, with_drift=True)
    mock_drift.return_value = _drift_decision()
    mock_retrain.return_value = _retrain_result()
    mock_promote_a.side_effect = ValueError("no candidate model found")
    mock_promote_f.return_value = _promoted_fault()
    result = run_retrain_pipeline(feat_path, mlflow_uri="sqlite:///test.db")
    assert result["skipped"] is False
    assert result["retrained"] is True
    assert result["anomaly_error"] == "no candidate model found"
    assert result["anomaly"]["decision"] == "error"
    assert result["outcome"] == "partial"
