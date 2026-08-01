"""Tests for the scheduled ops pipeline."""

from unittest.mock import MagicMock, patch

import pytest

from aether_pdm.ops.scheduler import main, run_scheduled_pipeline


@pytest.fixture(autouse=True)
def _hermetic_db():
    """Keep scheduler tests off the real dev database.

    ``run_scheduled_pipeline`` calls ``init_db()`` and ``get_session()``; both
    are patched so tests never touch ``data/aether_pdm.db``. ``BatchScorer``
    is mocked in every test below, so the yielded session is never used.
    """
    with (
        patch("aether_pdm.ops.scheduler.init_db"),
        patch("aether_pdm.ops.scheduler.get_session") as mock_session_cm,
    ):
        mock_session_cm.return_value.__enter__.return_value = MagicMock()
        yield


@patch("aether_pdm.ops.scheduler.run_retrain_pipeline")
@patch("aether_pdm.ops.scheduler.detect_drift")
@patch("aether_pdm.ops.scheduler.BatchScorer")
def test_pipeline_full_loop(mock_scorer, mock_drift, mock_retrain, tmp_path):
    mock_scorer.return_value.run.return_value = {"scored": 2, "alerts_raised": 1,
                                                 "alerts_suppressed_by_cooldown": 0,
                                                 "alerts_suppressed_by_hysteresis": 0,
                                                 "errors": [], "results": []}
    mock_drift.return_value = {"drift_fired": True, "mean_psi": 0.35, "worst_feature": "rms",
                               "n_features_drifted": 1, "feature_drift_report": []}
    mock_retrain.return_value = {
        "skipped": False,
        "outcome": "promoted",
        "anomaly": {"decision": "promoted"},
        "fault": {"decision": "promoted"},
    }

    features = tmp_path / "features.parquet"
    features.write_bytes(b"dummy")

    result = run_scheduled_pipeline(features, org="acme", mlflow_uri="sqlite:///test.db")
    assert result["batch"]["scored"] == 2
    assert result["drift"]["drift_fired"] is True
    assert result["retrain"]["outcome"] == "promoted"
    assert result["summary"]["retrained"] is True
    assert result["summary"]["promoted"] is True
    mock_retrain.assert_called_once()


@patch("aether_pdm.ops.scheduler.run_retrain_pipeline")
@patch("aether_pdm.ops.scheduler.detect_drift")
@patch("aether_pdm.ops.scheduler.BatchScorer")
def test_pipeline_no_drift_skips_retrain(mock_scorer, mock_drift, mock_retrain, tmp_path):
    """No drift -> retrain stage skipped inside run_retrain_pipeline.

    With Option B the scheduler delegates to run_retrain_pipeline whenever
    retrain=True; the pipeline's internal should_retrain() returns
    skipped=True when there is no drift.
    """
    mock_scorer.return_value.run.return_value = {"scored": 0, "alerts_raised": 0,
                                                 "alerts_suppressed_by_cooldown": 0,
                                                 "alerts_suppressed_by_hysteresis": 0,
                                                 "errors": [], "results": []}
    mock_drift.return_value = {"drift_fired": False, "mean_psi": 0.01,
                               "worst_feature": "none", "n_features_drifted": 0,
                               "feature_drift_report": []}
    mock_retrain.return_value = {"skipped": True, "skip_reason": "no_drift",
                                 "retrained": False, "outcome": "skipped"}

    features = tmp_path / "features.parquet"
    features.write_bytes(b"dummy")

    result = run_scheduled_pipeline(features, org="acme", mlflow_uri="sqlite:///test.db")
    assert result["drift"]["drift_fired"] is False
    assert result["retrain"]["skipped"] is True
    assert result["summary"]["retrained"] is False
    mock_retrain.assert_called_once()


@patch("aether_pdm.ops.scheduler.BatchScorer")
def test_pipeline_missing_features(mock_scorer, tmp_path):
    """Missing features file -> batch still runs, drift is an error, no crash."""
    mock_scorer.return_value.run.return_value = {"scored": 0, "alerts_raised": 0,
                                                 "alerts_suppressed_by_cooldown": 0,
                                                 "alerts_suppressed_by_hysteresis": 0,
                                                 "errors": [], "results": []}
    result = run_scheduled_pipeline(tmp_path / "missing.parquet", org="acme", mlflow_uri="sqlite:///test.db")
    assert result["batch"]["scored"] == 0
    assert "error" in result["drift"]
    assert result["retrain"]["skipped"] is True
    assert result["summary"]["retrained"] is False


@patch("aether_pdm.ops.scheduler.run_retrain_pipeline")
@patch("aether_pdm.ops.scheduler.detect_drift")
@patch("aether_pdm.ops.scheduler.BatchScorer")
def test_pipeline_retrain_disabled(mock_scorer, mock_drift, mock_retrain, tmp_path):
    """retrain=False skips retraining even when drift fired."""
    mock_scorer.return_value.run.return_value = {"scored": 1, "alerts_raised": 0,
                                                 "alerts_suppressed_by_cooldown": 0,
                                                 "alerts_suppressed_by_hysteresis": 0,
                                                 "errors": [], "results": []}
    mock_drift.return_value = {"drift_fired": True, "mean_psi": 0.4, "worst_feature": "rms",
                               "n_features_drifted": 1, "feature_drift_report": []}

    features = tmp_path / "features.parquet"
    features.write_bytes(b"dummy")

    result = run_scheduled_pipeline(
        features, org="acme", mlflow_uri="sqlite:///test.db", retrain=False
    )
    assert result["drift"]["drift_fired"] is True
    assert result["retrain"]["skipped"] is True
    assert result["summary"]["retrained"] is False
    mock_retrain.assert_not_called()


@patch("aether_pdm.ops.scheduler.run_retrain_pipeline")
@patch("aether_pdm.ops.scheduler.detect_drift")
@patch("aether_pdm.ops.scheduler.BatchScorer")
def test_pipeline_retrains_with_custom_drift_threshold(
    mock_scorer, mock_drift, mock_retrain, tmp_path
):
    """Custom drift_threshold below the drift_fired hardcode still triggers retrain."""
    mock_scorer.return_value.run.return_value = {"scored": 1, "alerts_raised": 0,
                                                 "alerts_suppressed_by_cooldown": 0,
                                                 "alerts_suppressed_by_hysteresis": 0,
                                                 "errors": [], "results": []}
    # drift_fired=False (hardcoded 0.25), but mean_psi 0.22 >= custom threshold 0.20
    mock_drift.return_value = {"drift_fired": False, "mean_psi": 0.22, "worst_feature": "rms",
                               "n_features_drifted": 0, "feature_drift_report": []}
    mock_retrain.return_value = {
        "skipped": False,
        "outcome": "promoted",
        "anomaly": {"decision": "promoted"},
        "fault": {"decision": "promoted"},
    }

    features = tmp_path / "features.parquet"
    features.write_bytes(b"dummy")

    result = run_scheduled_pipeline(features, org="acme", mlflow_uri="sqlite:///test.db",
                                    drift_threshold=0.20)
    # With Option B, run_retrain_pipeline is called whenever retrain=True
    mock_retrain.assert_called_once()
    # drift_threshold is forwarded so the internal should_retrain() honors it
    mock_retrain.assert_called_once_with(features, mlflow_uri="sqlite:///test.db",
                                         drift_threshold=0.20)
    assert result["retrain"]["outcome"] == "promoted"


@patch("aether_pdm.ops.scheduler.run_retrain_pipeline")
@patch("aether_pdm.ops.scheduler.detect_drift")
@patch("aether_pdm.ops.scheduler.BatchScorer")
def test_pipeline_no_assets(mock_scorer, mock_drift, mock_retrain, tmp_path):
    """No assets -> scored=0, pipeline still returns (drift still evaluated)."""
    mock_scorer.return_value.run.return_value = {"scored": 0, "alerts_raised": 0,
                                                 "alerts_suppressed_by_cooldown": 0,
                                                 "alerts_suppressed_by_hysteresis": 0,
                                                 "errors": [], "results": []}
    mock_drift.return_value = {"drift_fired": False, "mean_psi": 0.02,
                               "worst_feature": "none", "n_features_drifted": 0,
                               "feature_drift_report": []}
    mock_retrain.return_value = {"skipped": True, "skip_reason": "no_drift",
                                 "retrained": False, "outcome": "skipped"}

    features = tmp_path / "features.parquet"
    features.write_bytes(b"dummy")

    result = run_scheduled_pipeline(features, org="acme", mlflow_uri="sqlite:///test.db")
    assert result["batch"]["scored"] == 0
    assert result["summary"]["assets_scored"] == 0
    assert result["drift"]["drift_fired"] is False
    mock_retrain.assert_called_once()


@patch("aether_pdm.ops.scheduler.run_retrain_pipeline")
@patch("aether_pdm.ops.scheduler.detect_drift")
@patch("aether_pdm.ops.scheduler.BatchScorer")
def test_pipeline_drift_detection_error(mock_scorer, mock_drift, mock_retrain, tmp_path):
    """detect_drift raising on an unreadable file is non-fatal.

    Batch still ran and was persisted, drift is an error dict, and the
    retrain stage is skipped without crashing the pipeline.
    """
    mock_scorer.return_value.run.return_value = {"scored": 1, "alerts_raised": 0,
                                                 "alerts_suppressed_by_cooldown": 0,
                                                 "alerts_suppressed_by_hysteresis": 0,
                                                 "errors": [], "results": []}
    mock_drift.side_effect = RuntimeError("corrupt parquet")

    features = tmp_path / "features.parquet"
    features.write_bytes(b"dummy")

    result = run_scheduled_pipeline(features, org="acme", mlflow_uri="sqlite:///test.db")
    assert result["batch"]["scored"] == 1
    assert "error" in result["drift"]
    assert result["retrain"]["skipped"] is True
    assert result["summary"]["retrained"] is False
    mock_retrain.assert_not_called()


@patch("aether_pdm.ops.scheduler.run_scheduled_pipeline")
def test_cli_main_failure_exits_1(mock_pipeline, capsys):
    """main() exits 1 when the pipeline raises (cron-safe failure exit)."""
    mock_pipeline.side_effect = RuntimeError("boom")
    with (
        patch("sys.argv", ["run_ops_pipeline.py", "--features", "x.parquet"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


@patch("aether_pdm.ops.scheduler.run_scheduled_pipeline")
def test_cli_main_smoke(mock_pipeline):
    """main() parses args, prints a summary, exits 0 (cron-safe)."""
    mock_pipeline.return_value = {
        "batch": {"scored": 2, "alerts_raised": 1},
        "drift": {"drift_fired": False},
        "retrain": {"skipped": True, "reason": "no_drift_or_retrain_disabled"},
        "summary": {"assets_scored": 2, "alerts_raised": 1, "drift_fired": False,
                    "retrained": False, "promoted": False},
    }
    with (
        patch("sys.argv", ["run_ops_pipeline.py", "--features", "x.parquet", "--org", "acme"]),
        pytest.raises(SystemExit) as exc_info,
    ):
        main()
    assert exc_info.value.code == 0
    mock_pipeline.assert_called_once()
