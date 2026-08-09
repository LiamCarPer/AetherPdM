"""Smoke tests for the fresh-clone bootstrap demo (scripts/bootstrap_demo.py)."""

from pathlib import Path

import pytest


@pytest.mark.slow
def test_bootstrap_demo_smoke(tmp_path: Path, capsys) -> None:
    """Run the full bootstrap end-to-end with small params and assert success.

    Exercises the "interviewer clones the repo" path: synthetic data ->
    features -> train anomaly + fault -> promote (real MLflow sqlite store +
    GatedOps gate) -> "Bootstrap complete" banner. Uses tmp_path as workdir so
    no artifacts leak into the repo.

    Note: n_normal/n_faulty are deliberately above the bare minimum — with
    fewer healthy windows the IsolationForest healthy hull is too wide for the
    strict one-class boundary to separate the val faults (the promotion gate
    requires DR >= 0.80 AND FAR <= 0.10 on the val split).
    """
    from aether_pdm.signal.pipeline import FEATURE_VERSION
    from scripts import bootstrap_demo

    workdir = tmp_path / "demo"
    mlflow_uri = "sqlite:///" + (tmp_path / "mlflow_test.db").as_posix()

    rc = bootstrap_demo.main(
        [
            "--workdir", str(workdir),
            "--mlflow-uri", mlflow_uri,
            "--n-normal", "8",
            "--n-faulty", "10",
            "--n-estimators", "20",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0, f"bootstrap_demo.main() should exit 0, got {rc}"
    assert "Bootstrap complete." in out
    assert "promoted" in out

    # The demo must leave trainable artifacts behind in the workdir.
    features = workdir / "features" / f"features_{FEATURE_VERSION}.parquet"
    assert features.exists(), "bootstrap should produce a features parquet"
    # The MLflow sqlite store lands at the URI location (tmp_path/mlflow_test.db),
    # not necessarily inside the workdir — assert the one we passed exists.
    assert (tmp_path / "mlflow_test.db").exists(), "bootstrap should create a local mlflow db"
