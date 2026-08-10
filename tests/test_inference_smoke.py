"""Smoke tests for the inference engine with REAL models (bootstrap-trained).

The bootstrap (scripts/bootstrap_demo.py) trains + promotes models into a
local sqlite MLflow store using synthetic data (deterministic seeds). These
tests load the InferenceEngine against that store and score real waveforms,
proving the full fresh-clone path: bootstrap -> train -> promote -> serve -> score.

The fixture is session-scoped so the ~60s bootstrap runs ONCE for all tests.
"""

import pytest

from aether_pdm.data.synthetic import synthetic_waveform


@pytest.fixture(scope="session")
def bootstrap_mlflow_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Run the fresh-clone bootstrap once, return the MLflow sqlite URI."""
    from scripts import bootstrap_demo

    workdir = tmp_path_factory.mktemp("bootstrap")
    mlflow_uri = "sqlite:///" + (workdir / "mlflow_test.db").as_posix()

    rc = bootstrap_demo.main(
        [
            "--workdir", str(workdir),
            "--mlflow-uri", mlflow_uri,
            "--n-normal", "8",
            "--n-faulty", "10",
            "--n-estimators", "20",
        ]
    )
    assert rc == 0, f"bootstrap_demo.main() should exit 0, got {rc}"
    return mlflow_uri


@pytest.fixture(scope="session")
def engine(bootstrap_mlflow_uri: str):
    from aether_pdm.serve.inference import InferenceEngine

    eng = InferenceEngine(mlflow_uri=bootstrap_mlflow_uri)
    assert eng.model_available, "engine should have loaded models from bootstrap MLflow"
    return eng


@pytest.mark.slow
def test_inference_engine_loads(engine):
    """Engine loads BOTH models from the bootstrap MLflow store."""
    assert engine.anomaly_model is not None
    assert engine.fault_model is not None
    assert hasattr(engine, "anomaly_version")
    assert hasattr(engine, "fault_version")


@pytest.mark.slow
def test_inference_engine_scores_normal(engine):
    """A normal waveform scores healthy."""
    w = synthetic_waveform(length=4096, rpm=1772, fault_type="normal", seed=42)
    result = engine.score(w, sampling_rate=12000, rpm=1772)
    assert result["health_score"] >= 0.5
    assert result["alert"]["level"] == "healthy"
    assert result["fault"]["class"] == "normal"
    assert "lineage" in result  # GatedOps lineage echoed on every score


@pytest.mark.slow
def test_inference_engine_detects_fault(engine):
    """A fault waveform triggers a non-healthy alert + fault class."""
    w = synthetic_waveform(
        length=4096,
        rpm=1772,
        fault_type="inner_race",
        fault_diameter=0.021,
        seed=42,
    )
    result = engine.score(w, sampling_rate=12000, rpm=1772)
    assert result["alert"]["level"] in ("warning", "critical")
    assert result["fault"]["class"] != "normal"
    assert result["top_features"]  # top-5 features present
