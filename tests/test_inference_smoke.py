"""Smoke test for the inference engine with real models."""

import pytest

from aether_pdm.data.synthetic import synthetic_waveform


@pytest.mark.skip(reason="Requires trained models in MLflow (run train.py first)")
def test_inference_engine_loads():
    from aether_pdm.serve.inference import InferenceEngine
    engine = InferenceEngine()
    assert engine.anomaly_model is not None
    assert engine.fault_model is not None


@pytest.mark.skip(reason="Requires trained models in MLflow")
def test_inference_engine_scores_normal():
    from aether_pdm.serve.inference import InferenceEngine
    engine = InferenceEngine()
    w = synthetic_waveform(length=4096, rpm=1772, fault_type="normal", seed=42)
    result = engine.score(w, sampling_rate=12000, rpm=1772)
    assert result["health_score"] >= 0.5
    assert result["alert"]["level"] == "healthy"


@pytest.mark.skip(reason="Requires trained models in MLflow")
def test_inference_engine_detects_fault():
    from aether_pdm.serve.inference import InferenceEngine
    engine = InferenceEngine()
    w = synthetic_waveform(length=4096, rpm=1772, fault_type="inner_race", fault_diameter=0.021, seed=42)
    result = engine.score(w, sampling_rate=12000, rpm=1772)
    # Faulty signal should trigger a non-healthy alert
    assert result["alert"]["level"] in ("warning", "critical")
