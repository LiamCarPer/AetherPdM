"""Tests for ONNX edge export + validation.

Fast suite (``-m "not slow"``) runs only ``test_import_lazy_torch``: the
export module must stay torch-free and heavy-import-free on import. All ONNX
roundtrip tests are ``@pytest.mark.slow`` and run in the CI smoke step.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def test_import_lazy_torch():
    """Fast CI guard: importing the export module must not load torch/onnx.

    Runs in a subprocess so torch imported by OTHER tests in this session
    (slow suites) cannot pollute sys.modules and false-fail the guard.
    """
    import subprocess

    code = (
        "import sys; "
        "import aether_pdm.ops.export_edge; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
        "assert 'onnx' not in sys.modules, 'onnx imported'; "
        "assert 'onnxruntime' not in sys.modules, 'onnxruntime imported'; "
        "assert 'skl2onnx' not in sys.modules, 'skl2onnx imported'; "
        "print('lazy import OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        timeout=120,
    )
    assert result.returncode == 0, f"lazy import failed:\n{result.stderr}"
    assert "lazy import OK" in result.stdout


def _fit_tiny_detector(seed: int = 3, dim: int = 6, n: int = 120):
    """Tiny fitted TorchAnomalyDetector on synthetic healthy windows."""
    from aether_pdm.models.torch_anomaly import TorchAnomalyDetector

    rng = np.random.default_rng(seed)
    healthy = rng.normal(size=(n, dim))
    detector = TorchAnomalyDetector(
        input_dim=dim,
        hidden_dims=(8, 4),
        latent_dim=2,
        epochs=8,
        batch_size=16,
        seed=42,
    )
    detector.fit(healthy)
    return detector


@pytest.mark.slow
def test_torch_onnx_roundtrip(tmp_path):
    """Exported torch ONNX must reproduce the module's float32 outputs."""
    import torch

    from aether_pdm.ops.export_edge import export_model_onnx, validate_onnx

    detector = _fit_tiny_detector()
    onnx_path = tmp_path / "ae.onnx"
    export_model_onnx(detector, onnx_path, detector.input_dim, model_type="torch")
    assert onnx_path.exists()

    rng = np.random.default_rng(9)
    sample = rng.normal(size=(8, detector.input_dim)).astype(np.float32)
    module = detector.torch_module
    module.eval()
    with torch.no_grad():
        ref = module(torch.from_numpy(sample)).numpy()

    result = validate_onnx(onnx_path, sample, ref, atol=1e-4)
    assert result["passed"], result
    assert result["max_abs_err"] <= 1e-4
    assert result["runtime_ms"] >= 0.0


@pytest.mark.slow
def test_edge_bundle_writes_files(tmp_path):
    """Edge bundle writes model.onnx + edge_scorer.py that reproduce scores."""
    from aether_pdm.ops.export_edge import export_edge_bundle

    detector = _fit_tiny_detector()
    bundle = export_edge_bundle(detector, tmp_path, detector.input_dim, model_type="torch")
    onnx_path = Path(bundle["onnx_path"])
    scorer_path = Path(bundle["scorer_path"])
    assert onnx_path.exists()
    assert scorer_path.exists()

    # The generated scorer is standalone: import it from its own file and
    # check its scores reproduce TorchAnomalyDetector.anomaly_scores.
    spec = importlib.util.spec_from_file_location("edge_scorer", scorer_path)
    assert spec is not None and spec.loader is not None
    edge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edge)

    scorer = edge.EdgeAnomalyScorer()
    rng = np.random.default_rng(13)
    sample = rng.normal(size=(5, detector.input_dim))
    scores = np.array([scorer.score(row) for row in sample])
    ref = detector.anomaly_scores(sample)
    np.testing.assert_allclose(scores, ref, atol=1e-4)


@pytest.mark.slow
def test_sklearn_onnx_roundtrip(tmp_path):
    """IsolationForest -> ONNX (skl2onnx) reproduces decision_function."""
    from sklearn.ensemble import IsolationForest

    from aether_pdm.ops.export_edge import (
        export_edge_bundle,
        export_model_onnx,
        validate_onnx,
    )

    rng = np.random.default_rng(11)
    dim = 5
    x = rng.normal(size=(150, dim))
    model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
    model.fit(x)

    onnx_path = tmp_path / "if.onnx"
    export_model_onnx(model, onnx_path, dim, model_type="sklearn")
    assert onnx_path.exists()

    sample = rng.normal(size=(10, dim))
    ref = model.decision_function(sample)
    result = validate_onnx(onnx_path, sample, ref, atol=1e-4)
    assert result["passed"], result
    assert result["max_abs_err"] <= 1e-4

    # The generated sklearn edge scorer must emit the same decision function.
    bundle = export_edge_bundle(model, tmp_path / "if_bundle", dim, model_type="sklearn")
    spec = importlib.util.spec_from_file_location(
        "if_edge_scorer", Path(bundle["scorer_path"])
    )
    assert spec is not None and spec.loader is not None
    edge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(edge)
    scorer = edge.EdgeAnomalyScorer()
    np.testing.assert_allclose(scorer.scores(sample), ref, atol=1e-4)
