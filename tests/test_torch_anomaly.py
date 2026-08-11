"""Tests for the PyTorch autoencoder anomaly detector.

All tests except ``test_package_import_does_not_load_torch`` are marked
``@pytest.mark.slow`` so the fast CI suite (``-m "not slow"``) only runs the
lazy-import contract check. Torch is imported inside the slow tests only —
module top-level stays torch-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aether_pdm.eval.metrics import detection_rate


def _make_separable(
    n_healthy: int = 200,
    n_shifted: int = 60,
    dim: int = 8,
    shift: float = 3.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic healthy N(0, 1) vs shifted N(shift, 1) feature windows."""
    rng = np.random.default_rng(seed)
    healthy = rng.normal(0.0, 1.0, size=(n_healthy, dim))
    shifted = rng.normal(shift, 1.0, size=(n_shifted, dim))
    return healthy, shifted


def test_package_import_does_not_load_torch():
    """Fast CI guard: importing the package must never import torch.

    Runs in a subprocess so torch loaded by OTHER tests in this session
    (e.g. via mlflow.pytorch in slow suites) cannot pollute sys.modules
    and false-fail the guard.
    """
    import subprocess

    code = (
        "import sys; "
        "import aether_pdm; "
        "assert 'torch' not in sys.modules, 'torch imported'; "
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


@pytest.mark.slow
def test_autoencoder_reconstructs_healthy():
    """Healthy windows should reconstruct far better than shifted ones."""
    from aether_pdm.models.torch_anomaly import TorchAnomalyDetector

    healthy, shifted = _make_separable()
    detector = TorchAnomalyDetector(
        input_dim=healthy.shape[1],
        hidden_dims=(8, 4),
        latent_dim=2,
        epochs=20,
        batch_size=16,
        seed=42,
    )
    detector.fit(healthy)

    healthy_scores = detector.anomaly_scores(healthy)
    shifted_scores = detector.anomaly_scores(shifted)

    assert np.all(np.isfinite(healthy_scores))
    assert np.all(np.isfinite(shifted_scores))
    assert shifted_scores.mean() > healthy_scores.mean() + 1.0
    assert healthy_scores.shape == (len(healthy),)
    assert shifted_scores.shape == (len(shifted),)


@pytest.mark.slow
def test_threshold_finds_target_recall():
    """Calibration must hit the requested detection rate on separable data."""
    from aether_pdm.models.torch_anomaly import TorchAnomalyDetector

    healthy, shifted = _make_separable(n_healthy=200, n_shifted=60)
    detector = TorchAnomalyDetector(
        input_dim=healthy.shape[1],
        hidden_dims=(8, 4),
        latent_dim=2,
        epochs=20,
        batch_size=16,
        seed=42,
    )
    detector.fit(healthy)

    x_val = np.vstack([healthy[:40], shifted])
    y_val = np.array([0] * 40 + [1] * 60, dtype=np.int64)

    target = 0.86
    threshold = detector.find_threshold(x_val, y_val, target_recall=target)
    preds = detector.predict(x_val, threshold)

    assert detection_rate(y_val, preds) >= target - 1e-9
    # Separable data: threshold should sit in the gap, not at the extremes.
    assert threshold > 0.0
    # FAR must be well below the 10% gate on separable data.
    from aether_pdm.eval.metrics import false_alarm_rate

    assert false_alarm_rate(y_val, preds) < 0.10


@pytest.mark.slow
def test_save_load_roundtrip(tmp_path):
    """Saved and reloaded detectors must produce identical scores."""
    from aether_pdm.models.torch_anomaly import TorchAnomalyDetector

    healthy, shifted = _make_separable()
    detector = TorchAnomalyDetector(
        input_dim=healthy.shape[1],
        hidden_dims=(8, 4),
        latent_dim=2,
        epochs=15,
        batch_size=16,
        seed=42,
    )
    detector.fit(healthy)

    path = tmp_path / "ae.pt"
    detector.save(path)
    assert path.exists()

    loaded = TorchAnomalyDetector.load(path)
    np.testing.assert_allclose(
        detector.anomaly_scores(shifted),
        loaded.anomaly_scores(shifted),
        rtol=1e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        detector.anomaly_scores(healthy),
        loaded.anomaly_scores(healthy),
        rtol=1e-5,
        atol=1e-8,
    )
    assert loaded.input_dim == detector.input_dim
    assert loaded.latent_dim == detector.latent_dim


@pytest.mark.slow
def test_deterministic_seed():
    """Same seed + same data must reproduce identical scores."""
    from aether_pdm.models.torch_anomaly import TorchAnomalyDetector

    healthy, _ = _make_separable(n_healthy=150)
    d1 = TorchAnomalyDetector(
        input_dim=healthy.shape[1],
        hidden_dims=(8, 4),
        latent_dim=2,
        epochs=12,
        batch_size=16,
        seed=42,
    )
    d2 = TorchAnomalyDetector(
        input_dim=healthy.shape[1],
        hidden_dims=(8, 4),
        latent_dim=2,
        epochs=12,
        batch_size=16,
        seed=42,
    )
    d1.fit(healthy)
    d2.fit(healthy)
    np.testing.assert_allclose(
        d1.anomaly_scores(healthy),
        d2.anomaly_scores(healthy),
        rtol=1e-5,
        atol=1e-8,
    )


@pytest.mark.slow
def test_benchmark_runs_and_reports(tmp_path):
    """End-to-end tiny benchmark (epochs=3) writes a report and returns dicts."""
    from aether_pdm.ops.benchmark_anomaly import (
        benchmark_anomaly_detectors,
        write_benchmark_report,
    )

    rng = np.random.default_rng(7)
    dim = 6
    feats = [f"f{i}" for i in range(dim)]

    def block(n: int, mu: float, fault: str, split: str) -> pd.DataFrame:
        data = rng.normal(mu, 1.0, size=(n, dim))
        frame = pd.DataFrame(data, columns=feats)
        frame["window_id"] = np.arange(len(frame))
        frame["fault_type"] = fault
        frame["split"] = split
        return frame

    cwru = pd.concat(
        [
            block(120, 0.0, "normal", "train"),
            block(40, 0.0, "normal", "val"),
            block(40, 4.0, "inner_race", "val"),
        ],
        ignore_index=True,
    )
    paderborn = pd.concat(
        [
            block(30, 0.0, "normal", "test"),
            block(30, 4.0, "outer_race", "test"),
        ],
        ignore_index=True,
    ).drop(columns=["f5"])  # missing feature -> exercises the imputation path
    cwru_path = tmp_path / "cwru.parquet"
    pb_path = tmp_path / "paderborn.parquet"
    cwru.to_parquet(cwru_path)
    paderborn.to_parquet(pb_path)

    uri = "sqlite:///" + str(tmp_path / "mlruns.db").replace("\\", "/")
    result = benchmark_anomaly_detectors(
        cwru_path, pb_path, mlflow_uri=uri, epochs=3
    )
    out = tmp_path / "report.md"
    write_benchmark_report(result, out)

    assert set(result) >= {"if_baseline", "torch", "torch_wins", "report_path"}
    assert isinstance(result["torch_wins"], bool)
    assert result["if_baseline"]["detection_rate"] >= 0.0
    assert result["if_baseline"]["false_alarm_rate"] >= 0.0
    assert result["torch"]["detection_rate"] >= 0.0
    assert result["torch"]["false_alarm_rate"] >= 0.0
    assert result["torch"]["paderborn"]["n"] == 60
    assert result["if_baseline"]["paderborn"]["n"] == 60
    assert result["paderborn_imputed_columns"] == ["f5"]
    assert result["report_path"] == str(out)
    assert out.exists()

    report = out.read_text(encoding="utf-8")
    assert "torch_wins" in report
    assert "Domain Shift Probe" in report
