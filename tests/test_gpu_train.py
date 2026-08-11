"""Tests for the GPU/CPU torch anomaly training entrypoint.

Fast suite (``-m "not slow"``) runs only ``test_device_selection_cpu``
(device resolution is probed via a monkeypatched ``_cuda_available`` so no
CUDA hardware and no torch import are needed). The CPU-fallback training
test is ``@pytest.mark.slow`` and runs in the CI smoke step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _features_frame(
    tmp_path,
    seed: int = 5,
    dim: int = 6,
    n_train: int = 60,
    n_val_normal: int = 20,
    n_val_fault: int = 20,
):
    """Tiny separable feature Parquet: healthy train/val + shifted fault val."""
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(dim)]

    def block(n: int, mu: float, fault: str, split: str) -> pd.DataFrame:
        data = rng.normal(mu, 1.0, size=(n, dim))
        frame = pd.DataFrame(data, columns=cols)
        frame["fault_type"] = fault
        frame["split"] = split
        return frame

    frame = pd.concat(
        [
            block(n_train, 0.0, "normal", "train"),
            block(n_val_normal, 0.0, "normal", "val"),
            block(n_val_fault, 4.0, "inner_race", "val"),
        ],
        ignore_index=True,
    )
    path = tmp_path / "features.parquet"
    frame.to_parquet(path)
    return path


def test_device_selection_cpu(monkeypatch):
    """auto -> cpu when CUDA is unavailable; explicit values pass through."""
    import aether_pdm.ops.gpu_train as gt

    monkeypatch.setattr(gt, "_cuda_available", lambda: False)
    assert gt.resolve_device("auto") == "cpu"
    assert gt.resolve_device("cpu") == "cpu"
    assert gt.resolve_device("cuda") == "cuda"

    monkeypatch.setattr(gt, "_cuda_available", lambda: True)
    assert gt.resolve_device("auto") == "cuda"

    with pytest.raises(ValueError):
        gt.resolve_device("tpu")


def test_explicit_cuda_raises_without_gpu(monkeypatch):
    """device='cuda' without CUDA must raise, not silently fall back."""
    from pathlib import Path

    import aether_pdm.ops.gpu_train as gt

    monkeypatch.setattr(gt, "_cuda_available", lambda: False)
    # The device guard fires before any file is read, so a bogus path is fine.
    with pytest.raises(RuntimeError, match="cuda"):
        gt.train_torch_anomaly_gpu(
            Path("nonexistent.parquet"),
            output_mlflow_uri="sqlite:///unused.db",
            epochs=2,
            device="cuda",
        )


@pytest.mark.slow
def test_cpu_fallback_trains(tmp_path):
    """CPU fallback trains end-to-end and returns a metrics dict."""
    from aether_pdm.ops.gpu_train import train_torch_anomaly_gpu

    features_path = _features_frame(tmp_path)
    uri = "sqlite:///" + str(tmp_path / "mlruns.db").replace("\\", "/")
    result = train_torch_anomaly_gpu(features_path, uri, epochs=4, device="cpu")

    assert result["device"] == "cpu"
    assert result["epochs"] == 4
    assert result["epochs_run"] >= 1
    # Separable synthetic data: calibrated threshold must catch faults while
    # keeping healthy val quiet (find_threshold targets recall 0.86).
    assert result["detection_rate"] >= 0.86 - 1e-9
    assert result["false_alarm_rate"] < 0.20
    assert isinstance(result["threshold"], float)
    assert result["mlflow_run_id"]
    assert result["mlflow_uri"] == uri
    assert result["n_train"] == 60
    assert result["n_val"] == 40
