"""Single-GPU training entrypoint for the PyTorch autoencoder anomaly detector.

Reuses :class:`aether_pdm.models.torch_anomaly.TorchAnomalyDetector` with a
device-aware fit (a thin subclass moves the module to ``cuda``/``cpu`` for
the training loop and back to CPU afterwards, so the parent's numpy-based
scoring, threshold calibration and MLflow pytorch logging keep working on any
host).

Behavior:

- ``device="auto"`` -> ``"cuda"`` when ``torch.cuda.is_available()`` else
  ``"cpu"`` (CPU fallback keeps CI/edge boxes fully functional).
- An explicit ``device="cuda"`` on a box without CUDA raises a clear
  ``RuntimeError`` instead of silently falling back.
- Training data contract matches the sklearn pipeline: a feature Parquet with
  ``split`` (train/val) and ``fault_type`` (``"normal"`` = healthy) columns;
  only healthy rows of ``split="train"`` train the autoencoder, healthy val
  rows drive early stopping, and the full val split (normal=0, fault=1)
  calibrates the threshold at target recall 0.86 and reports DR/FAR.
- MLflow: logs device, epochs, epochs_run, hyperparameters and the val
  DR/FAR metrics; registers the raw module as ``aether-anomaly-torch``
  (pytorch flavor, pickle serialization) and stores the full wrapper
  (config + scaler stats) as a run artifact for exact reproducibility.

Usage (via scripts/train_gpu.py):
    python scripts/train_gpu.py \
        --features data/interim/features/features_v2.parquet \
        --mlflow-uri sqlite:///mlflow.db --epochs 50 --device auto
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from aether_pdm.eval.metrics import detection_rate, false_alarm_rate
from aether_pdm.models.anomaly import META_COLS as ANOMALY_META_COLS
from aether_pdm.models.torch_anomaly import (
    _LOSS_EPS,
    _PATIENCE,
    TorchAnomalyDetector,
)

DEFAULT_MLFLOW_URI = "sqlite:///mlflow.db"
TORCH_MODEL_NAME = "aether-anomaly-torch"
TARGET_RECALL = 0.86
SEED = 42

_CUDA_AVAILABLE: bool | None = None


def _cuda_available() -> bool:
    """Lazy, cached ``torch.cuda.is_available()`` probe (import torch once)."""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        import torch

        _CUDA_AVAILABLE = bool(torch.cuda.is_available())
    return _CUDA_AVAILABLE


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` or ``"cpu"``; validate explicit values.

    Args:
        device: ``"auto"`` (default), ``"cuda"`` or ``"cpu"``.

    Returns:
        The concrete device string.

    Raises:
        ValueError: For any other value.
    """
    if device == "auto":
        return "cuda" if _cuda_available() else "cpu"
    if device in ("cuda", "cpu"):
        return device
    raise ValueError(f"device must be 'auto', 'cuda' or 'cpu', got {device!r}")


class _DeviceAwareTorchAnomalyDetector(TorchAnomalyDetector):
    """``TorchAnomalyDetector`` whose fit() runs the module on a device.

    Mirrors the parent's deterministic CPU loop, but keeps data tensors and
    the module on ``self.device`` for the whole training run. After training
    the module is moved back to CPU and the best state dict (early-stopped on
    healthy val loss, patience 10) is restored in CPU tensors, so the
    parent's numpy-scoring methods (``anomaly_scores``, ``find_threshold``,
    ``predict``) and ``save()`` work unchanged.
    """

    def __init__(self, device: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.device = device

    def fit(
        self,
        x_healthy: np.ndarray,
        x_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        if self.device == "cpu":
            torch.set_num_threads(1)
        torch.manual_seed(self.seed)

        x_healthy = self._prepare_fit(x_healthy)
        self._build_module()
        model = self._module.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        x_t = torch.from_numpy(x_healthy.astype(np.float32)).to(self.device)
        loader = DataLoader(
            TensorDataset(x_t), batch_size=self.batch_size, shuffle=True
        )

        val_t: torch.Tensor | None = None
        if x_val is not None:
            val_t = torch.from_numpy(
                self._transform(x_val).astype(np.float32)
            ).to(self.device)

        history: dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "epochs_run": 0,
        }
        best_val = float("inf")
        best_state: dict[str, Any] | None = None
        wait = 0

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            n_seen = 0
            for (xb,) in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), xb)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * xb.size(0)
                n_seen += xb.size(0)
            history["train_loss"].append(total_loss / max(n_seen, 1))
            history["epochs_run"] = epoch + 1

            if val_t is not None:
                model.eval()
                with torch.no_grad():
                    val_loss = float(criterion(model(val_t), val_t).item())
                if not np.isfinite(val_loss):
                    val_loss = float("inf")
                history["val_loss"].append(val_loss)
                if val_loss < best_val - _LOSS_EPS:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    history["best_epoch"] = epoch + 1
                    wait = 0
                else:
                    wait += 1
                    if wait >= _PATIENCE:
                        break
            else:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                history["best_epoch"] = epoch + 1

        # Pin best weights on CPU so parent numpy scoring works everywhere.
        self._module = model.to("cpu")
        if best_state is not None:
            self._module.load_state_dict(
                {k: v.detach().cpu() for k, v in best_state.items()}
            )
        return history


def train_torch_anomaly_gpu(
    features_path: Path,
    output_mlflow_uri: str | None = None,
    epochs: int = 50,
    device: str = "auto",
) -> dict[str, Any]:
    """Train the torch autoencoder anomaly detector on a device (cuda/cpu).

    Args:
        features_path: Feature Parquet with ``split`` (train/val) and
            ``fault_type`` columns (``"normal"`` = healthy).
        output_mlflow_uri: MLflow tracking URI (default
            ``sqlite:///mlflow.db``).
        epochs: Maximum training epochs.
        device: ``"auto"`` (cuda when available, else cpu), ``"cuda"`` or
            ``"cpu"``.

    Returns:
        Dict with ``device``, ``epochs``, ``epochs_run``, ``detection_rate``,
        ``false_alarm_rate``, ``threshold``, ``mlflow_run_id``, ``mlflow_uri``,
        ``n_train``, ``n_val``.

    Raises:
        ValueError: On unknown device, missing split/fault_type columns, no
            healthy train rows, or a degenerate val split (no faults / no
            normal rows).
        RuntimeError: On explicit ``device="cuda"`` when CUDA is unavailable.
    """
    device = resolve_device(device)
    if device == "cuda" and not _cuda_available():
        raise RuntimeError(
            "device='cuda' requested but torch.cuda.is_available() is False; "
            "pass device='auto' for CPU fallback or install a CUDA build of torch"
        )

    uri = output_mlflow_uri or DEFAULT_MLFLOW_URI
    df = pd.read_parquet(features_path)
    if not {"split", "fault_type"}.issubset(df.columns):
        raise ValueError(
            f"'{features_path}' must contain 'split' and 'fault_type' columns"
        )
    feature_cols = [c for c in df.columns if c not in ANOMALY_META_COLS]
    if len(feature_cols) < 2:
        raise ValueError(f"Expected >= 2 feature columns, found {len(feature_cols)}")

    train_df = df[df["split"] == "train"]
    healthy_train = train_df[train_df["fault_type"] == "normal"]
    if healthy_train.empty:
        raise ValueError("No healthy (normal) rows in split='train'")

    val_df = df[df["split"] == "val"]
    if val_df.empty:
        raise ValueError("No rows in split='val'")
    y_val = np.where(val_df["fault_type"].to_numpy() == "normal", 0, 1).astype(np.int64)
    if not y_val.any():
        raise ValueError("Val split has no fault samples; cannot measure detection rate")
    if not (y_val == 0).any():
        raise ValueError("Val split has no normal samples; cannot measure false alarm rate")

    x_train = healthy_train[feature_cols].to_numpy(dtype=np.float64)
    healthy_val = val_df[val_df["fault_type"] == "normal"]
    x_val_healthy = (
        healthy_val[feature_cols].to_numpy(dtype=np.float64)
        if not healthy_val.empty
        else None
    )
    x_val = val_df[feature_cols].to_numpy(dtype=np.float64)

    detector = _DeviceAwareTorchAnomalyDetector(
        device=device,
        input_dim=len(feature_cols),
        epochs=epochs,
        seed=SEED,
    )
    history = detector.fit(x_train, x_val_healthy)
    threshold = detector.find_threshold(x_val, y_val, target_recall=TARGET_RECALL)
    preds = detector.predict(x_val, threshold)
    dr = float(detection_rate(y_val, preds))
    far = float(false_alarm_rate(y_val, preds))

    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(run_name="torch_anomaly_gpu_train") as run:
        mlflow.log_params({
            "model_type": "TorchAnomalyDetector",
            "device": device,
            "epochs": epochs,
            "epochs_run": int(history["epochs_run"]),
            "lr": detector.lr,
            "hidden_dims": str(list(detector.hidden_dims)),
            "latent_dim": detector.latent_dim,
            "seed": detector.seed,
            "n_train_samples": int(len(x_train)),
            "input_dim": detector.input_dim,
            "target_recall": TARGET_RECALL,
        })
        mlflow.log_metrics({
            "detection_rate": dr,
            "false_alarm_rate": far,
            "threshold": float(threshold),
        })
        # Lazy: importing mlflow.pytorch binds torch, and must not rebind the
        # module-level `mlflow` name inside this function (scope shadowing).
        import importlib

        import torch

        mlflow_pytorch = importlib.import_module("mlflow.pytorch")
        mlflow_pytorch.log_model(
            detector.torch_module,
            "model",
            registered_model_name=TORCH_MODEL_NAME,
            serialization_format="pickle",
            pip_requirements=[f"torch=={torch.__version__}"],
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            wrapper_path = Path(tmp_dir) / "torch_anomaly_detector.pt"
            detector.save(wrapper_path)
            mlflow.log_artifact(str(wrapper_path))
        run_id = run.info.run_id

    print(
        f"Torch anomaly model trained on '{device}'. "
        f"MLflow run: {run_id} (DR={dr:.4f}, FAR={far:.4f})"
    )
    return {
        "device": device,
        "epochs": epochs,
        "epochs_run": int(history["epochs_run"]),
        "detection_rate": dr,
        "false_alarm_rate": far,
        "threshold": float(threshold),
        "mlflow_run_id": run_id,
        "mlflow_uri": uri,
        "n_train": int(len(x_train)),
        "n_val": int(len(y_val)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the PyTorch autoencoder anomaly detector on a GPU "
        "(or CPU fallback) and log DR/FAR to MLflow"
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Feature Parquet with split/fault_type columns",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device: auto (cuda if available), cuda or cpu",
    )
    args = parser.parse_args()

    result = train_torch_anomaly_gpu(
        args.features,
        output_mlflow_uri=args.mlflow_uri,
        epochs=args.epochs,
        device=args.device,
    )
    print(
        f"device={result['device']} epochs_run={result['epochs_run']} "
        f"DR={result['detection_rate']:.4f} FAR={result['false_alarm_rate']:.4f} "
        f"threshold={result['threshold']:.4f} run={result['mlflow_run_id']}"
    )


if __name__ == "__main__":
    main()
