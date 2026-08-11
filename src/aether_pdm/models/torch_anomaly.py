"""PyTorch autoencoder anomaly detector for bearing vibration.

Trains a linear (MLP) autoencoder on healthy-only windows and scores every
window by reconstruction MSE: higher MSE = more anomalous.

PyTorch is imported **lazily** (inside methods only, never at module level)
so that ``import aether_pdm`` never loads torch. The package itself and fast
CI stay cheap; only this module's methods pay the torch import cost.

Design guarantees:

- **Deterministic**: fixed ``seed``, CPU execution, single-threaded torch
  kernels during ``fit`` so the same seed reproduces bit-identical scores.
- **Dropout resilience**: non-finite features (``NaN``/``Inf`` from sensor
  dropouts) are replaced with healthy-train per-column medians, and every
  feature is standardized with healthy-train mean/std (std floored at 1e-8
  so constant features cannot divide by zero).
- **Array contract**: raw feature matrix ``(N, input_dim)`` in, per-sample
  score vector ``(N,)`` out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_LOSS_EPS = 1e-12
_FILL_FALLBACK = 0.0
_STD_FLOOR = 1e-8
_PATIENCE = 10
_N_GRID = 256


def _to_float64(x: np.ndarray) -> NDArray[np.float64]:
    """Cast any numeric array to float64, no copy when already float64."""
    return np.asarray(x, dtype=np.float64)


class TorchAnomalyDetector:
    """MLP autoencoder anomaly detector (PyTorch, CPU, deterministic).

    Architecture (default ``hidden_dims=(64, 32, 16)``, ``latent_dim=8``):
    ``input_dim -> 64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64 -> input_dim``,
    ReLU activations (no activation on the final output layer), MSE loss,
    Adam optimizer.

    ``fit`` trains on healthy-only windows. When ``X_val`` (healthy-only) is
    provided, early stopping tracks its reconstruction loss with patience 10
    and restores the best state dict. ``find_threshold`` calibrates a
    per-window MSE threshold on labeled validation data to hit a target
    detection rate (recall of the fault class).

    Attributes:
        input_dim: Number of input features per window.
        hidden_dims: Encoder hidden widths (decoder mirrors them reversed).
        latent_dim: Bottleneck width.
        lr: Adam learning rate.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        seed: Random seed for reproducible CPU training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32, 16),
        latent_dim: int = 8,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if not hidden_dims or any(d < 1 for d in hidden_dims):
            raise ValueError(f"hidden_dims must be a non-empty tuple of positive "
                             f"ints, got {hidden_dims}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        # Fitted state (None until fit()/load()).
        self._module: Any = None
        self._mean: NDArray[np.float64] | None = None
        self._std: NDArray[np.float64] | None = None
        self._fill_values: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        x_healthy: np.ndarray,
        x_val: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Train the autoencoder on healthy-only windows.

        Args:
            x_healthy: Healthy training windows, shape ``(N, input_dim)``.
            x_val: Optional healthy-only windows, shape ``(M, input_dim)``.
                When provided, early stopping monitors its reconstruction
                loss and the best state dict is restored after training.

        Returns:
            History dict with keys ``train_loss`` (list of per-epoch losses),
            ``val_loss`` (list, empty when ``x_val`` is None), ``best_epoch``
            (1-based epoch of the best val loss, or last epoch), and
            ``epochs_run`` (number of epochs actually executed).
        """
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        # Single-threaded kernels make CPU training bit-deterministic for the
        # same seed (multi-threaded reductions can introduce float noise).
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)

        x_healthy = self._prepare_fit(x_healthy)
        self._build_module()
        model = self._module
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        x_t = torch.from_numpy(x_healthy.astype(np.float32))
        loader = DataLoader(
            TensorDataset(x_t), batch_size=self.batch_size, shuffle=True
        )

        val_t: torch.Tensor | None = None
        if x_val is not None:
            val_t = torch.from_numpy(self._transform(x_val).astype(np.float32))

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
                    best_state = _snapshot_state(model)
                    history["best_epoch"] = epoch + 1
                    wait = 0
                else:
                    wait += 1
                    if wait >= _PATIENCE:
                        break
            else:
                # No validation: keep the last epoch's weights.
                best_state = _snapshot_state(model)
                history["best_epoch"] = epoch + 1

        if best_state is not None:
            model.load_state_dict(best_state)
        return history

    # ------------------------------------------------------------- scoring

    def anomaly_scores(self, x: np.ndarray) -> NDArray[np.float64]:
        """Per-sample reconstruction MSE; higher = more anomalous.

        Args:
            x: Feature matrix, shape ``(N, input_dim)``.

        Returns:
            Score vector of shape ``(N,)`` (mean squared reconstruction
            error per window).
        """
        import torch

        self._require_fitted()
        x = self._transform(x)
        x_t = torch.from_numpy(x.astype(np.float32))
        self._module.eval()
        with torch.no_grad():
            mse = ((self._module(x_t) - x_t) ** 2).mean(dim=1)
        return mse.numpy().astype(np.float64)

    def predict(self, x: np.ndarray, threshold: float) -> NDArray[np.int64]:
        """Binary predictions: 1 = anomaly (score > threshold), else 0.

        Args:
            x: Feature matrix, shape ``(N, input_dim)``.
            threshold: Score threshold (from :meth:`find_threshold`).

        Returns:
            Integer label vector of shape ``(N,)`` with values in {0, 1}.
        """
        return (self.anomaly_scores(x) > threshold).astype(np.int64)

    # ---------------------------------------------------------- calibration

    def find_threshold(
        self,
        x_val: np.ndarray,
        y_val: np.ndarray,
        target_recall: float = 0.86,
    ) -> float:
        """Grid-search a score threshold that hits ``target_recall`` on val.

        Scans 256 candidate thresholds placed at evenly spaced **quantiles**
        of the observed scores (a linear grid is useless here: reconstruction
        MSE spans orders of magnitude — e.g. 0.1 to 1e6 on CWRU — so linear
        spacing would concentrate all resolution in the upper tail and leave
        a single candidate in the low range where the decision boundary
        lives). Among thresholds whose fault-class detection rate is
        ``>= target_recall`` the **largest** threshold is returned (most
        selective, i.e. lowest false-alarm rate while still catching the
        target share of faults). If no threshold reaches the target recall,
        the threshold that maximizes detection rate is returned (tie-broken
        toward the largest threshold).

        Args:
            x_val: Validation feature matrix, shape ``(N, input_dim)``.
            y_val: Binary labels, shape ``(N,)`` (0 = normal, 1 = fault).
            target_recall: Minimum fault detection rate to achieve.

        Returns:
            Score threshold ``t`` such that ``predict(x, t)`` flags samples
            with score ``> t``.

        Raises:
            ValueError: If validation data is empty, contains no fault
                samples, or scores are constant (calibration impossible).
        """
        y = np.asarray(y_val).ravel()
        if y.size == 0:
            raise ValueError("y_val is empty; cannot calibrate a threshold")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y_val must contain only 0 (normal) and 1 (fault) labels")
        faults = y == 1
        if not faults.any():
            raise ValueError(
                "y_val contains no fault (1) samples; a detection-rate target "
                "cannot be calibrated on healthy data alone"
            )

        scores = self.anomaly_scores(x_val)
        if scores.size == 0:
            raise ValueError("x_val is empty; cannot calibrate a threshold")
        if scores.min() == scores.max():
            raise ValueError(
                "anomaly scores are constant; no meaningful threshold exists "
                "(check for dead/flatline input features)"
            )

        # Quantile grid: dense where the data lives, robust to skewed ranges.
        grid = np.quantile(
            scores, np.linspace(0.0, 1.0, _N_GRID), method="linear"
        )
        # Vectorized grid: (N, 1) > (T,) -> (N, T) -> per-threshold recall.
        detections = scores[:, np.newaxis] > grid[np.newaxis, :]
        dr = detections[faults].mean(axis=0)

        mask = dr >= target_recall
        if mask.any():
            return float(grid[mask].max())
        best = np.flatnonzero(dr == dr.max())
        return float(grid[best[-1]])

    # ---------------------------------------------------------- persistence

    def save(self, path: Path | str) -> Path:
        """Persist the fitted detector (config + state dict + scaler stats).

        Args:
            path: Destination ``.pt`` file (parent directories created).

        Returns:
            The ``path`` that was written.

        Raises:
            RuntimeError: If the detector has not been fitted.
        """
        import torch

        self._require_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self._config(),
                "state_dict": self._module.state_dict(),
                "mean": self._mean,
                "std": self._std,
                "fill_values": self._fill_values,
            },
            output,
        )
        return output

    @classmethod
    def load(cls, path: Path | str) -> TorchAnomalyDetector:
        """Load a detector previously saved with :meth:`save`.

        Args:
            path: Source ``.pt`` file written by :meth:`save`.

        Returns:
            A fitted ``TorchAnomalyDetector`` reproducing the same scores.
        """
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        config = payload["config"]
        detector = cls(
            input_dim=config["input_dim"],
            hidden_dims=tuple(config["hidden_dims"]),
            latent_dim=config["latent_dim"],
            lr=config["lr"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            seed=config["seed"],
        )
        detector._build_module()
        detector._module.load_state_dict(payload["state_dict"])
        detector._mean = np.asarray(payload["mean"], dtype=np.float64)
        detector._std = np.asarray(payload["std"], dtype=np.float64)
        detector._fill_values = np.asarray(payload["fill_values"], dtype=np.float64)
        return detector

    # ------------------------------------------------------------- internal

    @property
    def torch_module(self) -> Any:
        """The underlying ``torch.nn.Module`` (autoencoder), or None.

        Exposed for MLflow pytorch-flavor logging, which requires a raw
        ``torch.nn.Module`` rather than a wrapper object.
        """
        return self._module

    def _config(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "latent_dim": self.latent_dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    def _build_module(self) -> None:
        from torch import nn

        enc_dims = [self.input_dim, *self.hidden_dims, self.latent_dim]
        dec_dims = [self.latent_dim, *reversed(self.hidden_dims), self.input_dim]

        layers: list[nn.Module] = []
        for a, b in zip(enc_dims[:-1], enc_dims[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(nn.ReLU())
        for a, b in zip(dec_dims[:-1], dec_dims[1:]):
            layers.append(nn.Linear(a, b))
            if b != self.input_dim:
                layers.append(nn.ReLU())
        self._module = nn.Sequential(*layers)

    def _prepare_fit(self, x_healthy: np.ndarray) -> NDArray[np.float64]:
        """Sanitize healthy train data and compute scaler statistics."""
        x = _to_float64(x_healthy)
        self._validate_dim(x)
        if x.shape[0] == 0:
            raise ValueError("x_healthy is empty; cannot train an autoencoder")

        self._fill_values = np.nanmedian(x, axis=0)
        self._fill_values = np.where(
            np.isfinite(self._fill_values), self._fill_values, _FILL_FALLBACK
        )
        x = self._sanitize(x)
        self._mean = x.mean(axis=0)
        self._std = x.std(axis=0)
        return self._scale(x)

    def _transform(self, x: np.ndarray) -> NDArray[np.float64]:
        """Sanitize + standardize with the statistics learned at fit time."""
        self._require_fitted()
        x = _to_float64(x)
        self._validate_dim(x)
        return self._scale(self._sanitize(x))

    def _validate_dim(self, x: NDArray[np.float64]) -> None:
        if x.ndim != 2:
            raise ValueError(
                f"Expected a 2D feature matrix (N, input_dim), got shape {x.shape}"
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension {x.shape[1]} does not match input_dim "
                f"{self.input_dim}"
            )

    def _require_fitted(self) -> None:
        if (
            self._module is None
            or self._mean is None
            or self._std is None
            or self._fill_values is None
        ):
            raise RuntimeError(
                "TorchAnomalyDetector is not fitted; call fit() or load() first"
            )

    def _sanitize(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Replace NaN/Inf (sensor dropout) with healthy-train medians."""
        if self._fill_values is None:
            raise RuntimeError(
                "TorchAnomalyDetector is not fitted; call fit() or load() first"
            )
        return np.where(np.isfinite(x), x, self._fill_values)

    def _scale(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Standardize with healthy-train mean/std (std floored at 1e-8)."""
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "TorchAnomalyDetector is not fitted; call fit() or load() first"
            )
        safe_std = np.where(self._std < _STD_FLOOR, 1.0, self._std)
        return (x - self._mean) / safe_std


def _snapshot_state(model: Any) -> dict[str, Any]:
    """Deep-copy the current state dict (decoupled from the live model)."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}
