"""Edge ONNX export + validation for AetherPdM anomaly models.

Exports either model flavor to ONNX for edge/embedded scoring:

- **torch** — the ``TorchAnomalyDetector`` autoencoder module via
  ``torch.onnx.export`` (opset 17, dynamic batch, float32). The exported
  graph is the raw autoencoder (reconstruction in, reconstruction out);
  input preprocessing (sensor-dropout fill + healthy-train standardization)
  is embedded in the generated ``edge_scorer.py`` so edge scores reproduce
  the wrapper's ``anomaly_scores`` contract exactly.
- **sklearn** — ``IsolationForest`` via ``skl2onnx``, wrapped in a one-step
  pipeline whose output is ``decision_function`` (higher = more normal,
  negative = anomaly). **EXPERIMENTAL**: the skl2onnx IsolationForest
  converter is not as battle-tested as the torch exporter; if it fails on a
  platform (or is dropped upstream) the error propagates with context and
  the torch route remains fully supported.

Heavy imports (``torch``, ``onnx``, ``skl2onnx``, ``onnxruntime``) stay
**lazy** inside functions: ``import aether_pdm.ops.export_edge`` never loads
them, so the fast CI suite stays cheap and torch-free (verified by
``tests/test_export_edge.py::test_import_lazy_torch``).

Array contracts:

- Raw feature matrix ``(N, input_dim)`` float64 in (float32 at the ONNX
  boundary), per-sample score vector ``(N,)`` out.
- The torch ONNX graph itself is ``input (N, input_dim) float32 ->
  output (N, input_dim) float32`` (reconstruction).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_ONNX_OPSET = 17
_ATOL_DEFAULT = 1e-4
_VALIDATION_RUNS = 3

# --------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------


def export_model_onnx(
    model: Any,
    output_path: Path | str,
    input_dim: int,
    model_type: str = "torch",
) -> Path:
    """Export an AetherPdM anomaly model to an ONNX graph.

    Args:
        model: The fitted model. For ``model_type="torch"`` either a
            :class:`aether_pdm.models.torch_anomaly.TorchAnomalyDetector`
            (its ``torch_module`` is exported) or a raw ``torch.nn.Module``.
            For ``model_type="sklearn"`` a fitted ``sklearn.ensemble.
            IsolationForest`` (or any sklearn estimator whose
            ``decision_function`` skl2onnx can convert).
        output_path: Destination ``.onnx`` file (parent dirs created).
        input_dim: Number of input features per sample.
        model_type: ``"torch"`` (default) or ``"sklearn"``.

    Returns:
        The ``output_path`` that was written.

    Raises:
        ValueError: If ``model_type`` is unknown or ``input_dim`` < 1.
        TypeError: If the model is not convertible for the requested flavor.
        RuntimeError: If the torch detector is not fitted.
    """
    if model_type not in ("torch", "sklearn"):
        raise ValueError(
            f"model_type must be 'torch' or 'sklearn', got {model_type!r}"
        )
    if input_dim < 1:
        raise ValueError(f"input_dim must be >= 1, got {input_dim}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if model_type == "torch":
        _export_torch(model, output_path, input_dim)
    else:
        _export_sklearn(model, output_path, input_dim)
    return output_path


def _export_torch(model: Any, output_path: Path, input_dim: int) -> None:
    """Export the autoencoder module with torch.onnx (opset 17, dynamic batch)."""
    import torch

    module = getattr(model, "torch_module", None)
    if module is None and isinstance(model, torch.nn.Module):
        module = model
    if module is None:
        raise TypeError(
            "model_type='torch' requires a fitted TorchAnomalyDetector "
            "(torch_module set) or a raw torch.nn.Module, got "
            f"{type(model).__name__}"
        )

    module.eval()
    dummy = torch.zeros((1, input_dim), dtype=torch.float32)
    # dynamo=False: the torch.export-based exporter (default since 2.9) needs
    # the optional `onnxscript` package; the legacy TorchScript exporter does
    # not, and fully covers our static nn.Sequential graphs (opset 17). If a
    # future torch removes the legacy path, add onnxscript instead.
    torch.onnx.export(
        module,
        (dummy,),
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=_ONNX_OPSET,
        dynamo=False,
    )


def _export_sklearn(model: Any, output_path: Path, input_dim: int) -> None:
    """Export an IsolationForest (decision_function output) via skl2onnx.

    The estimator is wrapped in a one-step ``Pipeline``; skl2onnx converts
    the pipeline and the IsolationForest converter emits the sklearn
    ``decision_function`` (negative score = anomaly).

    EXPERIMENTAL: skl2onnx's IsolationForest converter may be unavailable or
    fail on some platforms; errors propagate as-is (with context) and the
    torch route is unaffected.
    """
    from onnx import save
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        pipeline = model
    else:
        pipeline = Pipeline([("model", model)])

    try:
        onnx_model = convert_sklearn(
            pipeline,
            name="aether_anomaly_if",
            initial_types=[("input", FloatTensorType([None, input_dim]))],
            # Pin ai.onnx.ml to v3: skl2onnx 1.20 caps its TreeEnsemble opset
            # there while newer onnx/onnxruntime default to v4, which this
            # library rejects (RuntimeError from _update_domain_version).
            target_opset={"": _ONNX_OPSET, "ai.onnx.ml": 3},
        )
    except Exception as exc:  # honest propagation: document + surface upstream errors
        raise RuntimeError(
            "skl2onnx failed to convert the sklearn model (IsolationForest "
            "-> ONNX is experimental); use model_type='torch' for the "
            "supported route"
        ) from exc
    save(onnx_model, str(output_path))


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _score_output_name(session: Any) -> str:
    """Name of the anomaly-score output of an ONNX session.

    skl2onnx IsolationForest graphs expose two outputs (``label`` and
    ``scores``); the score one is selected by name. Torch exports expose a
    single ``output``, which is the fallback.
    """
    outputs = session.get_outputs()
    for out in outputs:
        if "score" in out.name.lower():
            return out.name
    return outputs[0].name


def validate_onnx(
    onnx_path: Path | str,
    sample_input: np.ndarray,
    reference_outputs: np.ndarray,
    atol: float = _ATOL_DEFAULT,
) -> dict[str, Any]:
    """Run an exported ONNX graph with onnxruntime and compare to references.

    Args:
        onnx_path: Exported ``.onnx`` file.
        sample_input: Input matrix, shape ``(N, input_dim)`` (or 1D, treated
            as a single sample). Cast to float32 before inference.
        reference_outputs: Reference outputs from the in-memory training
            model (same rows as ``sample_input``). Flattened before
            comparison so ``(N,)`` vs ``(N, 1)`` layouts are equivalent.
        atol: Absolute error tolerance; ``passed`` is True iff the max
            absolute error over all elements is ``<= atol``.

    Returns:
        Dict with ``max_abs_err`` (float, ``inf`` on shape mismatch),
        ``passed`` (bool) and ``runtime_ms`` (mean single-inference latency
        over ``_VALIDATION_RUNS``, after a warm-up run).
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = _score_output_name(session)

    x = np.asarray(sample_input, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    feeds = {input_name: x}
    session.run([output_name], feeds)  # warm-up (provider init / shape specialization)
    t0 = time.perf_counter()
    for _ in range(_VALIDATION_RUNS):
        out = session.run([output_name], feeds)[0]
    runtime_ms = (time.perf_counter() - t0) / _VALIDATION_RUNS * 1000.0

    ref = np.asarray(reference_outputs, dtype=np.float64)
    out_flat = np.asarray(out).reshape(-1)
    ref_flat = ref.reshape(-1)
    if out_flat.size != ref_flat.size:
        max_abs_err = float("inf")
    elif out_flat.size == 0:
        max_abs_err = 0.0
    else:
        max_abs_err = float(np.max(np.abs(out_flat.astype(np.float64) - ref_flat)))

    return {
        "max_abs_err": max_abs_err,
        "passed": bool(max_abs_err <= atol),
        "runtime_ms": float(runtime_ms),
    }


# --------------------------------------------------------------------------
# Edge bundle (model.onnx + standalone edge_scorer.py)
# --------------------------------------------------------------------------


def export_edge_bundle(
    model: Any,
    output_dir: Path | str,
    input_dim: int,
    model_type: str = "torch",
) -> dict[str, str]:
    """Write a self-contained edge bundle: ``model.onnx`` + ``edge_scorer.py``.

    The generated ``edge_scorer.py`` is a standalone module (dependencies:
    numpy + onnxruntime only) that loads ``model.onnx`` next to itself and
    scores feature vectors. For the torch flavor it embeds the fitted scaler
    statistics so edge scores reproduce ``TorchAnomalyDetector.anomaly_scores``
    exactly (including NaN/Inf dropout fill); for the sklearn flavor the
    ONNX output is directly the IsolationForest ``decision_function``.

    Args:
        model: Fitted model (see :func:`export_model_onnx`).
        output_dir: Directory where ``model.onnx`` and ``edge_scorer.py``
            are written (created if missing).
        input_dim: Number of input features per sample.
        model_type: ``"torch"`` or ``"sklearn"``.

    Returns:
        Dict with keys ``onnx_path``, ``scorer_path``, ``model_type``,
        ``input_dim``.

    Raises:
        RuntimeError: If a torch detector is not fitted (scaler statistics
            missing).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = export_model_onnx(model, output_dir / "model.onnx", input_dim, model_type)
    scorer_path = output_dir / "edge_scorer.py"
    scorer_path.write_text(
        _edge_scorer_source(model, model_type, input_dim), encoding="utf-8"
    )
    return {
        "onnx_path": str(onnx_path),
        "scorer_path": str(scorer_path),
        "model_type": model_type,
        "input_dim": str(int(input_dim)),
    }


def _scaler_stats(
    model: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Scaler statistics for the torch edge scorer (or identity for raw modules).

    Returns:
        ``(mean, std, fill)`` per-column arrays. For a raw ``torch.nn.Module``
        (no preprocessing contract) length-1 identity arrays are returned —
        they broadcast over any feature count.
    """
    mean = getattr(model, "_mean", None)
    std = getattr(model, "_std", None)
    fill = getattr(model, "_fill_values", None)
    if mean is None or std is None or fill is None:
        if getattr(model, "torch_module", None) is not None:
            raise RuntimeError(
                "TorchAnomalyDetector is not fitted; scaler statistics are "
                "missing (call fit() or load() first)"
            )
        # Raw nn.Module: the exported graph is the full contract.
        return np.zeros(1), np.ones(1), np.zeros(1)
    return (
        np.asarray(mean, dtype=np.float64),
        np.asarray(std, dtype=np.float64),
        np.asarray(fill, dtype=np.float64),
    )


def _edge_scorer_source(model: Any, model_type: str, input_dim: int) -> str:
    """Render the standalone edge scorer source for a model flavor."""
    if model_type == "torch":
        mean, std, fill = _scaler_stats(model)
        return (
            _TORCH_SCORER_TEMPLATE.replace("__INPUT_DIM__", repr(int(input_dim)))
            .replace("__MEAN__", repr(mean.tolist()))
            .replace("__STD__", repr(std.tolist()))
            .replace("__FILL__", repr(fill.tolist()))
        )
    return _SKLEARN_SCORER_TEMPLATE.replace("__INPUT_DIM__", repr(int(input_dim)))


_TORCH_SCORER_TEMPLATE = '''\
"""Auto-generated lightweight edge scorer for an AetherPdM anomaly model.

Generated by ``aether_pdm.ops.export_edge.export_edge_bundle``; do not edit.
Model: TorchAnomalyDetector (autoencoder) exported at ONNX opset 17.
The ONNX graph is the raw autoencoder; this scorer reproduces the full
``anomaly_scores`` contract: sensor-dropout fill (NaN/Inf -> healthy-train
median), healthy-train standardization, then reconstruction MSE.

Dependencies: numpy, onnxruntime (CPU-only execution).
"""

from pathlib import Path

import numpy as np
import onnxruntime

_INPUT_DIM = __INPUT_DIM__
_STD_FLOOR = 1e-8
_MEAN = __MEAN__
_STD = __STD__
_FILL = __FILL__


class EdgeAnomalyScorer:
    """ONNX autoencoder scorer: feature vector in, anomaly score out."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent / "model.onnx")
        self._session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        self._output_name = next(
            (o.name for o in outputs if "score" in o.name.lower()),
            outputs[0].name,
        )

    def _preprocess(self, x):
        # Mirrors TorchAnomalyDetector._sanitize/_scale (float64 numpy).
        x = np.where(np.isfinite(x), x, np.asarray(_FILL, dtype=np.float64))
        safe_std = np.where(np.asarray(_STD) < _STD_FLOOR, 1.0, _STD)
        return (x - np.asarray(_MEAN, dtype=np.float64)) / safe_std

    def score(self, feature_vector):
        """Anomaly score (reconstruction MSE) for one feature vector."""
        v = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        if v.shape[1] != _INPUT_DIM:
            raise ValueError(
                f"expected {_INPUT_DIM} features, got {v.shape[1]}"
            )
        x = self._preprocess(v).astype(np.float32)
        recon = self._session.run([self._output_name], {self._input_name: x})[0]
        return float(np.mean(np.square(recon - x)))

    def scores(self, features):
        """Anomaly scores for a (N, input_dim) feature matrix."""
        f = np.asarray(features, dtype=np.float64)
        if f.ndim == 1:
            f = f.reshape(1, -1)
        if f.shape[1] != _INPUT_DIM:
            raise ValueError(
                f"expected {_INPUT_DIM} features, got {f.shape[1]}"
            )
        x = self._preprocess(f).astype(np.float32)
        recon = self._session.run([self._output_name], {self._input_name: x})[0]
        return np.mean(np.square(recon - x), axis=1)


if __name__ == "__main__":
    import sys

    scorer = EdgeAnomalyScorer()
    vec = [float(v) for v in sys.argv[1:]]
    print(scorer.score(vec))
'''


_SKLEARN_SCORER_TEMPLATE = '''\
"""Auto-generated lightweight edge scorer for an AetherPdM anomaly model.

Generated by ``aether_pdm.ops.export_edge.export_edge_bundle``; do not edit.
Model: IsolationForest exported via skl2onnx (EXPERIMENTAL route).
Output = sklearn ``decision_function``: higher = more normal, negative =
anomaly.

Dependencies: numpy, onnxruntime (CPU-only execution).
"""

from pathlib import Path

import numpy as np
import onnxruntime

_INPUT_DIM = __INPUT_DIM__


class EdgeAnomalyScorer:
    """ONNX IsolationForest scorer: feature vector in, anomaly score out."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = str(Path(__file__).resolve().parent / "model.onnx")
        self._session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        # skl2onnx emits 'label' + 'scores'; the scorer needs the scores.
        outputs = self._session.get_outputs()
        self._output_name = next(
            (o.name for o in outputs if "score" in o.name.lower()),
            outputs[0].name,
        )

    def score(self, feature_vector):
        """Anomaly score (decision_function) for one feature vector."""
        v = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        if v.shape[1] != _INPUT_DIM:
            raise ValueError(
                f"expected {_INPUT_DIM} features, got {v.shape[1]}"
            )
        out = self._session.run([self._output_name], {self._input_name: v})[0]
        return float(np.asarray(out).reshape(-1)[0])

    def scores(self, features):
        """Anomaly scores for a (N, input_dim) feature matrix."""
        f = np.asarray(features, dtype=np.float32)
        if f.ndim == 1:
            f = f.reshape(1, -1)
        if f.shape[1] != _INPUT_DIM:
            raise ValueError(
                f"expected {_INPUT_DIM} features, got {f.shape[1]}"
            )
        out = self._session.run([self._output_name], {self._input_name: f})[0]
        return np.asarray(out).reshape(-1)


if __name__ == "__main__":
    import sys

    scorer = EdgeAnomalyScorer()
    vec = [float(v) for v in sys.argv[1:]]
    print(scorer.score(vec))
'''
