"""CLI for exporting an AetherPdM anomaly model to an ONNX edge bundle.

Writes ``model.onnx`` + ``edge_scorer.py`` via
:func:`aether_pdm.ops.export_edge.export_edge_bundle`, then validates the
exported graph with onnxruntime against the in-memory model (exit code 1 when
the validation gate fails).

Usage:
    python scripts/export_onnx.py \\
        --model artifacts/torch_anomaly_detector.pt --model-type torch \\
        --input-dim 36 --output-dir edge
    python scripts/export_onnx.py \\
        --model artifacts/isolation_forest.joblib --model-type sklearn \\
        --input-dim 36 --output-dir edge
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from aether_pdm.models.torch_anomaly import TorchAnomalyDetector
from aether_pdm.ops.export_edge import export_edge_bundle, validate_onnx


def _load_model(path: Path, model_type: str):
    """Load a torch detector (.pt) or sklearn estimator (.joblib/.pkl)."""
    if model_type == "torch":
        return TorchAnomalyDetector.load(path)
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with path.open("rb") as fh:
            return pickle.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an AetherPdM anomaly model to an ONNX edge bundle"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Model artifact: TorchAnomalyDetector .pt or sklearn .joblib/.pkl",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["torch", "sklearn"],
        default="torch",
        help="Model flavor (default: torch)",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        required=True,
        help="Number of input features per sample",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("edge"),
        help="Directory for model.onnx + edge_scorer.py (default: edge)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random validation sample (default: 42)",
    )
    args = parser.parse_args()

    model = _load_model(args.model, args.model_type)
    bundle = export_edge_bundle(model, args.output_dir, args.input_dim, args.model_type)

    rng = np.random.default_rng(args.seed)
    sample = rng.normal(size=(8, args.input_dim))
    if args.model_type == "torch":
        import torch

        module = model.torch_module
        module.eval()
        with torch.no_grad():
            ref = module(torch.from_numpy(sample.astype(np.float32))).numpy()
    else:
        ref = model.decision_function(sample)

    result = validate_onnx(bundle["onnx_path"], sample, ref, atol=1e-4)
    print(f"onnx_path:    {bundle['onnx_path']}")
    print(f"scorer_path:  {bundle['scorer_path']}")
    print(
        f"validation:   max_abs_err={result['max_abs_err']:.3e} "
        f"passed={result['passed']} runtime_ms={result['runtime_ms']:.3f}"
    )
    if not result["passed"]:
        raise SystemExit(
            f"ONNX validation failed (max_abs_err {result['max_abs_err']:.3e} "
            "> 1e-4); the edge bundle is NOT safe to ship"
        )


if __name__ == "__main__":
    main()
