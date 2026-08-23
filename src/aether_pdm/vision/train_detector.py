"""YOLOv8 fine-tune wrapper for surface-defect detection.

ultralytics is imported lazily inside functions. Training runs on CPU by
default (this repo has no GPU requirement); pass device explicitly to use one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_detector(
    data_yaml: Path,
    model_size: str = "n",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "cpu",
    project_dir: Path | None = None,
    run_name: str = "neu-det",
    mlflow_uri: str | None = None,
) -> dict[str, Any]:
    """
    Fine-tune a YOLOv8 model and return summary metrics.

    Returns dict with keys: run_dir, weights_path, mAP50, mAP50-95, epochs, device.
    """
    from ultralytics import YOLO

    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_yaml}")

    weights = f"yolov8{model_size}.pt"
    model = YOLO(weights)

    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "verbose": True,
    }
    if project_dir is not None:
        kwargs["project"] = str(Path(project_dir))
        kwargs["name"] = run_name
        kwargs["exist_ok"] = True

    results = model.train(**kwargs)

    # ultralytics DetMetrics exposes results_dict with keys like
    # 'metrics/mAP50(B)' — extract from there, falling back to attributes.
    rd = getattr(results, "results_dict", {}) or {}
    map50 = float(rd.get("metrics/mAP50(B)", 0.0))
    map_5095 = float(rd.get("metrics/mAP50-95(B)", 0.0))
    if map50 == 0.0:
        box = getattr(getattr(results, "box", None), "map50", None)
        map50 = float(box) if box else 0.0
    if map_5095 == 0.0:
        box = getattr(getattr(results, "box", None), "map", None)
        map_5095 = float(box) if box else 0.0

    save_dir = Path(getattr(results, "save_dir", Path("runs/detect") / run_name))
    weights_path = save_dir / "weights" / "best.pt"

    summary: dict[str, Any] = {
        "run_dir": str(save_dir),
        "weights_path": str(weights_path),
        "mAP50": map50,
        "mAP50-95": map_5095,
        "epochs": epochs,
        "device": device,
        "model_size": model_size,
    }

    if mlflow_uri:
        _log_to_mlflow(summary, mlflow_uri)

    return summary


def _log_to_mlflow(summary: dict[str, Any], mlflow_uri: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(mlflow_uri)
    with mlflow.start_run(run_name="vision_detector_train"):
        mlflow.log_params({
            "task": "surface_defect_detection",
            "model": f"yolov8{summary['model_size']}",
            "epochs": summary["epochs"],
            "device": summary["device"],
        })
        mlflow.log_metrics({
            "mAP50": summary["mAP50"],
            "mAP50-95": summary["mAP50-95"],
        })


def load_trained_detector(weights_path: str | Path) -> Any:
    """Load a fine-tuned detector for inference/evaluation."""
    from ultralytics import YOLO

    return YOLO(str(weights_path))
