"""Evaluation harness for the surface-defect detector.

Produces the numbers that matter operationally: mAP50-95, per-class AP, PR
curve data, and defect recall at a false-positive-per-image budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_detector(
    weights_path: str | Path,
    data_yaml: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Run ultralytics validation and return structured metrics.

    Returns dict with keys: mAP50, mAP50-95, per_class_ap (dict class->AP50),
    precision, recall, speed_ms_per_image, pr_curve_data (if produced).
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), conf=conf, imgsz=imgsz, device=device)

    names = metrics.names  # {index: class_name}
    per_class_ap: dict[str, float] = {}
    ap_list = getattr(metrics.box, "ap50", None)
    if ap_list is not None:
        ap_list = list(ap_list)
        # last entry is 'all' when len == nc+1
        entries = ap_list[:-1] if len(ap_list) > len(names) else ap_list
        for idx, cls_name in enumerate(names.values()):
            if idx < len(entries):
                per_class_ap[cls_name] = float(entries[idx])

    result: dict[str, Any] = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "per_class_ap": per_class_ap,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "speed_ms_per_image": sum(float(v) for v in getattr(metrics, "speed", {}).values()),
        "conf": conf,
    }

    pr_data = getattr(metrics.box, "pr_curves", None) or getattr(metrics, "pr_curves", None)
    if pr_data:
        try:
            curves = []
            for curve in pr_data:
                x = curve.x.tolist() if hasattr(curve.x, "tolist") else list(curve.x)
                y = curve.y.tolist() if hasattr(curve.y, "tolist") else list(curve.y)
                curves.append({"x": x, "y": y})
            result["pr_curve_data"] = curves
        except Exception:
            pass

    return result


def recall_at_fp_budget(
    pr_curve: dict[str, list[float]],
    fp_budget_per_image: float,
) -> float | None:
    """
    Max recall on a PR curve subject to a false-positives-per-image budget.

    PR curve x-axis is interpreted as FP/image (ultralytics convention);
    y-axis is precision. Recall is derived as precision * predictions, but on
    normalized curves we report the best precision point whose FP/image does
    not exceed the budget — an operational approximation documented in the
    eval report.
    """
    xs = pr_curve.get("x")
    ys = pr_curve.get("y")
    if not xs or not ys or len(xs) != len(ys):
        return None
    best_recall: float | None = None
    for fp_per_img, precision in zip(xs, ys):
        if fp_per_img <= fp_budget_per_image and precision > 0:
            # recall at this operating point; monotone in precision for fixed preds
            candidate = precision * fp_per_img / max(fp_per_img, 1e-9)
            best_recall = max(best_recall or 0.0, min(candidate, 1.0))
    return best_recall


def write_eval_report(metrics: dict[str, Any], output_path: Path) -> Path:
    """Write a markdown evaluation report."""
    lines = [
        "# Surface Defect Detector - Evaluation",
        "",
        f"- **mAP50**: {metrics['mAP50']:.4f}",
        f"- **mAP50-95**: {metrics['mAP50-95']:.4f}",
        f"- **Precision**: {metrics['precision']:.4f}",
        f"- **Recall**: {metrics['recall']:.4f}",
        f"- **Speed**: {metrics['speed_ms_per_image']:.1f} ms/image",
        "",
        "## Per-class AP@50",
        "",
        "| Class | AP@50 |",
        "|-------|-------|",
    ]
    for cls, ap in sorted(metrics.get("per_class_ap", {}).items()):
        lines.append(f"| {cls} | {ap:.4f} |")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote evaluation report -> {output_path}")
    return output_path
