"""Slow smoke tests for the detector fine-tune loop (tiny synthetic dataset)."""

from __future__ import annotations

from pathlib import Path

import pytest

from aether_pdm.vision.data import write_yolo_label


def _make_tiny_yolo_dataset(root: Path, n_images: int = 4, imgsz: int = 64) -> Path:
    """Create a minimal valid YOLO dataset with tiny solid-color JPEGs."""
    import numpy as np
    from PIL import Image

    images = root / "src_images"
    labels = root / "src_labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    palette = [(200, 200, 200), (80, 80, 80), (120, 90, 60)]
    for i in range(n_images):
        arr = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        arr[:, :] = palette[i % len(palette)]
        # bright defect square so the task is learnable in one epoch
        y0, x0 = imgsz // 4, imgsz // 4
        arr[y0 : y0 + 10, x0 : x0 + 10] = (255, 255, 255)
        Image.fromarray(arr).save(images / f"img_{i:02d}.jpg", quality=95)
        cls = i % 2  # two classes keeps the eval non-degenerate
        cx, cy = (x0 + 5) / imgsz, (y0 + 5) / imgsz
        labels_path = labels / f"img_{i:02d}.txt"
        lines = [f"{cls} {cx:.6f} {cy:.6f} {0.15:.6f} {0.15:.6f}"]
        labels_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from aether_pdm.vision.data import build_yolo_dataset

    return build_yolo_dataset(images, labels, root / "dataset", val_fraction=0.5, seed=3)


@pytest.mark.slow
def test_train_detector_smoke(tmp_path: Path):
    """One tiny CPU epoch end-to-end: train loop + metrics dict are wired."""
    from aether_pdm.vision.train_detector import train_detector

    data_yaml = _make_tiny_yolo_dataset(tmp_path, n_images=4, imgsz=64)

    summary = train_detector(
        data_yaml=data_yaml,
        model_size="n",
        epochs=1,
        imgsz=64,
        batch=2,
        device="cpu",
        project_dir=tmp_path / "runs",
        run_name="smoke",
    )
    assert "mAP50" in summary
    assert "mAP50-95" in summary
    assert Path(summary["weights_path"]).exists()


@pytest.mark.slow
def test_augmentation_pipeline_shapes():
    """Augment pipeline returns same-shape images for numpy input."""
    import numpy as np

    from aether_pdm.vision.augment import build_augmentation_pipeline, default_detector_augmentations

    pipe = build_augmentation_pipeline(default_detector_augmentations())
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    out = pipe(image=img)["image"]
    assert out.shape == img.shape
