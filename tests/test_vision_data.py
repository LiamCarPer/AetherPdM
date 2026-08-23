"""Fast tests for the NEU-DET dataset utilities (no cv2/ultralytics needed)."""

from pathlib import Path

import pytest

from aether_pdm.vision.data import (
    NEU_CLASSES,
    build_yolo_dataset,
    parse_yolo_label,
    split_image_ids,
    write_yolo_label,
    yolo_line,
)


def test_neu_classes_constant():
    assert NEU_CLASSES == [
        "crazing",
        "inclusion",
        "patches",
        "pitted_surface",
        "rolled-in_scale",
        "scratches",
    ]
    assert len(set(NEU_CLASSES)) == 6


def test_yolo_line_roundtrip():
    line = yolo_line(2, (10.0, 20.0, 30.0, 40.0), image_w=200, image_h=100)
    cls, cx, cy, w, h = parse_yolo_label(line)
    assert cls == 2
    # center x = (10+30)/2 / 200 = 0.1
    assert cx == pytest.approx(0.1, abs=1e-6)
    # center y = (20+40)/2 / 100 = 0.3
    assert cy == pytest.approx(0.3, abs=1e-6)
    assert w == pytest.approx(20 / 200, abs=1e-6)
    assert h == pytest.approx(20 / 100, abs=1e-6)


def test_parse_yolo_label_rejects_bad_lines():
    with pytest.raises(ValueError):
        parse_yolo_label("not a label")
    with pytest.raises(ValueError):
        parse_yolo_label("1 2 3")


def test_split_deterministic_and_disjoint():
    ids = [f"img{i:03d}" for i in range(50)]
    train_a, val_a = split_image_ids(ids, val_fraction=0.2, seed=42)
    train_b, val_b = split_image_ids(ids, val_fraction=0.2, seed=42)

    assert train_a == train_b
    assert val_a == val_b
    assert not (set(train_a) & set(val_a))
    assert set(train_a) | set(val_a) == set(ids)


def test_build_yolo_dataset_layout(tmp_path: Path):
    images = tmp_path / "src_images"
    labels = tmp_path / "src_labels"
    images.mkdir()
    labels.mkdir()

    for i in range(6):
        (images / f"defect_{i:03d}.jpg").write_bytes(b"fake-jpg")
        boxes = [(i % 6, (10, 10, 50, 50))]
        write_yolo_label(labels / f"defect_{i:03d}.txt", boxes, 200, 200)

    dest = tmp_path / "dataset"
    yaml_path = build_yolo_dataset(images, labels, dest, val_fraction=0.33, seed=7)

    assert yaml_path.exists()
    content = yaml_path.read_text(encoding="utf-8")
    assert "nc: 6" in content
    assert "scratches" in content

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        assert (dest / sub).exists(), f"missing {sub}"

    n_train = len(list((dest / "images/train").iterdir()))
    n_val = len(list((dest / "images/val").iterdir()))
    assert n_train + n_val == 6
    assert n_val >= 1
