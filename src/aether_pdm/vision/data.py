"""NEU-DET surface defect dataset: download and YOLO format conversion.

NEU-DET (hot-rolled steel strip surface defects) contains six defect classes
imaged on production strip. It is used here as the first visual-inspection
dataset per ADR-004.

Public mirrors of NEU-DET come in a few layouts. We prefer one that already
ships YOLO-format labels; otherwise annotations are converted from the
original per-image ground-truth files.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Known public mirrors hosting NEU-DET. Tried in order. The Marfbin repo
# ships the dataset in ultralytics layout (train/ + test/ with images/ and
# labels/ subfolders, plus data.yaml) — converted to our layout on extract.
MIRROR_URLS = [
    "https://github.com/Marfbin/NEU-DET-with-yolov8/archive/refs/heads/main.zip",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".bmp", ".png"}


def _download(url: str, dest: Path, timeout: int = 300) -> bool:
    import httpx

    print(f"  downloading {url}")
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            dest.write_bytes(response.content)
        size_kb = len(dest.read_bytes()) / 1024
        print(f"    -> {dest} ({size_kb:.0f} KB)")
        return True
    except Exception as exc:
        print(f"    FAILED: {exc}")
        return False


def download_neu_det(dest_dir: Path) -> Path:
    """Download NEU-DET, returning the directory that contains the dataset.

    Tries each known mirror in order. Raises RuntimeError when every mirror
    fails so the caller gets actionable instructions instead of silence.
    """
    dest_dir = Path(dest_dir)
    out_root = dest_dir / "neu_det"
    marker = out_root / "images"
    if marker.exists() and any(marker.iterdir()):
        print(f"NEU-DET already present at {out_root}")
        return out_root

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "neu_det_download.zip"

    errors: list[str] = []
    for url in MIRROR_URLS:
        if not _download(url, archive):
            errors.append(f"{url}: download failed")
            continue
        try:
            import zipfile

            with zipfile.ZipFile(archive) as zf:
                zf.extractall(out_root)
            archive.unlink()
            _normalize_extracted_layout(out_root)
            if not any(out_root.rglob("*.jpg")) and not any(out_root.rglob("*.png")):
                errors.append(f"{url}: extracted but no images found")
                continue
            return out_root
        except Exception as exc:  # noqa: BLE001 - report every mirror failure
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "Could not download NEU-DET from any known mirror. "
        "Download it manually and place images under "
        f"{out_root}/images with YOLO labels in {out_root}/labels. "
        "Tried:\n  " + "\n  ".join(errors)
    )


def _normalize_extracted_layout(out_root: Path) -> None:
    """
    Flatten extracted archives into our expected images/ + labels/ layout.

    Handles two known shapes:
      - a single top-level wrapper dir (flatten one level), then
      - ultralytics-style splits (train/images, train/labels, val/...,
        test/...) which get merged: split subfolders are renamed
        train_<split> so the file-level anti-leakage rule stays intact.
    """
    # flatten a single wrapper directory (e.g. NEU-DET-with-yolov8-main/)
    entries = [e for e in out_root.iterdir()]
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    if len(dirs) == 1 and not files and not any(out_root.glob("*.yaml")):
        inner = dirs[0]
        for item in inner.iterdir():
            shutil.move(str(item), str(out_root / item.name))
        inner.rmdir()

    # merge ultralytics-style splits: <split>/images + <split>/labels
    for split_dir in list(out_root.iterdir()):
        if not (split_dir.is_dir() and (split_dir / "images").exists()):
            continue
        prefix = split_dir.name  # train / test / val
        img_dst = out_root / f"images_{prefix}"
        lbl_dst = out_root / f"labels_{prefix}"
        if not img_dst.exists():
            shutil.move(str(split_dir / "images"), str(img_dst))
        if (split_dir / "labels").exists() and not lbl_dst.exists():
            shutil.move(str(split_dir / "labels"), str(lbl_dst))
        # leftover data.yaml etc. stay in the split dir; remove if now empty-ish
        remaining = [p for p in split_dir.iterdir()]
        if not remaining:
            split_dir.rmdir()

    # collapse images_<split>/labels_<split> into plain images/ + labels/
    merged_images = out_root / "images"
    merged_labels = out_root / "labels"
    any_img = list(out_root.glob("images_*/*.jpg")) or list(
        out_root.glob("images_*/*.png")
    )
    if any_img and not merged_images.exists():
        merged_images.mkdir(parents=True, exist_ok=True)
        merged_labels.mkdir(parents=True, exist_ok=True)
        for img_dir in sorted(out_root.glob("images_*")):
            tag = img_dir.name.replace("images_", "")
            for img in img_dir.iterdir():
                shutil.move(str(img), str(merged_images / f"{tag}_{img.name}"))
        for lbl_dir in sorted(out_root.glob("labels_*")):
            tag = lbl_dir.name.replace("labels_", "")
            for lbl in lbl_dir.iterdir():
                shutil.move(str(lbl), str(merged_labels / f"{tag}_{lbl.name}"))
        for d in sorted(out_root.glob("images_*")):
            d.rmdir()
        for d in sorted(out_root.glob("labels_*")):
            d.rmdir()


def yolo_line(
    class_index: int,
    box_xyxy: tuple[float, float, float, float],
    image_w: int,
    image_h: int,
) -> str:
    """Format one detection as a YOLO label line (class cx cy w h, normalized)."""
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0 / image_w
    cy = (y1 + y2) / 2.0 / image_h
    w = (x2 - x1) / image_w
    h = (y2 - y1) / image_h
    return f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def parse_yolo_label(line: str) -> tuple[int, float, float, float, float]:
    """Parse one YOLO label line into (class, cx, cy, w, h)."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"bad label line: {line!r}")
    return (
        int(parts[0]),
        float(parts[1]),
        float(parts[2]),
        float(parts[3]),
        float(parts[4]),
    )


def write_yolo_label(
    path: Path,
    boxes: list[tuple[int, tuple[float, float, float, float]]],
    img_w: int,
    img_h: int,
) -> None:
    """Write YOLO label lines for one image."""
    lines = [yolo_line(cls, box, img_w, img_h) for cls, box in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def split_image_ids(
    ids: list[str], val_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Deterministically split image ids into train/val (file-level, no leakage)."""
    ordered = sorted(ids)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ordered))
    n_val = max(1, int(round(len(ordered) * val_fraction)))
    val_ids = [ordered[i] for i in idx[:n_val]]
    train_ids = [ordered[i] for i in idx[n_val:]]
    return train_ids, val_ids


def build_yolo_dataset(
    images_src: Path,
    labels_src: Path,
    dest_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Path:
    """
    Organize images + YOLO labels into the ultralytics layout:

        dest_dir/
          images/train, images/val
          labels/train, labels/val
          dataset.yaml

    Returns the path to dataset.yaml.
    """
    images_src, labels_src, dest_dir = Path(images_src), Path(labels_src), Path(dest_dir)

    image_files = sorted(
        p for p in images_src.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not image_files:
        raise RuntimeError(f"No images found in {images_src}")

    ids = [p.stem for p in image_files]
    train_ids, val_ids = split_image_ids(ids, val_fraction=val_fraction, seed=seed)
    id_to_split = {i: "train" for i in train_ids}
    id_to_split.update({i: "val" for i in val_ids})

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (dest_dir / sub).mkdir(parents=True, exist_ok=True)

    yaml_lines = [
        f"path: {dest_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(NEU_CLASSES)}",
        f"names: {NEU_CLASSES}",
    ]

    for img_path in image_files:
        stem = img_path.stem
        split = id_to_split[stem]
        shutil.copy(img_path, dest_dir / "images" / split / img_path.name)
        src_label = labels_src / f"{stem}.txt"
        dst_label = dest_dir / "labels" / split / f"{stem}.txt"
        if src_label.exists():
            shutil.copy(src_label, dst_label)
        elif not dst_label.exists():
            dst_label.write_text("", encoding="utf-8")  # background image

    dataset_yaml = dest_dir / "dataset.yaml"
    dataset_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"Wrote YOLO dataset ({len(train_ids)} train / {len(val_ids)} val) -> {dataset_yaml}")
    return dataset_yaml
