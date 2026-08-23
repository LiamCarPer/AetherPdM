"""OpenCV utilities for industrial camera setups.

Covers the "industrial cameras / lighting / calibration" extras from the
2026 production-line CV baseline: chessboard calibration, undistortion,
and a frame-trigger simulation interface for testing consumers without
hardware. opencv (cv2) is imported lazily throughout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def calibrate_camera(
    calibration_images_dir: Path,
    checkerboard_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
) -> dict[str, Any]:
    """
    Calibrate a camera from checkerboard images.

    Args:
        calibration_images_dir: directory of checkerboard photos.
        checkerboard_size: (columns, rows) of INNER corners.
        square_size_mm: physical size of one checker square.

    Returns dict with:
        camera_matrix, dist_coeffs, reprojection_error, num_images_used,
        image_size

    Raises RuntimeError when fewer than 3 usable images are found.
    """
    import cv2

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    cols, rows = checkerboard_size
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    # mypy: image_size is always set on the first successfully read image;
    # the assert below narrows it for the calibrateCamera call.
    if image_size is None:
        raise RuntimeError("no calibration images could be read")
    skipped = 0

    image_paths = sorted(
        p
        for p in Path(calibration_images_dir).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            skipped += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        if not found:
            skipped += 1
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        obj_points.append(objp)
        img_points.append(corners_refined)

    if len(obj_points) < 3:
        raise RuntimeError(
            f"Only {len(obj_points)} usable checkerboard images "
            f"({skipped} skipped). Need at least 3 for calibration."
        )
    # At least one image was read to reach this point, so image_size is set.
    assert image_size is not None

    rms, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    return {
        "camera_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "reprojection_error": float(rms),
        "num_images_used": len(obj_points),
        "num_images_skipped": skipped,
        "image_size": list(image_size),
        "checkerboard": list(checkerboard_size),
    }


def undistort_image(
    image_path: Path,
    camera_matrix: list[list[float]],
    dist_coeffs: list[list[float]],
    output_path: Path | None = None,
) -> np.ndarray:
    """Undistort one image using calibration results."""
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read {image_path}")
    mtx = np.array(camera_matrix, dtype=np.float64)
    dist = np.array(dist_coeffs, dtype=np.float64)
    h, w = img.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    result = cv2.undistort(img, mtx, dist, None, new_mtx)
    x, y, cw, ch = roi
    result = result[y : y + ch, x : x + cw]
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)
    return result


def simulate_triggered_frames(
    n_frames: int = 10,
    frame_shape: tuple[int, int] = (480, 640),
    defect_probability: float = 0.3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Simulate PLC-triggered frame acquisition for testing consumers.

    Generates frames with metadata as a hardware trigger would produce them:
    each entry carries a frame id, a monotonic timestamp, and a synthetic
    surface with an optional injected defect blob.

    This exists so downstream consumers can be developed and tested without
    a camera or PLC on the bench.
    """
    import cv2

    rng = np.random.default_rng(seed)
    h, w = frame_shape
    frames: list[dict[str, Any]] = []
    t_ms = 0.0
    for i in range(n_frames):
        frame = rng.normal(128, 20, size=(h // 4, w // 4, 3)).astype(np.uint8)
        frame = np.kron(frame, np.ones((4, 4, 1), dtype=np.uint8))[:h, :w]
        has_defect = bool(rng.random() < defect_probability)
        if has_defect:
            cy, cx = rng.integers(h // 5, h - h // 5), rng.integers(w // 5, w - w // 5)
            radius = int(rng.integers(10, max(11, min(h, w) // 8)))
            cv2.circle(frame, (int(cx), int(cy)), radius, (30, 30, 200), -1)
        t_ms += float(rng.uniform(50, 500))
        frames.append({
            "frame_id": i,
            "timestamp_ms": round(t_ms, 1),
            "has_defect": has_defect,
            "shape": list(frame.shape),
        })
    return frames
