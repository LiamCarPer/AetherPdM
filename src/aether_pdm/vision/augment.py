"""Albumentations augmentation pipeline for defect detection.

Config-driven Compose built at call time; albumentations is imported lazily
inside the factory so importing this module stays light.
"""

from __future__ import annotations

from typing import Any


def build_augmentation_pipeline(config: dict[str, Any] | None = None, imgsz: int = 640) -> Any:
    """
    Build an Albumentations Compose from a config dict.

    Recognized keys (all optional):
        hflip: float            probability of horizontal flip (default 0.5)
        vflip: float            probability of vertical flip (default 0.0)
        brightness_contrast: {p: float, ...kwargs}
        gauss_noise: {p: float, ...kwargs}
        copy_paste: {p: float}  requires mask input; skipped gracefully if unsupported

    Returns the composed transform. Callers pass dicts with 'image' (and
    optionally 'bboxes'/'mask') to it.
    """
    import albumentations as A  # noqa: N812 - conventional alias

    cfg = dict(config or {})

    transforms: list[Any] = []

    hflip_p = float(cfg.get("hflip", 0.5))
    if hflip_p > 0:
        transforms.append(A.HorizontalFlip(p=hflip_p))

    vflip_p = float(cfg.get("vflip", 0.0))
    if vflip_p > 0:
        transforms.append(A.VerticalFlip(p=vflip_p))

    bc = cfg.get("brightness_contrast")
    if bc:
        params = dict(bc) if isinstance(bc, dict) else {}
        p = float(params.pop("p", 0.3))
        transforms.append(A.RandomBrightnessContrast(p=p, **params))

    gn = cfg.get("gauss_noise")
    if gn:
        params = dict(gn) if isinstance(gn, dict) else {}
        p = float(params.pop("p", 0.2))
        transforms.append(A.GaussNoise(p=p, **params))

    # Copy-paste synthetic augmentation. Albumentations >= 1.4 does not ship a
    # generic copy-paste for detection boxes in all versions, so we implement
    # paste support as a no-op placeholder when requested but unavailable,
    # and document that defect-paste augmentation is handled by the dataset
    # generator instead. Keeping the key in configs makes intent explicit.
    cp = cfg.get("copy_paste")
    if cp and isinstance(cp, dict) and hasattr(A, "CopyPaste"):
        params = {k: v for k, v in cp.items() if k != "p"}
        try:
            transforms.append(A.CopyPaste(p=float(cp.get("p", 0.3)), **params))
        except Exception:
            pass  # version without CopyPaste: skip silently

    return A.Compose(transforms)


def default_detector_augmentations() -> dict[str, Any]:
    """Default augmentation config matching configs/vision/detector.yaml."""
    return {
        "hflip": 0.5,
        "vflip": 0.5,
        "brightness_contrast": {"p": 0.3},
        "gauss_noise": {"p": 0.2},
    }
