# ADR-004: Multi-Modal Condition Monitoring (Vibration + Visual Inspection)

**Status:** Accepted
**Date:** 2026-08-12
**Author:** LiamCarPer

## Context

ADR-001 scoped AetherPdM to bearings/rotating equipment monitored through
accelerometer vibration. Target-role job descriptions for industrial CV demand
production-line camera skills: defect detection with YOLO-family models,
fine-tuning on small domain datasets, augmentation strategy, edge inference, and
rigorous evaluation (mAP, per-class AP, recall at false-positive budgets).

Bearing components (races, balls, rollers) are machined metal parts whose
surface defects are a real inspection target. Visual surface inspection is the
same asset, sensed by a second modality.

## Decision

Extend the vertical to **multi-modal condition monitoring**: vibration remains
the primary modality; visual surface inspection is added as a secondary one.

- First dataset: NEU-DET (hot-rolled steel strip surface defects, 6 classes,
  academic use) as a proxy for machined-metal surface inspection. Bearing-surface
  imagery is the target domain.
- Model: YOLOv8 fine-tuned from pretrained weights (not trained from scratch),
  consistent with how production teams actually work.
- Augmentation: Albumentations pipeline including copy-paste synthetic defect
  augmentation.
- Inference: exported ONNX consumed by onnxruntime. `ultralytics` is used for
  training/fine-tuning only and lives in the optional `vision` extra.

## Consequences

**Positive:**
- Directly evidenced camera-CV capability in an industrial setting.
- Multi-modal story matches real PdM vendor offerings.
- Reuses existing infrastructure: MLflow tracking, GatedOps promotion gates,
  ONNX export path, CI discipline.

**Negative:**
- `ultralytics` is AGPL-3.0. Confined to the training toolchain via the optional
  `vision` extra; shipped inference artifacts are ONNX weights consumed by
  MIT-licensed onnxruntime. Repos distributing this code commercially should
  review AGPL obligations first.
- Datasets are proxies (steel strip, not bearing imagery). Transfer to bearing
  imagery is future work requiring a labelled bearing-surface dataset.
- New heavy dependencies (`ultralytics`, `albumentations`, `opencv-python-headless`)
  confined to the optional `vision` extra; all imports lazy.

## Compliance

Vibration paths are untouched. All vision code lives under
`src/aether_pdm/vision/` with lazy imports and its own test modules marked
`slow` where heavy.
