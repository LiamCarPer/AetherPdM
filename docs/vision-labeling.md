# Model-assisted defect labeling with CVAT

CVAT (Computer Vision Annotation Tool) provides the semi-supervised labeling
workflow: a model pre-labels frames, a human reviews/corrects, and the
corrected annotations feed back into fine-tuning.

## Run CVAT locally

```bash
docker compose -f infra/cvat/docker-compose.yml up -d
# open http://localhost:8080 (admin/admin on first setup)
```

## Semi-supervised loop

1. Export raw inspection frames to `data/vision/raw/<batch>/`
2. Create a CVAT project with labels matching `NEU_CLASSES`
3. Upload frames as a CVAT task
4. Attach a model checkpoint (e.g. the latest `best.pt`) as an automatic
   annotation agent — CVAT calls it over its nuclio function interface
5. Human review: accept / fix / reject boxes in the CVAT UI
6. Export as YOLO 1.1 format, drop into `data/raw/neu_det/`, re-run
   `build_yolo_dataset` and fine-tune

## Why this matters

Manual bounding-box annotation is the dominant cost in production CV.
Model-assisted pre-annotation typically cuts human time per image by 50-80%
on stable classes, which is what makes periodic retraining affordable.
