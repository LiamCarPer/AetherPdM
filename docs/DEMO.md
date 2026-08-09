# AetherPdM demo — end-to-end walkthrough

This guide takes a first-time visitor from a fresh clone to a working score:
synthetic vibration data -> DSP features -> anomaly + fault models -> promotion
through the GatedOps gate -> a live scoring request that echoes lineage.

No CWRU download, no Docker, and no pre-trained models are needed: everything
is deterministic and generated locally. This is the same path the
`scripts/bootstrap_demo.py` script was built for.

---

## 1. Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/LiamCarPer/AetherPdM.git
cd AetherPdM
uv sync --extra dev --extra web
```

---

## 2. Train and promote both models (fresh clone, no data download)

```bash
uv run python scripts/bootstrap_demo.py
```

This runs the whole loop:

1. generates a deterministic synthetic dataset (train/val/test splits, fixed
   seed) under `data/demo_bootstrap/`;
2. runs the signal pipeline (windowing, FFT, envelope) to produce a features
   Parquet file;
3. trains the anomaly detector (IsolationForest with a strict healthy-boundary
   threshold) and the fault classifier (RandomForest);
4. **promotes both through the GatedOps gate** — the decision is made by
   `gatedops.gate.engine.evaluate_gate` against the thresholds in
   `configs/promote.yaml`, and every promoted version records a
   `gatedops.manifest` tag;
5. prints a ready-to-run scoring request.

Expected outcome (identifiers and versions vary):

```text
=== Promoting anomaly model ===
  decision=promoted (Metrics pass gate: DR=1.0000 >= 0.8, FAR=0.0000 <= 0.1)
=== Promoting fault model ===
  decision=promoted (Metrics pass gate: f1_macro=1.0000 >= 0.9, balanced_accuracy=1.0000 >= 0.9)
```

If either gate rejects, the script exits non-zero — the same fail-closed
behavior that keeps bad models out of production in CI.

## 3. Serve the models

The bootstrap registry lives in the workdir
(`data/demo_bootstrap/mlflow.db`), so point the API at it. From the repo
root:

```bash
AETHER_MLFLOW_TRACKING_URI=sqlite:///data/demo_bootstrap/mlflow.db \
  uv run uvicorn aether_pdm.serve.app:app --port 8000
```

Alternatively run uvicorn from inside the workdir:

```bash
cd data/demo_bootstrap
uv run uvicorn aether_pdm.serve.app:app --port 8000
```

By default (`AETHER_API_KEY_AUTH_ENABLED` unset) the API accepts requests
without a key — development mode. When auth is enabled, send the key in the
`X-API-Key` header (see `scripts/manage_keys.py`).

## 4. Score a waveform

```bash
curl -s -X POST http://localhost:8000/v1/assets/synth-demo/score \
  -H "Content-Type: application/json" \
  -d '{"waveform": [0.1, 0.2, ...], "sampling_rate": 12000}'
```

The bootstrap script prints a command with a real waveform snippet, or
generate one yourself:

```bash
uv run python -c "import json; from aether_pdm.data.synthetic import synthetic_waveform; print(json.dumps({'waveform': synthetic_waveform(2048, fault_type='inner_race', fault_diameter=0.021, seed=1).tolist(), 'sampling_rate': 12000}))"
```

A score response looks like (values illustrative):

```json
{
  "health_score": 0.31,
  "anomaly_score": 0.87,
  "fault": {"class": "inner_race", "confidence": 0.92},
  "alert": {"level": "critical", "reason": "detected_inner_race_fault"},
  "top_features": [{"name": "kurtosis", "contribution": 0.45}],
  "model_versions": {"anomaly": "1", "fault": "1"},
  "lineage": {
    "anomaly": {
      "model_name": "aether-anomaly",
      "model_version": "1",
      "artifact_hash": "...",
      "git_sha": "...",
      "run_id": "...",
      "data_hash": "..."
    },
    "fault": {
      "model_name": "aether-fault-clf",
      "model_version": "1",
      "artifact_hash": "...",
      "git_sha": "...",
      "run_id": "...",
      "data_hash": "..."
    }
  }
}
```

The `lineage` block comes from the `gatedops.manifest` tag recorded at
promotion time — the same lineage contract [GatedOps](https://github.com/LiamCarPer/GatedOps)
uses for its reference models.

## 5. Browse the registry

Point the MLflow UI at the bootstrap registry:

```bash
uv run python -m mlflow server --backend-store-uri sqlite:///data/demo_bootstrap/mlflow.db \
  --host 127.0.0.1 --port 5001
```

Open http://localhost:5001: `aether-anomaly` and `aether-fault-clf` each have
promoted versions carrying the `gatedops.manifest` and `gatedops.status` tags,
plus the `staging` alias on the latest trained version.

## 6. The drift -> retrain -> promote loop

The autonomous ops loop is a CLI:

```bash
uv run python scripts/run_ops_pipeline.py --features data/interim/features/features_v2.parquet --org acme
```

Drift detection decides whether to retrain; retrained models go to `staging`;
the promotion gate (GatedOps engine) decides whether they reach `production`.
If the gate rejects, the previous production model stays active.

## 7. Real data (CWRU / Paderborn)

The synthetic path needs no downloads, but the repo also supports the standard
bearing datasets. The README covers the full flow:

```bash
uv run python -m aether_pdm.ingest.download_cwru
uv run python -m aether_pdm.signal.pipeline
uv run python -m aether_pdm.models.train \
  --features data/interim/features/features_v1.parquet \
  --anomaly-config configs/train_anomaly.yaml \
  --fault-config configs/train_fault.yaml
```

## Troubleshooting

- **The API says no models are available.** The API and the bootstrap must
  point at the same registry — set `AETHER_MLFLOW_TRACKING_URI` or run uvicorn
  from the workdir.
- **First run is slow.** The first contact with a fresh registry creates the
  MLflow schema; later runs are fast.
- **Auth errors (401).** Only when `AETHER_API_KEY_AUTH_ENABLED=true`; send an
  `X-API-Key` header (create keys with `scripts/manage_keys.py`).
- **Versions/ids differ from this guide.** Every run creates new ids and
  versions depend on registry history — the shape of the output is what
  matters.
