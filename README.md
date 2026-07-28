# AetherPdM — Predictive Maintenance for Rotating Equipment

**B2B condition monitoring + predictive maintenance** for bearings and rotating machinery.
From vibration signal to fault classification, health score, and operational alerts.

## The Problem

Unplanned downtime in industrial rotating equipment costs **$50B+/year** globally.
Maintenance teams are flooded with false alarms (40–60% of alerts are noise), leading
to **alert fatigue** and missed real failures. Rule-based threshold systems don't adapt
to different machines, loads, or operating conditions.

## What AetherPdM Does

```
[ Sensor / File ] → [ Signal Features ] → [ Anomaly Score + Fault Class ] → [ Alert + Explain ]
```

- Ingest vibration waveforms (CWRU, Paderborn, custom)
- Extract domain-aware features — time-domain, frequency-domain, envelope
- Train anomaly detectors (IsolationForest) and fault classifiers (RandomForest)
- Serve via REST API with model versioning and lineage
- Persist alerts with explanations for operator review

## Architecture

```
                         ┌──────────────────────────────────┐
                         │         FastAPI (score)          │
                         │  POST /v1/assets/{id}/score      │
                         │  GET  /v1/alerts                 │
                         └──────┬──────────────────────────-┘
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
         ┌────────────┐ ┌────────────┐ ┌──────────────┐
         │ Anomaly    │ │ Fault      │ │ Risk/RUL     │
         │ Detector   │ │ Classifier │ │ (light)      │
         └────────────┘ └────────────┘ └──────────────┘
                │               │               │
                └───────────────┼───────────────┘
                                ▼
         ┌─────────────────────────────────────────┐
         │          Feature Store (Parquet)        │
         │   rms, peak, crest, kurtosis, BPFO,    │
         │   BPFI, BSF, band_power_*, fft_peak_*  │
         └─────────────────────────────────────────┘
                                ▲
         ┌─────────────────────────────────────────┐
         │       Signal DSP Pipeline               │
         │   Window → FFT → Envelope → Features    │
         └─────────────────────────────────────────┘
                                ▲
         ┌─────────────────────────────────────────┐
         │       Raw Store (Parquet)               │
         │   Normalized waveform by asset_id       │
         └─────────────────────────────────────────┘
```

## Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.12 |
| DSP/ML | NumPy, SciPy, scikit-learn |
| Tracking | MLflow |
| API | FastAPI + uvicorn |
| Database | PostgreSQL 16 |
| Storage | MinIO (S3-compatible) |
| Containers | Docker Compose |
| Observability | Prometheus metrics endpoint |

## Quick Start

```bash
# Prerequisites: Python 3.12, uv, Docker Desktop

# 1. Clone and install
git clone https://github.com/LiamCarPer/AetherPdM.git
cd AetherPdM
uv sync --group dev

# 2. Download CWRU dataset
uv run python -m aether_pdm.ingest.download_cwru

# 3. Start infrastructure
docker compose -f infra/docker/docker-compose.yml up -d

# 4. Train models
uv run python -m aether_pdm.models.train_anomaly
uv run python -m aether_pdm.models.train_fault

# 5. Run API
uv run uvicorn aether_pdm.serve.app:app --reload

# 6. Score an asset
curl -X POST http://localhost:8000/v1/assets/motor-001/score \
  -H "Content-Type: application/json" \
  -d '{"waveform": [0.1, 0.2, ...], "sampling_rate": 12000}'
```

## Metrics That Matter

| Metric | Why It Matters |
|--------|---------------|
| **Fault F1 / Balanced Accuracy** | Can we trust the classifier across all fault types? |
| **False Alarm Rate** | How many unnecessary work orders do we generate? |
| **Score p95 latency** | Can we score in real-time on the plant floor? |
| **Drift Detection → Retrain Impact** | Does the system degrade gracefully over time? |

## Dataset Strategy

| Dataset | Use |
|---------|-----|
| **CWRU** (Case Western) | MVP training and benchmarks — industry standard |
| **Paderborn (PU)** | Domain shift realism — real damage, not artificial |
| **Synthetic** (custom) | Drift simulation, multi-tenant testing, edge stress |

## Project Structure

```
aether-pdm/
├── configs/           # YAML configs for training, assets
├── data/              # Raw downloads → processed parquet
├── docs/              # Architecture, ADRs, domain notes
├── infra/docker/      # Docker Compose + Dockerfiles
├── src/aether_pdm/
│   ├── ingest/        # Data downloaders and normalizers
│   ├── signal/        # Windowing, FFT, envelope, DSP
│   ├── features/      # Feature computation pipelines
│   ├── models/        # Training, inference, calibration
│   ├── eval/          # Temporal split, anti-leakage metrics
│   ├── serve/         # FastAPI application
│   ├── db/            # SQLAlchemy models + repository
│   └── ops/           # Drift monitors, retrain triggers
├── pipelines/         # Prefect/orchestration entrypoints
├── services/api/      # API service entrypoint
├── services/worker/   # Background worker entrypoint
├── web/               # Ops dashboard (Streamlit)
├── tests/             # Unit, integration, contract tests
└── .github/workflows/ # CI pipeline
```

## Limits & Honest Scope

- **Vertical**: bearings and rotating equipment only (see ADR-001)
- **Algorithms**: scikit-learn based (not deep learning unless it clearly outperforms)
- **RUL**: light severity scoring, not full remaining-useful-life prediction (Fase 2)
- **Cloud**: not deployed; runs locally via Docker Compose (cloud target: Azure/AWS TBD)

## License

MIT
