# Contributing

AetherPdM is a production-style predictive-maintenance system for rotating
equipment: vibration ingestion, domain-aware signal features, anomaly/fault
models, and a gated promote/serve flow. Contributions that improve the signal
pipeline, the models, or the operations layer are welcome.

## Ground Rules

- **Conventional Commits** are required (`feat:`, `fix:`, `docs:`, `test:`,
  `ci:`, `chore:`, `refactor:`).
- Every change must pass CI: lint (ruff), type check (mypy), the fast test
  suite (`-m "not slow"`, with coverage), and the training smoke test.
- **No leaking future data into training**: the anti-leakage train/test split
  is a core invariant — changes to data handling must not regress it.
- **Promotion goes through the GatedOps gate contract**: `ops/promote.py`
  delegates to `gatedops` and records a `gatedops.manifest` lineage tag.
  Changes must keep that contract and its tests green.
- No secrets — API keys are managed via `scripts/manage_keys.py` and env vars,
  never in code or history.

## Development Setup

```bash
# Python 3.12+
uv sync --extra dev --extra web
pre-commit install

# Run the checks locally
uv run ruff check src/ web/
uv run mypy src/ web/ --config-file pyproject.toml
uv run pytest tests/ -m "not slow"
uv run pre-commit run --all-files

# Slow / download-gated tests (optional, need network + data)
uv run pytest tests/ -m "slow or download"
```

## Where Things Live

| Area | Path |
| :--- | :--- |
| Data ingestion (CWRU / Paderborn) | `src/aether_pdm/ingest/` |
| Signal processing (windowing, envelope, BPFO features) | `src/aether_pdm/signal/` |
| Models (anomaly, fault, calibration, training) | `src/aether_pdm/models/` |
| Operations (retrain, drift, domain shift, promote, scheduler) | `src/aether_pdm/ops/` |
| Serving API (FastAPI, auth, multi-tenant) | `src/aether_pdm/serve/`, `src/aether_pdm/db/` |
| Streamlit dashboard | `web/` |
| Docs (PRD, ADRs, model cards, scheduling) | `docs/` |
| Docker/Grafana/Prometheus infra | `infra/` |
| Tests | `tests/` |

## Making Changes

1. Fork the repository and create a branch (`git checkout -b feat/my-change`).
2. Make your change and add tests — every module has a dedicated test file;
   promotion and the GatedOps gate contract are covered in `tests/test_promote.py`.
3. Run the local checks (see above).
4. Open a pull request; CI runs automatically.

## Reporting Security Issues

See [SECURITY.md](SECURITY.md) — report privately, never in a public issue.
