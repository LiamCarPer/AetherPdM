# Scheduling: Autonomous Ops Loop

The scheduled ops pipeline (`aether_pdm.ops.scheduler`) is the capstone
orchestration of the monitoring loop. It is **cron-compatible**: it exits `0`
on success and non-zero on failure, so it can be driven by cron, a systemd
timer, or a scheduled container without any custom glue.

## What the Loop Does

Each run executes four stages in order:

| # | Stage | Module | What happens |
|---|-------|--------|--------------|
| 1 | **Batch score** | `ops/batch_scorer.py` | Score every registered asset (org-scoped) with hysteresis + cooldown alert rules; persist scores + alerts. |
| 2 | **Drift check** | `ops/drift.py` | Compare train (reference) vs test (production) feature distributions (PSI + KS). |
| 3 | **Retrain decision** | `ops/retrain.py` | If drift fired, retrain anomaly + fault models on current data. |
| 4 | **Promote** | `ops/promote.py` | Run promotion gates; new models are promoted only if metrics pass. If rejected, the previous production models stay active (implicit rollback). |

The scheduler does **not** re-implement any of these stages — it calls the
existing modules in sequence and aggregates their results.

## CLI

```bash
uv run python -m aether_pdm.ops.scheduler \
  --features data/interim/features/features_v1.parquet \
  --org acme \
  --hysteresis 3 \
  --cooldown-min 30 \
  --drift-threshold 0.25 \
  --batch-limit 100
```

A thin wrapper exists at `scripts/run_ops_pipeline.py` for convenience.

| Flag | Default | Purpose |
|------|---------|---------|
| `--features` | `data/interim/features/features_v1.parquet` | Features Parquet used for drift + retrain |
| `--org` | all tenants | Only score assets of this org (admin op: `None` = all) |
| `--mlflow-uri` | `$AETHER_MLFLOW_TRACKING_URI` or `sqlite:///mlflow.db` | MLflow tracking URI |
| `--hysteresis` | `3` | Consecutive non-healthy scores before an alert |
| `--cooldown-min` | `30` | Minutes to suppress re-alerting same asset+level |
| `--drift-threshold` | `0.25` | Mean PSI at/above which drift forces retraining |
| `--no-retrain` | `False` | Skip the retrain/promote stage even if drift fired |
| `--batch-limit` | `100` | Max assets scored per batch run |

## Cron Setup

```cron
# Every 30 minutes, score + evaluate drift + retrain if needed (org-scoped)
*/30 * * * * cd /path/to/AetherPdM && uv run python scripts/run_ops_pipeline.py \
  --features data/interim/features/features_v1.parquet \
  --org acme >> /var/log/aetherpdm_ops.log 2>&1
```

Because the process exits non-zero on failure, cron will email/alert on
failures (or you can wrap it in a supervisor that restarts / pages).

## Docker Scheduling

The compose file (`infra/docker/docker-compose.yml`) ships a `batch` service
gated behind the `batch` profile, so `docker compose up` does not start it.

```bash
# One-shot run against compose MLflow + Postgres
docker compose -f infra/docker/docker-compose.yml --profile batch run batch

# Build it first if the image has not been built
docker compose -f infra/docker/docker-compose.yml --profile batch build batch
```

The `batch` service:

- Uses the same build context + `Dockerfile.train` as the `train` service
  (no separate image, no profile conflict — `train` stays gated behind `train`).
- Overrides the image entrypoint to run `aether_pdm.ops.scheduler` (the
  `train` entrypoint would otherwise win).
- Talks to the compose MLflow (`AETHER_MLFLOW_TRACKING_URI`) and Postgres
  (`AETHER_DB_URL`), so scores/alerts land in the same DB as the API.
- Mounts the `data` named volume so the features Parquet
  (`/app/data/interim/features/features_v1.parquet`) can be shared with the API.

### Scheduling the container

Run it on a schedule with a systemd timer, or mount the container into cron:

**systemd timer** (`/etc/systemd/system/aetherpdm-batch.{service,timer}`):

```ini
# aetherpdm-batch.service
[Unit]
Description=AetherPdM scheduled ops pipeline

[Service]
WorkingDirectory=/path/to/AetherPdM
ExecStart=/usr/bin/docker compose -f infra/docker/docker-compose.yml --profile batch run --rm batch
```

```ini
# aetherpdm-batch.timer
[Unit]
Description=Run AetherPdM ops pipeline every 30 min

[Timer]
OnCalendar=*:0/30

[Install]
WantedBy=timers.target
```

```bash
systemctl daemon-reload
systemctl enable --now aetherpdm-batch.timer
```

**Cron on the host**:

```cron
*/30 * * * * cd /path/to/AetherPdM && docker compose -f infra/docker/docker-compose.yml --profile batch run --rm batch
```

## Failure Handling

- **Exit codes**: `0` = full loop succeeded; `1` = fatal error (e.g. retrain
  pipeline raised). Use the exit code for cron/systemd alerting.
- **Non-fatal, recorded errors** (still exit 0, the loop continues):
  - Features file missing → batch still runs; drift recorded as
    `{"error": "features file not found"}`; retrain skipped.
  - No assets registered → `scored=0`; drift/retrain still evaluated.
  - Engine has no models / MLflow unavailable → per-asset errors are recorded
    in `batch["errors"]`; the pipeline still returns.
  - Promotion gate rejects → new models stay in staging; previous production
    stays active (implicit rollback), reported via `retrain["outcome"]`.
- **Logging**: run with `>> logfile 2>&1` and check the structured summary
  printed at the end (`assets_scored`, `alerts_raised`, `drift_fired`,
  `retrained`, `promoted`).

## Multi-Tenant Note

- Pass `--org acme` to score only `acme`'s assets (drift/retrain run on the
  shared features file regardless of org scope).
- Omit `--org` to score **all tenants** — treat this as an admin operation
  when running scheduled.
