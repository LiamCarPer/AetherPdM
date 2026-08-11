# Agent Context — AetherPdM

> Durable state file for agent sessions. **Read this first** before doing anything
> in this repo. Update it when state changes significantly. Be honest — this is
> the resume mechanism when a session is interrupted.

---

## 1. What this repo is

**AetherPdM** — B2B predictive maintenance (PdM) for rotating equipment (bearings).
Python 3.12, uv-managed monorepo at `src/aether_pdm/`. It is the **vertical product**
paired with **GatedOps** (platform repo: `LiamCarPer/GatedOps`, pinned
`@v0.2.0` git dependency). Dependency direction is intentional and one-way:
Aether depends on GatedOps; GatedOps never depends on Aether.

Stack: FastAPI + SQLAlchemy + MLflow + scikit-learn + Prometheus/Grafana + Streamlit
dashboard + Docker Compose (9 services) + GitHub Actions CI.

---

## 2. Completed state (verified, CI-green)

- **Full MLOps lifecycle**: ingest (CWRU + Paderborn + synthetic) → signal pipeline
  (v2 features incl. fault-frequency ratio features) → train (IsolationForest anomaly,
  RandomForest fault) → calibrate → GatedOps-gated promote → FastAPI serve → alerts DB →
  Streamlit dashboard → drift monitors → retrain → batch scoring (hysteresis + cooldown)
  → scheduled ops pipeline → Prometheus metrics → API key auth → multi-tenant (org/plant).
- **Fresh-clone demo works**: `scripts/bootstrap_demo.py` (synthetic → features → train →
  GatedOps promote → print curl). CI-proven scoring path (test_inference_smoke.py no
  longer skipped; 4 smoke tests pass on fresh runner).
- **Compose stack boots in CI** (`compose-smoke` job — caught & fixed 4 real Docker bugs:
  git missing in builder, README.md not copied, MLflow root-path poll, missing psycopg2).
- **Real CWRU data verified end-to-end** (50 files, catalog remapped to host-verified IDs,
  splits train/val/test with severity-representative VAL_FILES). Anomaly gate passes on
  real CWRU (DR=0.86, FAR=0.002 with strict_boundary). **Fault gate recalibrated 0.90→0.70**
  (empirical ceiling ~0.76 f1_macro on real CWRU val; documented in promote.py, promote.yaml,
  PRD, model card). Promote now returns `decision=promoted` on real CWRU.
- **Tests**: ~270 fast + 4 smoke, all green. `src/` ruff+mypy clean (tests/ has ~7
  pre-existing style warnings — do not fix unless asked).
- **Governance**: ADR-001 (bearings-only), ADR-002 (API key auth), ADR-003 (multi-tenant),
  PRD, model cards, domain-shift report, VHS demo GIFs committed to repo.

---

## 3. CURRENT MISSION (in progress, interrupted)

User directive: implement **all 6 missing items** (ranked by interview impact), then
report back. See todo list in conversation for full detail.

| # | Item | Why | Status |
|---|------|-----|--------|
| 1 | **PyTorch anomaly detector beating IsolationForest baseline** (autoencoder/1D-CNN on CWRU/Paderborn, MLflow, gated) | CV claims PyTorch; 3 target companies demand DL | **INTERRUPTED — half-started** |
| 2 | **Committed cloud deployment** (terraform/bicep, configs, README; NO live-endpoint claim) | README "not deployed" line = #1 interview vulnerability | Not started |
| 3 | **ONNX export + lightweight edge scorer** | Backs Jetson/TensorRT + edge story | Not started |
| 4 | **Streaming ingest** (MQTT or Kafka → features) | "Data pipeline" = #1 demanded skill; repo is batch-only | Not started |
| 5 | **GPU training script** (single-GPU torch, CPU-fallback CI smoke) | Backs GPU claims; cheap once torch exists | Not started (depends on #1) |
| 6 | **RUL estimator** (degradation-trend, honestly scoped — NOT fake time-to-failure) | Stretch item | Not started |

### R1 exact state (resume from here)

**Files modified (NOT committed):**
- `pyproject.toml` — torch dep added. ⚠️ **HAS A BUG**: duplicate `[[tool.uv.index]]`
  block at the bottom (one `explicit = true` named "pytorch-cpu" — correct — plus a bare
  duplicate). **Clean up on resume**: keep the named explicit index + `[tool.uv.sources]`
  torch = { index = "pytorch-cpu" }, delete the stray bottom `[[tool.uv.index]]` block.
- `uv.lock` — regenerated with torch entry.

**NOT created yet:** `src/aether_pdm/models/torch_anomaly.py`,
`src/aether_pdm/ops/benchmark_anomaly.py`, `scripts/run_benchmark_anomaly.py`,
`tests/test_torch_anomaly.py`. No torch code exists.

**Torch install state:** wheel download was in progress when interrupted; verify with
`uv run python -c "import torch; print(torch.__version__)"` (may still be downloading;
CPU-only wheel via pytorch-cpu index).

**Baseline to beat (measured, real CWRU v2 features on disk):**
`data/interim/cwru_features/features_v2.parquet` — IsolationForest strict_boundary:
**DR=0.86, FAR=0.002 on val**. Gate: DR>=0.80 AND FAR<=0.10.
Paderborn features on disk: `data/interim/paderborn/features_v1.parquet` (4490 rows).

### Required architecture constraints for R1 (from original delegation)
- torch imports must be **lazy** (inside functions) — `import aether_pdm` must NOT import torch.
- `TorchAnomalyDetector` (MLP autoencoder): fit(healthy-only), anomaly_scores=MSE,
  find_threshold(target_recall), save/load, deterministic seed, CPU default.
- `ops/benchmark_anomaly.py` — honest comparison vs IF on val + Paderborn, MLflow log,
  markdown report. Win OR documented honest loss both acceptable.
- Tests `@pytest.mark.slow` (torch training slow; keep fast CI clean).
- MLflow model name `aether-anomaly-torch` (distinct from sklearn).

---

## 4. Resume instructions (next agent)

1. Fix the duplicate `[[tool.uv.index]]` block in `pyproject.toml` (delete the bare one
   at the bottom), keep the named explicit pytorch-cpu index.
2. Verify torch installs: `uv sync && uv run python -c "import torch; print(torch.__version__)"`.
3. Implement R1 per the delegation spec above (files listed). Run the REAL benchmark:
   `uv run python scripts/run_benchmark_anomaly.py --cwru-features data/interim/cwru_features/features_v2.parquet --paderborn-features data/interim/paderborn/features_v1.parquet --output reports/anomaly-benchmark.md --epochs 50`. Report real numbers.
4. Proceed R3 (GPU script — depends on torch existing), R2 (ONNX), R4 (streaming), R5 (RUL), R6 (cloud configs — NO live-endpoint claim, cannot verify from this env).
5. Final: full QA gate (ruff src/, mypy src/, pytest fast+smoke), ops check, commit, push, report to user.

---

## 5. Environment & operational notes

- **OS**: Windows (win32). Shell: PowerShell 5.1. **No Docker Desktop** (virtualization
  disabled) — Docker is verified ONLY via the `compose-smoke` CI job. **No cloud creds**.
- **uv** is the package manager. Local venv: `.venv`. MLflow local: `sqlite:///mlflow.db`.
- **git**: commits use conventional style (`feat|fix|test|docs|chore(scope): ...`).
  Do NOT commit `docs/azure-deployment-runbook.md` (untracked, belongs to another thread)
  unless told.
- **Slow local**: this Windows box is heavily loaded; full pytest takes ~6 min; the
  inference smoke (bootstrap) takes ~2 min on CI runners but can take 15+ min locally
  under load. Prefer CI to prove slow paths.
- **Data on disk** (all gitignored): CWRU raw (50 .mat), `data/interim/cwru/cwru_normalized.parquet`,
  `data/interim/cwru_features/features_v2.parquet`, `data/interim/paderborn/features_v1.parquet`.
- **Known pre-existing test lint**: ~7 ruff warnings in tests/ (E501/N806 etc.) — leave alone.

---

## 6. Change log

- 2026-08-10: Created this file. R1 (PyTorch) interrupted mid-`uv add torch`. pyproject+uv.lock
  modified, NOT committed, pyproject has duplicate uv.index block to fix.
