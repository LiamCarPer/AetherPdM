# AetherPdM — Azure Deployment Runbook (free tier)

> Goal: take AetherPdM from "runs locally via Docker Compose" to a real cloud
> deployment with monitoring and a scheduled retrain loop, inside the free tier.
> This is the single highest-value interview signal for the MLOps pivot: the
> README's "not deployed" line is what interviewers probe first.
>
> Budget guardrail: Azure gives $200 of free credit for 30 days plus
> always-free services. Stay under it, and delete the resource group when the
> demo week is over. Re-verify current pricing/sku names before you run
> anything — Azure changes tiers often.

## 0. Prerequisites

- [ ] Azure account (free tier signup at https://azure.microsoft.com/free — needs a credit card for identity; you will NOT be charged while staying in free-credit usage)
- [ ] `az` CLI installed and logged in: `az login`
- [ ] A dedicated resource group so teardown is one command:
      `az group create --name aetherpdm-rg --location westeurope`
- [ ] Decide the deployment SKU line (see §1) — recommended: **App Service (F1, free) + PostgreSQL Flexible Server (Burstable B1ms, tiny) + Blob storage (free-ish) + Container Apps Jobs (cron)**

## 1. Infrastructure (one Terraform or az-cli pass)

| Component | Recommended | Why |
|---|---|---|
| API + inference | Azure App Service, F1 (free) — FastAPI via uvicorn | Zero cost, HTTPS by default |
| Model store / lineage | Azure Blob Storage (LRS) as MLflow artifact store | Replaces MinIO; ~free in demo volume |
| Feature/raw data | Blob + Parquet; keep Postgres for scores/alerts/assets | Matches current SQLAlchemy layer |
| Metadata DB | Azure Database for PostgreSQL Flexible Server, Burstable B1ms, 32GB | Smallest SKU; cancel after demo |
| MLflow server | Container App (single replica, or share the App Service) | Pairs with blob artifact store |
| Ops loop (batch score → drift → retrain → promote) | Azure Container Apps Job with cron trigger (schedule `*/30 * * * *`) | Native scheduler, autostop, cheap |
| Dashboard | Streamlit in a second App Service or `az webapp` on the F1 slot | Current `web/` works as-is |

Key az-cli sketch (verify SKU names before running):

```bash
az group create --name aetherpdm-rg --location westeurope

# Postgres (smallest SKU; ~EUR 15/mo equivalent, inside the $200 credit)
az postgres flexible-server create --resource-group aetherpdm-rg \
  --name aetherpdm-pg --sku-name Standard_B1ms --tier Burstable \
  --storage-size 32 --admin-user aether --admin-password '<generate, store in Key Vault>'

# App Service plan (F1 = free) + web app
az appservice plan create --resource-group aetherpdm-rg --name aetherpdm-plan \
  --sku F1 --is-linux
az webapp create --resource-group aetherpdm-rg --plan aetherpdm-plan \
  --name aetherpdm-api --runtime "PYTHON:3.12" --startup-file "uvicorn aether_pdm.serve.app:app --host 0.0.0.0 --port 8000"

# Blob storage for MLflow artifacts
az storage account create --resource-group aetherpdm-rg --name aetherpdmsa \
  --sku Standard_LRS
# → MLFLOW_TRACKING_URI + artifact store point here (see §2 env)

# Container Apps environment for the cron ops job
az containerapp env create --resource-group aetherpdm-rg \
  --name aetherpdm-env --location westeurope
```

## 2. Environment & secrets (never in git)

- App Service app settings:
  - `AETHER_DATABASE_URL` → Postgres connection string
  - `MLFLOW_TRACKING_URI` → MLflow server URL
  - `MLFLOW_ARTIFACT_URI` → `az://aetherpdmsa/artifacts` (or `blob://…`)
  - `AETHER_API_KEY_AUTH_ENABLED=true`
  - `AETHER_DEFAULT_ORG=acme`
- Use `az webapp config appsettings set` + Azure Key Vault for the admin password;
  do not put the password in compose/CI.
- Create the API key: `uv run python scripts/manage_keys.py create --name plant-1 --org acme`

## 3. Deploy + verify checklist (each item = a README update + CV evidence)

1. [ ] API live: `GET /health` returns 200 from the App Service URL
2. [ ] Model serving round-trip: `POST /v1/assets/motor-001/score` with a
      waveform returns health_score + `gatedops.manifest` lineage
3. [ ] Auth enforced: same call without `X-API-Key` → 401
4. [ ] Alerts visible: `GET /v1/alerts` returns persisted rows
5. [ ] Metrics endpoint reachable: `GET /metrics` shows
      `aetherpdm_predictions_total` counters incrementing
6. [ ] MLflow artifacts land in Blob (model versions 4+ after retrain)
7. [ ] Cron ops job runs on schedule: batch score → drift check → retrain →
      promote; check `az containerapp job execution list`
8. [ ] Readiness probe wired into App Service (health check path `/health`)
9. [ ] Load test sanity: 50 concurrent score requests stay under ~2s p95

## 4. README updates (turn the gap into evidence)

- Remove/replace "Cloud: not deployed; runs locally via Docker Compose (Azure/AWS TBD)"
- Add a `Deployment (Azure)` section: architecture diagram link, resource list,
  one-command bootstrap, teardown command
- Add the live endpoint URL + Grafana/Prometheus story if kept

## 5. Teardown (end of demo week, no surprises on the card)

```bash
az group delete --name aetherpdm-rg --yes --no-wait
```

## 6. Optional stretch (strong interview signals)

- **Edge story**: `mlflow models export-format onnx` (or `onnxruntime` serving)
  + a tiny standalone scorer container — "industrial ML = edge" is a phrase
  every PdM employer echoes (Augury, Siemens Energy)
- **Streaming ingest**: MQTT/Kafka → batch job — the gap report ranks
  "data pipeline / streaming" as the #1 requirement across the target boards
- **Drift report**: a documented domain-shift study (CWRU → Paderborn) already
  exists in `reports/` — surface it in the README
