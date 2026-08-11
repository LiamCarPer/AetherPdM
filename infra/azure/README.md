# AetherPdM — Azure Terraform (free-tier)

Terraform for the free-tier Azure deployment described in
[`docs/azure-deployment-runbook.md`](../../docs/azure-deployment-runbook.md).

## Status — honest

**These configs are committed; the deployment has NOT been executed.** No live
endpoint exists. There are no Azure credentials wired into CI and no `terraform`
binary is assumed on this machine — run the steps below on a machine with
`terraform` + `az` when an Azure account is available (free $200 credit).

These files were written to match the current `azurerm` 4.x schema but have
**not** been `terraform validate`d yet. Before any apply:

```bash
terraform init
terraform validate
```

## Prerequisites

- Azure account with the free $200 credit (https://azure.microsoft.com/free)
- `az` CLI, logged in: `az login`
- Terraform >= 1.5 (https://developer.hashicorp.com/terraform/downloads)

## Plan / apply

```bash
terraform init
terraform plan   -var postgres_admin_password='<generate-and-store>'
terraform apply -var postgres_admin_password='<generate-and-store>'
```

The password is marked `sensitive` in `variables.tf`; prefer
`TF_VAR_postgres_admin_password` or a secret-backed tfvars file over shell
history. Never commit the password.

## What gets created

| Resource | SKU / note |
|---|---|
| Resource group `aetherpdm-rg` | teardown = delete this one RG |
| Service plan + Linux web app (FastAPI) | `F1` free |
| PostgreSQL Flexible Server + `aether_pdm` DB | `B_Standard_B1ms`, 32 GB |
| Storage account + `mlflow-artifacts` container | Standard LRS blob |
| Container App Environment + cron job | `*/30 * * * *` ops loop |

The ops job image (`var.ops_job_image`, default
`aetherpdm.azurecr.io/aetherpdm/api:latest`) must be built and pushed before the
first scheduled run; the API code itself is deployed to the App Service
separately (`az webapp deploy` / CI).

## Cost guardrails (stay in the free tier)

- `F1` App Service plan = free (60 CPU-min/day, 1 GB RAM — fine for a demo).
- Postgres `B1ms` ≈ EUR 15/mo equivalent — inside the $200 credit, but cancel
  after the demo week.
- Blob LRS: effectively free at demo volume.
- Container Apps Job: billed per execution only; a 30-min cron ≈ 48 short
  runs/day.
- Set a budget alert in the portal (e.g. $50) so nothing sneaks past.

## Teardown (end of demo week)

```bash
terraform destroy -var postgres_admin_password='<generate-and-store>'
# or, last resort (bypasses Terraform state):
az group delete --name aetherpdm-rg --yes --no-wait
```

Deleting the resource group removes everything in one command.
