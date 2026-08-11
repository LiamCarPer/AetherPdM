# AetherPdM — Azure free-tier deployment (Terraform)
#
# Target SKU line (see docs/azure-deployment-runbook.md):
#   App Service F1 (free) + PostgreSQL Flexible B1ms (32 GB) +
#   Blob Storage (MLflow artifacts) + Container Apps Job (cron ops loop).
#
# NOTE: committed as reviewable IaC — the deployment has NOT been executed.
# No live endpoint exists. Written to match the current azurerm 4.x schema;
# run `terraform init && terraform validate` before any apply.

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
}

locals {
  # MLflow tracking server runs as a Container App with external ingress in
  # this environment; FQDN under the environment's default domain.
  # (Placeholder wiring — confirm the actual app name at first apply.)
  mlflow_tracking_uri = "https://mlflow.${azurerm_container_app_environment.main.default_domain}"
}

# ---------------------------------------------------------------------------
# Resource group — teardown = delete this one RG
# ---------------------------------------------------------------------------
resource "azurerm_resource_group" "main" {
  name     = "${var.resource_prefix}-rg"
  location = var.location
  tags     = var.tags
}

# ---------------------------------------------------------------------------
# App Service (F1 = free tier) hosting the FastAPI + uvicorn app
# ---------------------------------------------------------------------------
resource "azurerm_service_plan" "api" {
  name                = "${var.resource_prefix}-plan"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "F1"
  tags                = var.tags

  lifecycle {
    prevent_destroy = false # demo: teardown expected after the demo week
  }
}

resource "azurerm_linux_web_app" "api" {
  name                = var.app_name
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  service_plan_id     = azurerm_service_plan.api.id
  https_only          = true
  tags                = var.tags

  app_settings = {
    "WEBSITES_PORT"               = "8000"
    "AETHER_DB_URL"               = "postgresql+psycopg://${var.postgres_admin_user}:${urlencode(var.postgres_admin_password)}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/aether_pdm"
    "AETHER_MLFLOW_TRACKING_URI"  = local.mlflow_tracking_uri
    "AETHER_API_KEY_AUTH_ENABLED" = "false" # demo: key auth off on free tier
  }

  site_config {
    application_stack {
      python_version = "3.12"
    }
    health_check_path = "/health"
  }

  lifecycle {
    prevent_destroy = false
  }
}

# ---------------------------------------------------------------------------
# PostgreSQL Flexible Server (Burstable B1ms, 32 GB) — scores/alerts/assets
# ---------------------------------------------------------------------------
resource "azurerm_postgresql_flexible_server" "main" {
  name                   = "${var.resource_prefix}-pg"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "16"
  administrator_login    = var.postgres_admin_user
  administrator_password = var.postgres_admin_password
  storage_mb             = 32768
  sku_name               = "B_Standard_B1ms"
  tags                   = var.tags

  lifecycle {
    prevent_destroy = false
  }
}

# Demo only: let Azure services reach Postgres (an F1 App Service cannot
# VNet-join). NOT a public-internet rule — 0.0.0.0 here means "Azure only".
resource "azurerm_postgresql_flexible_server_firewall_rule" "azure_services" {
  name             = "allow-azure-services"
  server_id        = azurerm_postgresql_flexible_server.main.id
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

resource "azurerm_postgresql_flexible_server_database" "main" {
  name      = "aether_pdm"
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "utf8"
  collation = "en_US.utf8"
}

# ---------------------------------------------------------------------------
# Blob storage (Standard LRS) — MLflow artifact store (replaces MinIO)
# ---------------------------------------------------------------------------
resource "azurerm_storage_account" "mlflow" {
  name                            = "${replace(var.resource_prefix, "-", "")}sa"
  resource_group_name             = azurerm_resource_group.main.name
  location                        = azurerm_resource_group.main.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  tags                            = var.tags

  lifecycle {
    prevent_destroy = false
  }
}

resource "azurerm_storage_container" "mlflow" {
  name                  = "mlflow-artifacts"
  storage_account_id    = azurerm_storage_account.mlflow.id
  container_access_type = "private"
}

# ---------------------------------------------------------------------------
# Container Apps — cron ops job (batch score -> drift -> retrain -> promote)
# ---------------------------------------------------------------------------
resource "azurerm_container_app_environment" "main" {
  name                = "${var.resource_prefix}-env"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  tags                = var.tags
}

resource "azurerm_container_app_job" "ops" {
  name                         = "${var.resource_prefix}-ops-job"
  resource_group_name          = azurerm_resource_group.main.name
  location                     = azurerm_resource_group.main.location
  container_app_environment_id = azurerm_container_app_environment.main.id
  tags                         = var.tags

  template {
    container {
      name   = "ops"
      image  = var.ops_job_image
      cpu    = "0.25"
      memory = "0.5Gi"

      command = ["python", "-m", "aether_pdm.ops.scheduler"]
      args = [
        "--features", "data/interim/features/features_v2.parquet",
        "--org", "default",
      ]

      env {
        name  = "AETHER_DB_URL"
        value = "postgresql+psycopg://${var.postgres_admin_user}:${urlencode(var.postgres_admin_password)}@${azurerm_postgresql_flexible_server.main.fqdn}:5432/aether_pdm"
      }
      env {
        name  = "AETHER_MLFLOW_TRACKING_URI"
        value = local.mlflow_tracking_uri
      }
    }
  }

  cron {
    name        = "ops-every-30-min"
    schedule    = "*/30 * * * *"
    timezone    = "UTC"
    concurrency = 1
  }

  lifecycle {
    prevent_destroy = false
  }
}
