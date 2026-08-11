# Terraform variables for the azurerm (Azure) provider — see main.tf.

variable "location" {
  description = "Azure region for all resources"
  type        = string
  default     = "westeurope"
}

variable "resource_prefix" {
  description = "Prefix for all resource names (keeps the demo resource group easy to identify and delete)"
  type        = string
  default     = "aetherpdm"
}

variable "app_name" {
  description = "Globally unique name of the App Service (Linux web app)"
  type        = string
  default     = "aetherpdm-api"
}

variable "postgres_admin_user" {
  description = "PostgreSQL Flexible Server administrator login"
  type        = string
  default     = "aether"
}

variable "postgres_admin_password" {
  description = "PostgreSQL Flexible Server administrator password. Required — supply via -var, TF_VAR_postgres_admin_password, or a secret-backed tfvars file. Never commit it."
  type        = string
  sensitive   = true
}

variable "ops_job_image" {
  description = "Container image for the ops cron job (must be built and pushed to a registry first)"
  type        = string
  default     = "aetherpdm.azurecr.io/aetherpdm/api:latest"
}

variable "tags" {
  description = "Tags applied to every resource"
  type        = map(string)
  default = {
    project     = "aetherpdm"
    environment = "demo"
  }
}
