# ADR-003: Multi-Tenant Access Control

**Status:** Accepted
**Date:** 2026-08-01
**Author:** LiamCarPer

## Context

AetherPdM targets B2B customers (maintenance teams at industrial plants). A single
deployment must serve multiple tenants (organizations) without cross-tenant data
leakage. The `Asset` model already carried `org`/`plant` string columns, and the
`ApiKey` model (ADR-002, auth) seeded an `org` column — but nothing enforced
isolation at the API layer.

## Decision

Tenant isolation is driven by the **API key's org**:

1. **Tenant resolution** — `require_api_key` returns a frozen `Tenant(org, key_name)`
   derived from the `ApiKey.org` column. When auth is disabled (dev mode), the
   tenant is `Tenant(org="default", key_name=None)`.

2. **Org-scoped queries** — every `/v1/*` endpoint scopes asset/alert/score reads
   by joining through `Asset.org`. `list_assets(org=...)`, `list_alerts(org=...)`,
   `list_scores(org=...)`, `get_asset(org=...)`.

3. **Write ownership guard** — `upsert_asset(expected_org=tenant.org)` rejects
   (HTTP 403 + transaction rollback) attempts to score/claim an asset owned by
   another org. This closes the cross-org asset hijack vector.

4. **Org/plant management** — `GET /v1/orgs`, `/v1/orgs/{org_id}/assets`,
   `/v1/orgs/{org_id}/plants`; cross-org access → 403 for real (non-default) tenants.

5. **Dev escape hatch** — when `tenant.org == "default"` (auth off), cross-org
   reads/upserts are allowed for convenience. Production keys carry real orgs, so
   this never leaks in production. Provisioning must NOT create keys with
   `org="default"` when auth is enabled.

## Consequences

**Positive:**
- Cross-tenant data isolation enforced at the query + write layers
- Closed a real cross-org asset hijack vulnerability (org B scoring org A's asset
  could previously steal ownership + read A's history)
- Scales to org → plant → asset hierarchy without schema changes (string keys)

**Negative:**
- Org/plant are string columns with no FK (referential integrity by convention,
  not constraint). Alembic migration needed if cascade deletes become required.
- `acknowledge_alert` is org-scoped in the repository but not yet HTTP-exposed.
- Batch scoring (`org=None`) is an admin operation spanning all tenants.

## Compliance

Every endpoint that touches tenant-scoped data must resolve the tenant from the
API key (or default tenant in dev) and scope queries accordingly. New endpoints
must follow the same pattern.
