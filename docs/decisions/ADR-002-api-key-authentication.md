# ADR-002: API Key Authentication for API Contracts

**Status:** Accepted
**Date:** 2026-08-01
**Author:** LiamCarPer

## Context

AetherPdM exposes a REST API (FastAPI) for scoring, alerts, and asset management.
A B2B product must authenticate clients. The project targets a single-tenant MVP
that grows into multi-tenant (see ADR-003); auth was designed to support that path
from the start.

## Decision

Authenticate every `/v1/*` request with an **API key** sent in the `X-API-Key`
header:

1. **Key format** — `aether_<prefix>_<secret>` (192-bit random secret, hex-encoded
   so the format is unambiguous to parse).

2. **Storage** — only `key_prefix` (first 8 chars, for indexed lookup) and a
   PBKDF2-HMAC-SHA256 hash of the secret are persisted. The plaintext key is shown
   exactly once at creation (CLI: `scripts/manage_keys.py create`).

3. **Hash versioning** — stored hashes are versioned (`v2:{salt}:{dk}` at 600k
   iterations per OWASP 2023; legacy unversioned `salt:dk` at 100k still verifies,
   so rotation is transparent and no existing key is invalidated).

4. **Enforcement toggle** — `AETHER_API_KEY_AUTH_ENABLED` (default `false` for dev).
   When disabled, `require_api_key` returns the default tenant and all requests
   pass. When enabled, missing/invalid/revoked keys → HTTP 401.

5. **Open endpoints** — `/health`, `/metrics`, `/docs`, `/openapi.json` remain
   unauthenticated by design (orchestration + Prometheus scrape + API docs).

6. **Key management** — `src/aether_pdm/ops/apikeys.py` (create/list/revoke) +
   CLI. Keys carry an `org` column consumed by the multi-tenant layer (ADR-003).

## Consequences

**Positive:**
- Secure-by-default pattern ready for production (flip the env toggle)
- Plaintext secrets never at rest; rotation-friendly versioned hashes
- Auth foundation for tenant-scoped keys (ADR-003)

**Negative:**
- The Streamlit dashboard sends `X-API-Key` only when `AETHER_API_KEY` is set
  (empty default → unauthenticated dashboard still works in dev)
- `/metrics` and `/health` are intentionally open — must sit behind network policy
  or a proxy in production
- Key prefix lookup before hash verify creates a minor timing oracle on prefix
  existence (prefixes are exposed in `list` output; standard practice, not a vuln)

## Compliance

Every new `/v1/*` endpoint must require `require_api_key` and resolve the tenant
from the key. Open endpoints are limited to health/metrics/docs.
