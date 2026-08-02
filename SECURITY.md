# Security Policy

## Scope

AetherPdM is a **B2B condition-monitoring / predictive-maintenance system** for
rotating equipment: vibration ingestion, signal features, anomaly/fault models,
REST serving with API-key authentication, and operator-facing alerts. It is a
production-style reference implementation, not a deployed service.

In scope for security reports:
- **Authentication & tenancy:** API-key handling (`ops/apikeys.py`,
  `serve/auth.py`), multi-tenant access control, key rotation
- **Model & artifact integrity:** promotion logic that delegates to the
  GatedOps gate, `gatedops.manifest` lineage tags, feature/artifact loading
- **Serving API:** FastAPI endpoints (`/v1/assets/{id}/score`, `/v1/alerts`)
  and inference behavior (input validation, resource limits)
- **Data handling:** ingestion scripts, feature store (Parquet), and database
  access patterns

Out of scope:
- Vulnerabilities in third-party dependencies (FastAPI, scikit-learn, MLflow,
  SQLAlchemy) — report those to their respective maintainers

## Reporting a Vulnerability

Please **do not** open a public issue for security-sensitive findings.

Report privately via GitHub's **Private Vulnerability Reporting** (Security
tab → Report a vulnerability), or open a standard issue if the finding is not
sensitive.

What to include:
- A clear description of the issue and its impact (e.g., "a tenant can read
  another tenant's alerts")
- Steps to reproduce (ideally with the compose stack and API calls)
- Suggested remediation, if known

## Disclosure Policy

- **Acknowledgement:** you will be acknowledged for validated reports (unless you prefer to remain anonymous).
- **Response target:** an initial response within 5 business days.
- **Fix window:** validated critical/high findings are prioritized; fixes land through the CI pipeline (lint, type check, fast tests, training smoke).

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `main` | Yes — CI enforced (lint, type check, tests with coverage, training smoke) |
| Latest release (see [Releases](https://github.com/LiamCarPer/AetherPdM/releases)) | Yes |
| Older releases | No — upgrade |

## Verification

Every fix is verified by the automated gates: lint (ruff), type check (mypy),
the fast test suite (`-m "not slow"`, with coverage), the training smoke test,
and the promotion tests that exercise the GatedOps gate contract and lineage
manifest recording.
