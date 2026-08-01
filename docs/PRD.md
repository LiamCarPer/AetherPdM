# AetherPdM — Product Requirements Document

**Status:** Draft v1  
**Date:** 2026-07-28  
**Author:** LiamCarPer

---

## 1. Problem Statement

Industrial rotating equipment (motors, pumps, compressors) experiences unplanned downtime costing $50B+ annually. Most facilities rely on either:
- **Reactive maintenance** — fix after failure (highest cost, longest downtime)
- **Calendar-based preventive maintenance** — replace parts on schedule regardless of condition (wastes useful life)
- **Rule-based condition monitoring** — fixed thresholds per machine (40–60% false alarm rate → alert fatigue)

AetherPdM addresses the gap between "no data" and "expensive enterprise PdM suite" with an open, API-first predictive maintenance system for bearings.

**Target user:** Maintenance engineers and reliability teams at mid-size industrial plants (50–500 assets) who want ML-driven insights without a six-figure vendor contract.

## 2. Users & Stakeholders

| Role | Needs | How AetherPdM Helps |
|------|-------|---------------------|
| **Maintenance Engineer** | Know which asset needs attention today, not next month | Health score per asset, ranked by severity |
| **Reliability Manager** | Reduce unplanned downtime, measure ROI | False alarm rate, lead-time gains, drift detection |
| **ML Engineer** (future) | Retrain models when conditions change | Versioned models, drift monitors, retrain gate |

## 3. Use Cases

### Must-have (Phase 1)

| ID | Use Case | Acceptance |
|----|----------|------------|
| UC-01 | **Score a single asset** — POST a vibration waveform, receive health score + fault class + alert | Response < 500ms p95 for 1s window |
| UC-02 | **Review active alerts** — list recent alerts sorted by severity | Filterable by asset, level, time range |
| UC-03 | **Train initial models** — reproduce CWRU pipeline end-to-end | Train anomaly + fault clf, log to MLflow |
| UC-04 | **Understand a fault** — see which features drove the decision | Top-3 features with contribution scores |

### Should-have (Phase 2)

| ID | Use Case | Acceptance |
|----|----------|------------|
| UC-05 | **Multi-asset topology** — org → plant → asset hierarchy | Config-driven, no code changes |
| UC-06 | **Batch scoring** — score all assets on schedule | Cron trigger or Prefect flow |
| UC-07 | **Drift detection** — alert when feature distributions shift | PSI/KS monitor per feature |
| UC-08 | **Domain shift eval** — train on CWRU, evaluate on Paderborn | Documented performance delta |
| UC-09 | **API key security** — authenticate API requests | Valid key → 200; missing/invalid/revoked → 401; plaintext never stored |
| UC-10 | **Tenant isolation** — org-scoped data access | Cross-org read → 403; cross-org write → 403 + rollback; dev-mode default org documented |

### Won't-have (explicitly out of scope)

- Gearbox / belt / hydraulic diagnostics (see ADR-001)
- Real RUL prediction (severity ordinal only, not time-to-failure)
- Mobile app (responsive web only)
- Real-time streaming (file-upload and batch only)

## 4. Success Metrics

| Metric | Target | Why |
|--------|--------|-----|
| **Fault F1 (macro)** | ≥ 0.90 on CWRU held-out files | Core classifier quality |
| **False Alarm Rate** | ≤ 10% at operational threshold | Reduces alert fatigue, builds trust |
| **Score latency p95** | ≤ 500 ms | Real-time enough for plant floor |
| **Data quality fail rate** | ≤ 2% of windows | Catch sensor / wiring issues |
| **Model retrain cycle** | Documented drift → retrain → improve or rollback | MLOps maturity |

## 5. Release Criteria (Phase 1)

- [ ] `docker compose up` — all services running
- [ ] 2 models in MLflow registry with version tags
- [ ] Scoring endpoint returns `ScoreResponse` for CWRU-like input
- [ ] Alerts persisted to PostgreSQL
- [ ] All tests passing in CI
- [ ] Demo: normal → inner race → outer race → ball, each with explanation

## 6. Constraints

- **Vertical:** bearings only (rolling-element)
- **Stack:** Python 3.12, scikit-learn (not deep learning unless proven better)
- **Data:** CWRU (MVP) → Paderborn (depth) → Synthetic (MLOps)
- **Deployment:** Docker Compose (cloud TBD after Phase 2)
