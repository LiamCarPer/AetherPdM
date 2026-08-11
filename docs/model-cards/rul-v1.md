# Model Card: AetherPdM Degradation-Trend RUL Estimator (v1)

## Summary
Remaining-useful-life estimate for the dashboard / CV demo: fits a linear
trend on a degradation index (`1 - health_score`) over elapsed hours and
extrapolates it to a failure threshold (`models/rul.py`).

**Honest scope (read first):** this is a **degradation-trend extrapolation,
NOT a calibrated time-to-failure model**. AetherPdM's datasets (CWRU,
Paderborn, synthetic) contain no run-to-failure ground truth, so the hours
reported here cannot be validated as true time-to-failure. Every output
carries this disclaimer.

## Model Details
- **Algorithm**: ordinary least squares linear trend via `scipy.stats.linregress`
  - `RUL = (failure_threshold - current_index) / slope`
- **Input**: time series of `(elapsed_hours, health_score)` per asset
  (e.g. rows from `score_records`; source: `serve/inference.py` health score)
- **Output**: `rul_hours` (float or `None`), status, slope, R², confidence
  flag, 95% CI band on the extrapolated time-to-threshold
- **Degradation index**: `1 - clip(health_score, 0, 1)` (0 = pristine, 1 = failure)
- **Version**: v1 (no MLflow artifact — stateless estimator, no training)
- **Entrypoints**:
  - `aether_pdm.models.rul.DegradationTrendRUL` (library)
  - `aether_pdm.models.rul.estimate_rul_from_scores` (convenience wrapper)
  - `scripts/estimate_rul.py` (CLI: `--scores-json <file>` or `--synthetic`)

## Intended Use
- Maintenance-planning aid: prioritize inspections / schedule maintenance
  for assets that show a **consistent, detectable degradation trend**
- Demo / CV narrative for "health → degradation → remaining-life" flow
- Input to human review, not to autonomous shutdown or safety decisions

## Honest Guards (behavioral contract)
| Condition | Result |
|-----------|--------|
| `< min_points` observations (default 5) | `RUL = None`, status `insufficient_data` |
| Slope ≤ 0 (flat / improving) | `RUL = None`, status `no_detectable_degradation_trend` |
| Current index ≥ failure threshold | `RUL = 0.0`, status `failure_threshold_reached` |
| R² < 0.5 (or unbounded CI) | still reported, but `confidence = "low"` |
| Any valid fit | `confidence` is at most `"medium"` — never `"high"` |

Confidence is deliberately capped at `"medium"` even for R² ≈ 1: the
trend → failure-threshold mapping is **unvalidated extrapolation**, so a
perfect in-sample fit cannot claim calibrated certainty. The 95% CI band
reflects slope uncertainty only (from `linregress` standard error); model /
threshold uncertainty is carried by the R² flag and the confidence label.

## Training Data
- **None.** No ML training happens in this module — the "model" is a linear
  fit computed on the asset's own score history at call time.
- Synthetic degradation ramps (`aether_pdm.data.synthetic.degradation_ramp`,
  rising fault diameter over time) exist **for demos and tests only** —
  they are simulated trajectories, not run-to-failure ground truth.

## Evaluation
- Unit tests: `tests/test_rul.py` (9 tests) covering the degradation-index
  mapping, fit summary, positive-trend RUL + CI, no-trend guard,
  insufficient-data guard, low-R² flag, and the synthetic ramp end-to-end.
- **No validation against real run-to-failure data. None exists in the
  repository.** Calibrating this estimator's hours against true
  time-to-failure on NASA IMS or XJTU-SY bearing run-to-failure datasets is
  explicit future work.

## Known Limitations
- Assumes **linear** degradation in the index — real wear-in / wear-out
  curves are often nonlinear; extrapolation quality degrades far from the
  observed window.
- Health-score scale is calibrated on CWRU/synthetic data; the index →
  threshold mapping may not transfer to other machines or operating
  conditions (see the domain-shift findings in `anomaly-v1.md`).
- A flat or improving trend legitimately yields no RUL — that is a feature,
  not a bug; it prevents inventing failure dates from noise.
- Single health score per timestamp; sensor dropouts are dropped (logged).

## Governance
- PRD: calibrated time-to-failure remains a won't-have; this trend RUL is
  the honest interim (see `docs/PRD.md`).
- Every result dict from `estimate_rul_from_scores` includes a `disclaimer`
  string; the CLI prints it on every run.
- Future work gate: before any production use of the hours themselves,
  validate against run-to-failure datasets (NASA IMS / XJTU-SY) and
  re-calibrate the failure threshold.

## Demo Output (reproducible)
```text
$ uv run python scripts/estimate_rul.py --synthetic
RUL estimate for asset 'asset-001' (degradation-trend extrapolation)
  status      : ok
  RUL         : 30.9 hours (95% CI: 29.9 - 32.0)
  slope       : 0.001826 index/hour
  R^2         : 0.993
  confidence  : medium (never 'high' - unvalidated extrapolation)
  n_points    : 24
  threshold   : 0.9
```
Seed 42, 24 inspection points over 480 hours, outer-race fault ramp.
