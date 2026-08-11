# Model Card: AetherPdM Anomaly Detector (v1)

## Summary
Unsupervised anomaly detection on bearing vibration feature vectors using IsolationForest.
Flags windows that deviate from healthy baseline.

## Model Details
- **Algorithm**: IsolationForest (scikit-learn)
- **Input**: Feature vector (RMS, kurtosis, crest factor, skew, band powers, envelope energy)
- **Output**: anomaly_score (0-1) + binary anomaly flag
- **Version**: v1 (MLflow registered: `aether-anomaly`)

## Intended Use
- Real-time health scoring of rotating equipment (motors, pumps, compressors)
- Detect bearing faults (inner race, outer race, ball) before catastrophic failure
- Output feeds alerting rules in the AetherPdM API

## Training Data
- **Source**: CWRU bearing dataset + synthetic waveforms
- **Split**: file-level (no leakage between train/val/test)
- **Label**: only healthy (normal) windows used for training
- **Feature version**: v1

## Evaluation Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| False Alarm Rate (FAR) | <= 10% | Operationally critical (alert fatigue) |
| Detection Rate (recall) | >= 80% | How many real faults caught |
| Threshold | calibrated on val split | Via `calibrate.py` |

## Known Limitations
- Trained on CWRU/synthetic data — domain shift to other equipment expected
- Single-axis accelerometer data only
- No RUL prediction — this is health scoring, not remaining life

## Governance
- ADR-001: bearings/rotating equipment only
- Promoted to production via gate (FAR <= 10%, recall >= 80%)

## Deep Learning Benchmark (PyTorch Autoencoder, 2026-08-11)

The PyTorch CV claim is backed by a registered, benchmarked artifact:
`aether-anomaly-torch` (`aether_pdm.models.torch_anomaly.TorchAnomalyDetector`,
MLP autoencoder: 36 → 64 → 32 → 16 → 8 → 16 → 32 → 64 → 36, ReLU, MSE, Adam,
trained on 1,890 healthy CWRU v2 train windows, early-stopped at epoch 39/50).
Full protocol and code: `scripts/run_benchmark_anomaly.py` →
`reports/anomaly-benchmark.md`.

| Model | DR (CWRU val) | FAR (CWRU val) | DR (Paderborn*) | FAR (Paderborn*) |
|---|---|---|---|---|
| IsolationForest (strict_boundary) | 0.8277 | 0.0014 | 1.0000 | 1.0000 |
| PyTorch Autoencoder | **0.8636** | **0.0000** | 1.0000 | 1.0000 |

**Verdict: torch_wins = True** on the same 3,534 val rows (2,118 faults,
1,416 normal; normal=0, fault=1). The AE passes the production gate
(DR >= 0.80, FAR <= 0.10) and Pareto-dominates the IF baseline: higher
detection rate (0.864 vs 0.828) at lower false-alarm rate (0.000 vs 0.0014)
with its threshold calibrated at target recall 0.86.

\* Paderborn probe uses CWRU-calibrated thresholds and imputes the 9 CWRU
features missing from Paderborn v1 with CWRU healthy-train medians. Both
models saturate at DR=FAR=1.0 — severe cross-domain distribution shift
(mean PSI 5.65, see `reports/domain-shift-cwru-to-paderborn.md`); neither
threshold transfers, so this is a documented limitation, not a model
difference. Reproducible: seed 42, CPU-only, `uv run python
scripts/run_benchmark_anomaly.py --cwru-features
data/interim/cwru_features/features_v2.parquet --paderborn-features
data/interim/paderborn/features_v1.parquet --output
reports/anomaly-benchmark.md --epochs 50`.
