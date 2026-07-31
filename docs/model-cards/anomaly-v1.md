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
