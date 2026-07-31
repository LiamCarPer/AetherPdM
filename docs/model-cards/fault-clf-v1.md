# Model Card: AetherPdM Fault Classifier (v1)

## Summary
Multi-class bearing fault classification (normal, inner_race, outer_race, ball)
using RandomForest on vibration feature vectors.

## Model Details
- **Algorithm**: RandomForestClassifier (scikit-learn)
- **Input**: Feature vector (time/freq/envelope features + bearing fault frequencies)
- **Output**: fault class + confidence probability
- **Version**: v1 (MLflow registered: `aether-fault-clf`)

## Intended Use
- Classify detected faults to guide maintenance action
- Confidence threshold to avoid false positives
- Complements anomaly detector (which triggers first)

## Training Data
- **Source**: CWRU + synthetic
- **Classes**: normal, inner_race, outer_race, ball
- **Split**: file-level anti-leakage
- **Feature version**: v1

## Evaluation Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| F1 macro | >= 0.90 | All classes balanced |
| Balanced accuracy | >= 0.90 | Handles class imbalance |

## Known Limitations
- CWRU/synthetic domain — real plant data may differ
- Fault severity (diameter) not modeled — just type
- Classifier confidence can be overconfident (mitigated by calibration)

## Governance
- ADR-001 compliant
- Promotion gate: f1_macro >= 0.90 AND balanced_accuracy >= 0.90 on val split
