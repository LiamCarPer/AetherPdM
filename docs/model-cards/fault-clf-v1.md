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
| F1 macro | >= 0.70 | Data-grounded: real CWRU val ceiling ~0.76 (see below) |
| Balanced accuracy | >= 0.70 | Matches f1_macro gate; random baseline is 0.25 (4 classes) |

## Real-Data Validation (2026-08)
Measured val f1_macro ceiling ≈ 0.76 on CWRU held-out files. The promotion
gate threshold of 0.70 is set below the measured ceiling and above chance
(0.25 for 4 classes), so a candidate must beat random by a wide margin to
promote.

- **Confusions documented**: ball↔outer@0.007, normal→ball 32%
- **Synthetic val achieves 1.0** — synthetic is NOT representative of real
  CWRU difficulty; gate calibration relies on real CWRU measurements only.
- Experiment sweep (all on real CWRU val): ratio features (scale-invariant)
  flat; severity-representative val split 0.74→0.76; HistGB /
  HistGB+balanced / RF 500/16/2/balanced_subsample at 0.70 / 0.69 / 0.75;
  feature selection (drop h2/h3, no amps, mutual-info top20) at
  0.76 / 0.76 / 0.71.

## Known Limitations
- CWRU/synthetic domain — real plant data may differ
- Fault severity (diameter) not modeled — just type
- Classifier confidence can be overconfident (mitigated by calibration)

## Governance
- ADR-001 compliant
- Promotion gate: f1_macro >= 0.70 AND balanced_accuracy >= 0.70 on val split
  (data-grounded 2026-08; see Real-Data Validation above)
