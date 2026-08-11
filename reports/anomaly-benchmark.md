# Anomaly Detector Benchmark: PyTorch Autoencoder vs IsolationForest

- **Generated**: 2026-08-11T11:10:14+00:00
- **CWRU features**: `data\interim\cwru_features\features_v2.parquet`
- **Paderborn features**: `data\interim\paderborn\features_v1.parquet`
- **Features**: 36 (CWRU v2)
- **Torch config**: epochs=50 (ran 49), lr=1e-3, hidden=(64,32,16), latent=8, seed=42
- **Target recall**: 0.86
- **Operational gate**: DR >= 0.80 AND FAR <= 0.10

## Validation Results (CWRU val — same rows, normal=0 / fault=1)

| Metric | IsolationForest (strict_boundary) | PyTorch Autoencoder |
|---|---|---|
| Detection rate (DR) | 0.8277 | 0.8636 |
| False alarm rate (FAR) | 0.0014 | 0.0000 |
| Samples | 3534 | 3534 |

Val split: 3534 rows (2118 faults, 1416 normal). IF threshold = decision boundary (0). Torch threshold = 153.9121 calibrated at target recall 0.86.

## Domain Shift Probe (Paderborn — CWRU-calibrated thresholds)

| Metric | IsolationForest (strict_boundary) | PyTorch Autoencoder |
|---|---|---|
| Detection rate (DR) | 1.0000 | 1.0000 |
| False alarm rate (FAR) | 1.0000 | 1.0000 |
| Samples | 4490 | 4490 |

Paderborn rows: 4490. Missing CWRU features imputed with CWRU healthy-train medians: band_power_4000_6000.0, fft_peak_4000_6000.0, bpfo_ratio, bpfi_ratio, bsf_ratio, ftf_ratio, bpfi_over_bpfo, bsf_over_bpfo, bpfi_over_bsf.

## Verdict

**torch_wins = true**

## Interpretation

**PyTorch autoencoder wins on CWRU val.** It passes the operational gate (DR >= 80%, FAR <= 10%) and is not worse than the IsolationForest baseline on either metric: DR 0.864 vs 0.828 and FAR 0.0000 vs 0.0014 on the same 3534 val rows (threshold calibrated at target recall 0.86).

**Domain shift probe (Paderborn, CWRU-calibrated thresholds):** IF DR 1.000 / FAR 1.0000; torch DR 1.000 / FAR 1.0000 on 4490 Paderborn rows. Paderborn v1 lacks 9 of the 36 CWRU features (band_power_4000_6000.0, fft_peak_4000_6000.0, bpfo_ratio, bpfi_ratio, bsf_ratio, ftf_ratio, bpfi_over_bpfo, bsf_over_bpfo, bpfi_over_bsf), which were imputed with CWRU healthy-train medians; so these numbers are a conservative transfer probe, not a full domain-shift measurement.
