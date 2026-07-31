# Domain Shift Report: CWRU to Paderborn

- **Source features**: `data\interim\features\features_v1.parquet`
- **Target features**: `data\interim\paderborn\features_v1.parquet`
- **Generated**: 2026-07-31T23:07:44+00:00
- **Models loaded**: yes

## Summary

- **Mean PSI**: 5.648
- **Severely drifted features**: 27
- **Worst feature**: band_power_1000_2000 (PSI = 8.283)

## Anomaly Model Performance

| Metric | Source (CWRU) | Target (Paderborn) | Delta |
|---|---|---|---|
| false_alarm_rate | N/A | 1.0000 | N/A |
| detection_rate | N/A | 1.0000 | N/A |
| best_detection_rate | N/A | 0.7540 | N/A |

**Notes:**
- Source: Validation split has no normal samples (split='test'); cannot compute false alarm rate.

## Fault Classifier Performance

| Metric | Source (CWRU) | Target (Paderborn) | Delta |
|---|---|---|---|
| f1_macro | 1.0000 | 0.0917 | -0.9083 |
| balanced_accuracy | 1.0000 | 0.0961 | -0.9039 |

## Feature Drift (top 10 by PSI)

| Feature | PSI | KS stat | Status |
|---|---|---|---|
| band_power_1000_2000 | 8.2831 | 0.9519 | severe |
| band_power_2000_4000 | 8.2831 | 0.9996 | severe |
| crest | 8.2831 | 0.9923 | severe |
| ftf_amp | 8.2831 | 1.0000 | severe |
| fft_peak_1000_2000 | 8.2831 | 0.9941 | severe |
| kurtosis | 8.2831 | 0.9943 | severe |
| ftf_h2_amp | 8.2831 | 1.0000 | severe |
| fft_peak_2000_4000 | 8.2831 | 1.0000 | severe |
| band_power_0_500 | 7.9358 | 0.9891 | severe |
| fft_peak_0_500 | 7.4531 | 0.9728 | severe |

## Interpretation

Severe feature drift detected (mean PSI = 5.648, 27 feature(s) severe). CWRU-trained models are unlikely to transfer to Paderborn without retraining or domain adaptation.

The worst-drifting feature is **band_power_1000_2000** (PSI = 8.283).

Fault classifier macro-F1 degrades by 0.908 on Paderborn, confirming that sensor/system differences matter beyond raw feature drift.
