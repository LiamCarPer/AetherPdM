"""Tests for signal pipeline."""

from pathlib import Path

import numpy as np

from aether_pdm.data.synthetic import generate_dataset, synthetic_waveform
from aether_pdm.signal.pipeline import FEATURE_VERSION, process_dataset, process_waveform


def test_process_waveform_returns_features():
    """Pipeline should produce feature columns for a valid waveform."""
    w = synthetic_waveform(length=4096, rpm=1772, seed=42)
    df = process_waveform(w, sampling_rate=12000, rpm=1772, window_size=2048, overlap=0.5)
    assert len(df) > 0
    assert "rms" in df.columns
    assert "kurtosis" in df.columns
    assert "bpfo_amp" in df.columns
    assert "bpfi_amp" in df.columns
    assert "window_id" in df.columns


def test_process_waveform_short_signal():
    """Very short waveform should return empty DataFrame."""
    w = np.array([0.1, 0.2, 0.3])
    df = process_waveform(w, sampling_rate=12000, window_size=1024)
    assert len(df) == 0


def test_process_waveform_fault_type_differentiation():
    """Normal and faulty waveforms should produce different kurtosis."""
    normal = synthetic_waveform(length=4096, rpm=1772, fault_type="normal", seed=42)
    faulty = synthetic_waveform(
        length=4096, rpm=1772, fault_type="inner_race", fault_diameter=0.021, seed=42
    )
    df_n = process_waveform(normal, 12000, rpm=1772)
    df_f = process_waveform(faulty, 12000, rpm=1772)
    assert df_f["peak"].mean() > df_n["peak"].mean()


def test_process_dataset_synthetic():
    """End-to-end pipeline on synthetic data should produce features."""
    input_path = generate_dataset(
        Path("data/interim/test_pipeline"), n_normal=2, n_faulty=2, seed=0
    )
    result = process_dataset(input_path, window_size=1024, overlap=0.5, max_waveforms=2)
    assert "feature_version" in result.columns
    assert result["feature_version"].iloc[0] == FEATURE_VERSION, (
        f"Expected feature_version={FEATURE_VERSION}, got {result['feature_version'].iloc[0]}"
    )
    assert "fault_type" in result.columns
    assert "rms" in result.columns
    assert "kurtosis" in result.columns
    # v2 feature set: ratio features must be present on rpm-carrying rows.
    assert "bpfi_over_bpfo" in result.columns
