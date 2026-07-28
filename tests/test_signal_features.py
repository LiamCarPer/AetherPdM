"""Tests for signal feature computation."""

import numpy as np
import pytest

from aether_pdm.signal.features import (
    bearing_fault_frequencies,
    compute_all_features,
    frequency_features,
    time_features,
)


def test_time_features_sine():
    """RMS of a unit-amplitude sine should be ~0.707."""
    fs = 1000
    t = np.arange(0, 1, 1 / fs)
    sine = np.sin(2 * np.pi * 50 * t)
    feats = time_features(sine)
    assert feats["rms"] == pytest.approx(0.707, rel=0.05)
    assert feats["peak"] == pytest.approx(1.0, rel=0.01)
    assert feats["crest"] == pytest.approx(1.414, rel=0.05)


def test_frequency_features_band_energy():
    """Energy in a band containing the dominant frequency should be highest."""
    fs = 1000
    t = np.arange(0, 1, 1 / fs)
    signal = np.sin(2 * np.pi * 100 * t)
    feats = frequency_features(signal, fs)
    # 100 Hz should fall in band (0, 500)
    assert "band_power_0_500" in feats
    assert feats["band_power_0_500"] > 0


def test_bearing_fault_frequencies_cwru():
    """CWRU SKF 6205 at 1772 RPM should produce plausible BPFO/BPFI."""
    freqs = bearing_fault_frequencies(rpm=1772, n_balls=9, ball_diameter=0.3126, pitch_diameter=1.537)
    assert 95 < freqs["bpfo"] < 115  # typical BPFO ~ 105 Hz
    assert 150 < freqs["bpfi"] < 170  # typical BPFI ~ 160 Hz


def test_compute_all_features_full():
    """Full feature pipeline should produce expected keys."""
    np.random.seed(42)
    signal = np.random.randn(2048) * 0.1 + np.sin(np.linspace(0, 200 * np.pi, 2048))
    feats = compute_all_features(signal, sampling_rate=12000, rpm=1772)
    for key in ["rms", "peak", "crest", "kurtosis", "skew", "envelope_peak", "envelope_energy"]:
        assert key in feats, f"Missing feature: {key}"
    for name in ["bpfo", "bpfi", "bsf"]:
        assert f"{name}_amp" in feats, f"Missing bearing feature: {name}_amp"
