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
    freqs = bearing_fault_frequencies(
        rpm=1772, n_balls=9, ball_diameter=0.3126, pitch_diameter=1.537
    )
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
    # Scale-invariant ratio features (v2) must exist when rpm is provided.
    for name in ["bpfo", "bpfi", "bsf", "ftf"]:
        assert f"{name}_ratio" in feats, f"Missing ratio feature: {name}_ratio"
    for key in ["bpfi_over_bpfo", "bsf_over_bpfo", "bpfi_over_bsf"]:
        assert key in feats, f"Missing pairwise dominance feature: {key}"


def test_compute_all_features_no_rpm_has_no_ratio_features():
    """Ratio features are only meaningful with fault frequencies; rpm=None must
    produce a feature set identical in shape to v1 (no ratio keys)."""
    np.random.seed(42)
    signal = np.random.randn(2048) * 0.1 + np.sin(np.linspace(0, 200 * np.pi, 2048))
    feats = compute_all_features(signal, sampling_rate=12000, rpm=None)
    for name in ["bpfo", "bpfi", "bsf", "ftf"]:
        assert f"{name}_ratio" not in feats, f"Unexpected ratio feature with rpm=None: {name}_ratio"
    for key in ["bpfi_over_bpfo", "bsf_over_bpfo", "bpfi_over_bsf"]:
        assert key not in feats, f"Unexpected pairwise feature with rpm=None: {key}"
    # Amplitude features are also absent without rpm (they are computed inside
    # the same block) — the base feature set must still be intact.
    for key in ["rms", "peak", "crest", "kurtosis", "skew", "envelope_peak", "envelope_energy"]:
        assert key in feats, f"Missing base feature: {key}"


def test_ratio_features_scale_invariant():
    """Ratios must be invariant to a global signal scale change (load/severity)."""
    np.random.seed(7)
    base = np.random.randn(4096) * 0.1 + np.sin(np.linspace(0, 400 * np.pi, 4096))
    feats_a = compute_all_features(base, sampling_rate=12000, rpm=1772)
    feats_b = compute_all_features(base * 3.0, sampling_rate=12000, rpm=1772)
    for key in ["bpfo_ratio", "bpfi_ratio", "bsf_ratio", "ftf_ratio",
                "bpfi_over_bpfo", "bsf_over_bpfo", "bpfi_over_bsf"]:
        assert feats_b[key] == pytest.approx(feats_a[key], rel=1e-6), (
            f"{key} should be scale-invariant, got {feats_a[key]} vs {feats_b[key]}"
        )
