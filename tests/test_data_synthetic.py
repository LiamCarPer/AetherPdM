"""Tests for synthetic data generator."""

from pathlib import Path

import numpy as np
import pandas as pd

from aether_pdm.data.synthetic import generate_dataset, synthetic_waveform


def test_synthetic_waveform_shape():
    """Should return an array of the requested length."""
    w = synthetic_waveform(length=2048, sampling_rate=12000, fault_type="normal", seed=0)
    assert len(w) == 2048
    assert w.dtype == float


def test_synthetic_waveform_normal():
    """Normal waveform should have lower RMS than inner_race fault."""
    normal = synthetic_waveform(length=4096, rpm=1772, fault_type="normal", seed=42)
    faulty = synthetic_waveform(
        length=4096, rpm=1772, fault_type="inner_race", fault_diameter=0.021, seed=42
    )
    assert np.std(faulty) > np.std(normal)


def test_synthetic_waveform_seed_reproducibility():
    """Same seed should produce identical waveform."""
    a = synthetic_waveform(length=2048, seed=123)
    b = synthetic_waveform(length=2048, seed=123)
    np.testing.assert_array_almost_equal(a, b)


def test_generate_dataset_structure():
    """Generated dataset should have expected columns."""
    result = generate_dataset(Path("data/interim/test_synthetic"), n_normal=3, n_faulty=3, seed=0)
    df = pd.read_parquet(result)
    assert len(df) == 6
    assert "waveform" in df.columns
    assert "fault_type" in df.columns
    assert "split" in df.columns
    assert "severity" in df.columns


def test_generate_dataset_has_val_split():
    """Val split should exist and contain BOTH normal and faulty samples."""
    result = generate_dataset(
        Path("data/interim/test_synthetic_val"),
        n_normal=10,
        n_faulty=12,
        seed=42,
    )
    df = pd.read_parquet(result)
    val = df[df["split"] == "val"]
    assert not val.empty, "generate_dataset should emit a 'val' split"
    assert (val["fault_type"] == "normal").any(), "val split should hold normal samples"
    assert (val["fault_type"] != "normal").any(), "val split should hold faulty samples"

    # Deterministic split assignment: first n_val_normal normal rows are val,
    # first n_test_faulty faulty rows are test, next n_val_faulty are val.
    normal_rows = df[df["fault_type"] == "normal"]
    faulty_rows = df[df["fault_type"] != "normal"]
    assert set(normal_rows.iloc[:3]["split"]) == {"val"}
    assert set(normal_rows.iloc[3:]["split"]) == {"train"}
    assert set(faulty_rows.iloc[:5]["split"]) == {"test"}
    assert set(faulty_rows.iloc[5:10]["split"]) == {"val"}
    assert set(faulty_rows.iloc[10:]["split"]) == {"train"}
