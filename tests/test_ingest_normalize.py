"""Tests for CWRU normalizer."""

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from aether_pdm.ingest.normalize_cwru import (
    CWRU_CATALOG,
    LOAD_RPM,
    TEST_FILES,
    VAL_FILES,
    build_feature_parquet,
    determine_split,
    normalize_file,
)


def test_determine_split():
    """Test files should be held out, others should be train."""
    for fid in TEST_FILES:
        assert determine_split(fid) == "test"
    assert determine_split("222") == "test"
    assert determine_split("097") == "train"
    assert determine_split("105") == "train"
    assert determine_split("199") == "train"


def test_determine_split_val():
    """Validation files should map to 'val', test files to 'test', rest to 'train'."""
    assert determine_split("97") == "val"
    assert determine_split("98") == "val"
    assert determine_split("109") == "val"
    assert determine_split("122") == "val"
    assert determine_split("197") == "val"
    assert determine_split("198") == "val"
    assert determine_split("222") == "test"
    assert determine_split("105") == "train"


def test_split_sets_disjoint():
    """TEST_FILES and VAL_FILES must never overlap."""
    assert TEST_FILES.isdisjoint(VAL_FILES)


def test_split_ids_exist_in_catalog():
    """Every TEST_FILES and VAL_FILES id must be a known catalog id.

    Guarantees the split sets only reference host-verified files that the
    downloader can actually fetch (catalog is the downloader's source of IDs).
    """
    catalog_ids = {fid for fid, _, _, _ in CWRU_CATALOG}
    assert TEST_FILES | VAL_FILES <= catalog_ids


def test_load_rpm_mapping():
    """Standard CWRU loads should map to known RPMs."""
    assert LOAD_RPM[0] == 1797
    assert LOAD_RPM[1] == 1772
    assert LOAD_RPM[2] == 1750
    assert LOAD_RPM[3] == 1730


def test_build_feature_parquet_no_files_raises_clear_error(tmp_path):
    """Empty/missing input dir -> RuntimeError with a clear message, not KeyError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(RuntimeError, match="No files were successfully normalized"):
        build_feature_parquet(empty_dir, tmp_path / "out")


def test_build_feature_parquet_zero_rows_raises_clear_error(tmp_path):
    """Unmappable .mat files (0 rows) -> RuntimeError, not KeyError('fault_type').

    Mirrors the old downloader bug: 'normal.mat' has a non-numeric stem, so
    no channel variables match and normalize_file yields 0 rows.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    sio.savemat(str(raw / "normal.mat"), {"unrelated_var": np.zeros(50)})

    with pytest.raises(RuntimeError, match="No files were successfully normalized"):
        build_feature_parquet(raw, tmp_path / "out")


def test_normalize_file_matches_padded_cwru_variables(tmp_path):
    """Official CWRU files name channels 'X097_DE_time' (zero-padded, X-prefixed)."""
    mat_path = tmp_path / "97.mat"
    sio.savemat(
        str(mat_path),
        {
            "X097_DE_time": np.zeros(12000),
            "X097_FE_time": np.zeros(12000),
            "X097RPM": np.array([1797.0]),
        },
    )

    df = normalize_file(mat_path)

    assert not df.empty
    assert set(df["channel"]) == {"DE", "FE"}
    assert (df["fault_type"] == "normal").all()
    assert (df["file_id"] == "97").all()
    assert (df["split"] == "val").all()
    assert (df["sampling_rate"] == 12000).all()


def test_normalize_file_accepts_unpadded_variable_names(tmp_path):
    """Some mirrors store 'X105_DE_time' without zero padding."""
    mat_path = tmp_path / "105.mat"
    sio.savemat(
        str(mat_path),
        {"X105_DE_time": np.zeros(12000), "X105RPM": np.array([1797.0])},
    )

    df = normalize_file(mat_path)

    assert not df.empty
    assert (df["fault_type"] == "inner_race").all()
    assert (df["split"] == "train").all()


def test_build_feature_parquet_writes_rows(tmp_path):
    """A properly downloaded numeric .mat normalizes to Parquet rows."""
    raw = tmp_path / "raw"
    raw.mkdir()
    sio.savemat(
        str(raw / "97.mat"),
        {"X097_DE_time": np.zeros(12000), "X097RPM": np.array([1797.0])},
    )

    out = build_feature_parquet(raw, tmp_path / "out")

    assert out.exists()
    result = pd.read_parquet(out)
    assert len(result) == 1
    assert result.iloc[0]["fault_type"] == "normal"
    assert result.iloc[0]["split"] == "val"
