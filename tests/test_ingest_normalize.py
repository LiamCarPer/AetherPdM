"""Tests for CWRU normalizer."""

from aether_pdm.ingest.normalize_cwru import (
    LOAD_RPM,
    TEST_FILES,
    VAL_FILES,
    determine_split,
)


def test_determine_split():
    """Test files should be held out, others should be train."""
    for fid in TEST_FILES:
        assert determine_split(fid) == "test"
    assert determine_split("097") == "train"
    assert determine_split("105") == "train"


def test_determine_split_val():
    """Validation files should map to 'val', test files to 'test', rest to 'train'."""
    assert determine_split("97") == "val"
    assert determine_split("98") == "val"
    assert determine_split("109") == "val"
    assert determine_split("122") == "val"
    assert determine_split("134") == "val"
    assert determine_split("125") == "test"
    assert determine_split("105") == "train"


def test_split_sets_disjoint():
    """TEST_FILES and VAL_FILES must never overlap."""
    assert TEST_FILES.isdisjoint(VAL_FILES)


def test_load_rpm_mapping():
    """Standard CWRU loads should map to known RPMs."""
    assert LOAD_RPM[0] == 1797
    assert LOAD_RPM[1] == 1772
    assert LOAD_RPM[2] == 1750
    assert LOAD_RPM[3] == 1730
