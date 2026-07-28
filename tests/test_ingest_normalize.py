"""Tests for CWRU normalizer."""

from aether_pdm.ingest.normalize_cwru import LOAD_RPM, TEST_FILES, determine_split


def test_determine_split():
    """Test files should be held out, others should be train."""
    for fid in TEST_FILES:
        assert determine_split(fid) == "test"
    assert determine_split("097") == "train"
    assert determine_split("105") == "train"


def test_load_rpm_mapping():
    """Standard CWRU loads should map to known RPMs."""
    assert LOAD_RPM[0] == 1797
    assert LOAD_RPM[1] == 1772
    assert LOAD_RPM[2] == 1750
    assert LOAD_RPM[3] == 1730
