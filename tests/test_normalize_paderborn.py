"""Tests for the Paderborn .mat normalizer."""

import numpy as np
import pandas as pd
import pytest

from aether_pdm.ingest.normalize_paderborn import (
    MIN_VIBRATION_SAMPLES,
    PADERBORN_CATALOG,
    SAMPLING_RATE,
    bearing_id_from_filename,
    build_feature_parquet,
    extract_rpm,
    extract_torque,
    extract_vibration,
    normalize_file,
    parse_mat,
    rpm_from_filename,
    torque_from_filename,
)


def _make_paderborn_mat(tmp_path, file_id="KI04", rpm=1500, n=1000, with_current=True):
    """Write a synthetic Paderborn-style .mat file and return its path."""
    import scipy.io as sio

    rng = np.random.default_rng(sum(ord(c) for c in file_id))
    mat = {
        "vibration": rng.standard_normal(n),
        "N": np.array([rpm]),
        "L": np.array([0.7]),
        "H": np.array([1000]),
    }
    if with_current:
        mat["current"] = np.random.default_rng(7).standard_normal(n)
    path = tmp_path / f"{file_id}.mat"
    sio.savemat(str(path), mat)
    return path


def test_catalog_contains_default_subset_files():
    """The catalog must cover the default download subset (K001, KI04, KO04)."""
    for file_id in ("K001", "KI04", "KO04"):
        assert file_id in PADERBORN_CATALOG


def test_normalize_file_healthy(tmp_path):
    """K001 -> normal, split test, 64 kHz, diameter 0.0, severity healthy."""
    path = _make_paderborn_mat(tmp_path, file_id="K001")
    df = normalize_file(path)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["file_id"] == "K001"
    assert row["fault_type"] == "normal"
    assert row["split"] == "test"
    assert row["sampling_rate"] == SAMPLING_RATE
    assert row["fault_diameter"] == 0.0
    assert row["severity"] == "healthy"
    assert row["channel"] == "vibration"
    assert len(row["waveform"]) == 1000
    assert isinstance(row["waveform"], list)


def test_normalize_file_inner_race(tmp_path):
    """KI04 -> inner_race, real damage, severity severe."""
    path = _make_paderborn_mat(tmp_path, file_id="KI04")
    row = normalize_file(path).iloc[0]

    assert row["fault_type"] == "inner_race"
    assert row["severity"] == "severe"
    assert np.isnan(row["fault_diameter"])  # unknown diameter for damaged


def test_normalize_file_outer_race(tmp_path):
    """KO04 -> outer_race, real damage."""
    path = _make_paderborn_mat(tmp_path, file_id="KO04")
    row = normalize_file(path).iloc[0]

    assert row["fault_type"] == "outer_race"
    assert row["severity"] == "severe"


def test_normalize_file_artificial_inner_race_severity(tmp_path):
    """KI01 (artificial) -> inner_race with severity incipient."""
    path = _make_paderborn_mat(tmp_path, file_id="KI01")
    row = normalize_file(path).iloc[0]

    assert row["fault_type"] == "inner_race"
    assert row["severity"] == "incipient"


def test_normalize_file_unknown_fallback(tmp_path):
    """Unknown file stem -> fault_type unknown, no crash."""
    path = _make_paderborn_mat(tmp_path, file_id="ZZ99")
    row = normalize_file(path).iloc[0]

    assert row["fault_type"] == "unknown"
    assert row["severity"] == "unknown"
    assert row["file_id"] == "ZZ99"


def test_normalize_file_rpm_extracted(tmp_path):
    """RPM should come from the N variable (e.g. 900)."""
    path = _make_paderborn_mat(tmp_path, file_id="K002", rpm=900)
    row = normalize_file(path).iloc[0]
    assert row["rpm"] == 900


def test_build_feature_parquet(tmp_path):
    """Multiple .mat files -> single Parquet, all rows split='test'."""
    _make_paderborn_mat(tmp_path, file_id="K001")
    _make_paderborn_mat(tmp_path, file_id="KI04")
    _make_paderborn_mat(tmp_path, file_id="KO04")

    out_dir = tmp_path / "out"
    result = build_feature_parquet(tmp_path, out_dir)

    assert result == out_dir / "paderborn_normalized.parquet"
    df = pd.read_parquet(result)
    assert len(df) == 3
    assert (df["split"] == "test").all()
    assert set(df["fault_type"]) == {"normal", "inner_race", "outer_race"}


def test_build_feature_parquet_no_files(tmp_path):
    """No .mat files -> FileNotFoundError."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No .mat files"):
        build_feature_parquet(empty, tmp_path / "out")


def test_parse_mat_strips_metadata(tmp_path):
    """parse_mat should return only real variables (no __header__ etc.)."""
    path = _make_paderborn_mat(tmp_path, file_id="K001")
    mat = parse_mat(path)

    assert "vibration" in mat
    assert "__header__" not in mat
    assert "__version__" not in mat
    assert "__globals__" not in mat


def test_extract_vibration_prefers_key(tmp_path):
    """The 'vibration' key should win over other large arrays."""
    path = _make_paderborn_mat(tmp_path, file_id="K001", n=2000)
    mat = parse_mat(path)

    vib = extract_vibration(mat)
    assert vib.shape == (2000,)
    assert np.issubdtype(vib.dtype, np.floating)


def test_extract_vibration_fallback(tmp_path):
    """Mat without 'vibration' -> largest 1-D numeric array is used."""
    import scipy.io as sio

    path = tmp_path / "K003.mat"
    sio.savemat(
        str(path),
        {
            "current": np.random.randn(5000),
            "N": np.array([1500]),
            "L": np.array([0.7]),
            "H": np.array([1000]),
        },
    )
    mat = parse_mat(path)

    vib = extract_vibration(mat)
    assert vib.shape == (5000,)
    assert np.allclose(vib, mat["current"])


def test_extract_vibration_missing(tmp_path):
    """Mat with only scalar/tiny arrays -> ValueError."""
    import scipy.io as sio

    path = tmp_path / "K004.mat"
    sio.savemat(
        str(path),
        {"N": np.array([1500]), "L": np.array([0.7]), "H": np.array([1000])},
    )
    mat = parse_mat(path)

    with pytest.raises(ValueError, match="vibration signal"):
        extract_vibration(mat)


def test_extract_vibration_ignores_small_arrays(tmp_path):
    """Arrays below MIN_VIBRATION_SAMPLES are not valid fallbacks."""
    import scipy.io as sio

    path = tmp_path / "K005.mat"
    sio.savemat(
        str(path),
        {
            "metadata": np.arange(MIN_VIBRATION_SAMPLES - 10),
            "N": np.array([1500]),
        },
    )
    mat = parse_mat(path)

    with pytest.raises(ValueError, match="vibration signal"):
        extract_vibration(mat)


def test_extract_rpm_defaults_to_1500():
    """Missing speed variable -> 1500."""
    assert extract_rpm({"L": np.array([0.7])}) == 1500.0


def test_extract_rpm_from_scalar():
    """N scalar (squeezed) -> float value."""
    assert extract_rpm({"N": np.array(900)}) == 900.0


def test_extract_torque():
    """L variable -> torque value."""
    assert extract_torque({"L": np.array([0.7])}) == 0.7
    assert extract_torque({}) == 0.0


# ---------------------------------------------------------------------------
# Official PU struct layout (verified against a real K001 archive)
# ---------------------------------------------------------------------------


def _make_pu_struct(n_vib1=5000, n_vib2=5000, speed=900.0):
    """Build a synthetic struct matching the official PU .mat layout.

    The variable is named after the file and holds a struct with ``Info``,
    ``X``, ``Y`` and ``Description`` fields. ``Y`` is a channel table with
    ``Name``/``Type``/``Data``/``Unit``/``Raster`` entries, including the
    ``vibration_1`` / ``vibration_2`` accelerometer channels.
    """
    channel_dtype = np.dtype(
        [("Name", "O"), ("Type", "O"), ("Data", "O"), ("Unit", "O"), ("Raster", "O")]
    )
    rng = np.random.default_rng(11)
    channels = np.empty(3, dtype=channel_dtype)
    channels["Name"] = np.array(["speed", "vibration_1", "vibration_2"], dtype=object)
    channels["Data"] = np.array(
        [
            np.full(500, speed),
            rng.standard_normal(n_vib1),
            rng.standard_normal(n_vib2),
        ],
        dtype=object,
    )
    channels["Type"] = np.array([4, 4, 4], dtype=object)
    channels["Unit"] = np.array(["", "", ""], dtype=object)
    channels["Raster"] = np.array(
        ["HostService", "HostService", "HostService"], dtype=object
    )

    struct_dtype = np.dtype(
        [("Info", "O"), ("X", "O"), ("Y", "O"), ("Description", "O")]
    )
    return np.array(
        (
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
            channels,
            np.empty(0, dtype=object),
        ),
        dtype=struct_dtype,
    )


def _make_pu_struct_mat(tmp_path, stem="N15_M07_F10_K001_4", n_vib=5000):
    """Write a synthetic PU-struct .mat file via savemat and return its path."""
    import scipy.io as sio

    path = tmp_path / f"{stem}.mat"
    sio.savemat(str(path), {stem: _make_pu_struct(n_vib1=n_vib, n_vib2=n_vib)})
    return path


def test_extract_vibration_from_struct(tmp_path):
    """Struct Y table with vibration channels -> vibration_1 extracted."""
    mat = {"N15_M07_F10_K001_4": _make_pu_struct()}
    vib = extract_vibration(mat)
    assert vib.shape == (5000,)
    assert np.issubdtype(vib.dtype, np.floating)


def test_extract_vibration_struct_prefers_largest_vibration():
    """vibration_2 (larger) should win over vibration_1."""
    mat = {"N15_M07_F10_K001_4": _make_pu_struct(n_vib1=3000, n_vib2=8000)}
    vib = extract_vibration(mat)
    assert vib.shape == (8000,)


def test_extract_vibration_struct_missing_raises():
    """Struct without vibration channels and no large arrays -> ValueError."""
    channels = np.array(
        [("speed", 4, np.full(50, 900.0), "", "HostService")],
        dtype=[("Name", "O"), ("Type", "O"), ("Data", "O"), ("Unit", "O"), ("Raster", "O")],
    )
    struct_dtype = np.dtype(
        [("Info", "O"), ("X", "O"), ("Y", "O"), ("Description", "O")]
    )
    struct = np.array(
        (
            np.empty(0, dtype=object),
            np.empty(0, dtype=object),
            channels,
            np.empty(0, dtype=object),
        ),
        dtype=struct_dtype,
    )
    mat = {"N15_M07_F10_K001_4": struct}

    with pytest.raises(ValueError, match="vibration"):
        extract_vibration(mat)


def test_extract_rpm_from_struct_speed_channel():
    """No top-level N -> mean of the struct speed channel."""
    mat = {"N15_M07_F10_K001_4": _make_pu_struct(speed=900.0)}
    assert extract_rpm(mat) == 900.0


def test_normalize_file_real_pu_filename_healthy(tmp_path):
    """Real PU filename N15_M07_F10_K001_4 -> normal, 1500 rpm, 0.7 Nm."""
    path = _make_paderborn_mat(tmp_path, file_id="N15_M07_F10_K001_4", rpm=1500)
    row = normalize_file(path).iloc[0]

    assert row["file_id"] == "N15_M07_F10_K001_4"
    assert row["fault_type"] == "normal"
    assert row["severity"] == "healthy"
    assert row["rpm"] == 1500  # from filename N15
    assert row["load_hp"] == 0.7  # from filename M07


def test_normalize_file_real_pu_filename_inner_race(tmp_path):
    """Real PU filename N09_M07_F10_KI04_1 -> inner_race severe, 900 rpm."""
    path = _make_paderborn_mat(tmp_path, file_id="N09_M07_F10_KI04_1", rpm=900)
    row = normalize_file(path).iloc[0]

    assert row["fault_type"] == "inner_race"
    assert row["severity"] == "severe"
    assert row["rpm"] == 900  # from filename N09
    assert np.isnan(row["fault_diameter"])


def test_normalize_file_struct_roundtrip(tmp_path):
    """Full path: savemat struct -> parse_mat -> normalize_file."""
    path = _make_pu_struct_mat(tmp_path, stem="N09_M07_F10_K001_1", n_vib=5000)
    row = normalize_file(path).iloc[0]

    assert row["fault_type"] == "normal"
    assert row["sampling_rate"] == SAMPLING_RATE
    assert row["rpm"] == 900  # filename N09 (mat has no top-level N)
    assert len(row["waveform"]) == 5000
    assert row["split"] == "test"


def test_bearing_id_from_filename():
    """PU filename -> catalog bearing ID."""
    assert bearing_id_from_filename("N15_M07_F10_K001_4") == "K001"
    assert bearing_id_from_filename("N09_M07_F10_KI04_1") == "KI04"
    assert bearing_id_from_filename("K001") == "K001"
    assert bearing_id_from_filename("N09_M07_F10_KO04_2") == "KO04"


def test_rpm_from_filename():
    """N<rpm/10> convention."""
    assert rpm_from_filename("N15_M07_F10_K001_4") == 1500.0
    assert rpm_from_filename("N09_M07_F10_K001_4") == 900.0
    assert rpm_from_filename("K001") is None


def test_torque_from_filename():
    """M<torque*10> convention."""
    assert torque_from_filename("N15_M07_F10_K001_4") == 0.7
    assert torque_from_filename("N15_M01_F04_K001_4") == 0.1
    assert torque_from_filename("K001") is None
