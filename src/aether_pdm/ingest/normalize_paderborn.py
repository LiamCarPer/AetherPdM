"""
Paderborn .mat -> normalized Parquet pipeline (same unified schema as CWRU).

Unified schema:
  asset_id, file_id, channel, sampling_rate, rpm, load_hp,
  fault_type, fault_diameter, severity, split, waveform

Paderborn maps to:
  fault_type: normal | inner_race | outer_race | ball (from file prefix)
  fault_diameter: 0.0 for healthy, unknown (np.nan) for damaged
  severity: "healthy" | "incipient" (artificial) | "severe" (real) | "unknown"
  split: "test" (target domain for domain shift study)
  rpm: from N variable (1500 or 900)
  sampling_rate: 64000

The Paderborn distribution ships .rar archives containing .mat files. Each .mat
holds raw vibration (and optional motor current) recorded at 64 kHz. The normalizer
handles both plain-variable .mat files and struct-style files via
``scipy.io.loadmat(..., squeeze_me=True)``.

Usage:
    python -m aether_pdm.ingest.normalize_paderborn \
        [--input data/raw/paderborn] [--output data/interim/paderborn]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scipy.io as sio
except ImportError:
    sio = None  # type: ignore

# ---------------------------------------------------------------------------
# Paderborn dataset constants
# ---------------------------------------------------------------------------

SAMPLING_RATE = 64000

# Minimum length (samples) a 1-D numeric array must have to be considered a
# vibration candidate when the 'vibration' key is missing. Prevents scalar
# metadata (N, L, H) or short helper arrays from being mistaken for the signal.
MIN_VIBRATION_SAMPLES = 100

# File ID -> (fault_type, damage_kind)
# Based on the PU (Paderborn University) bearing data center documentation.
#   K0xx  -> healthy bearings (accelerated lifetime test survivors)
#   KAxx  -> artificial damage (mostly outer race; KI01-KI03 artificial inner race)
#   KIxx  -> inner race damage (KI01-KI03 artificial, KI04+ real)
#   KOxx  -> outer race damage (real)
#   KBxx  -> ball damage (real)
PADERBORN_CATALOG: dict[str, tuple[str, str]] = {
    # Healthy (normal)
    "K001": ("normal", "healthy"),
    "K002": ("normal", "healthy"),
    "K003": ("normal", "healthy"),
    "K004": ("normal", "healthy"),
    "K005": ("normal", "healthy"),
    "K006": ("normal", "healthy"),
    # Artificial damage: outer race
    "KA04": ("outer_race", "artificial"),
    "KA05": ("outer_race", "artificial"),
    "KA15": ("outer_race", "artificial"),
    "KA16": ("outer_race", "artificial"),
    "KA22": ("outer_race", "artificial"),
    "KA30": ("outer_race", "artificial"),
    # Inner race damage (artificial + real)
    "KI01": ("inner_race", "artificial"),
    "KI03": ("inner_race", "artificial"),
    "KI04": ("inner_race", "real"),
    "KI05": ("inner_race", "artificial"),
    "KI07": ("inner_race", "artificial"),
    "KI08": ("inner_race", "artificial"),
    "KI14": ("inner_race", "real"),
    "KI16": ("inner_race", "real"),
    "KI17": ("inner_race", "real"),
    "KI18": ("inner_race", "real"),
    "KI21": ("inner_race", "real"),
    # Ball damage (real)
    "KB23": ("ball", "real"),
    "KB24": ("ball", "real"),
    "KB27": ("ball", "real"),
    # Outer race damage (real)
    "KO04": ("outer_race", "real"),
    "KO05": ("outer_race", "real"),
}

# damage_kind -> unified severity label
SEVERITY_BY_DAMAGE = {
    "healthy": "healthy",
    "artificial": "incipient",
    "real": "severe",
    "unknown": "unknown",
}

# Variables that may hold shaft speed inside the .mat (in priority order)
_RPM_KEYS = ("N", "n", "Speed", "speed", "rpm", "RPM")

# Variables that may hold load torque (Nm) inside the .mat
_TORQUE_KEYS = ("L", "Torque", "torque", "load", "M")

# Channel names (inside the struct 'Y' table) that carry raw acceleration.
_VIBRATION_CHANNEL_RE = re.compile(r"vibration", re.IGNORECASE)

# Max nesting depth when recursively searching nested structs for signal arrays.
_MAX_STRUCT_DEPTH = 8


def _unwrap(value: object) -> object:
    """Unwrap 0-d object/structured arrays returned by ``loadmat``.

    ``squeeze_me=True`` still wraps struct fields in 0-d object arrays in
    places; iterating them directly yields scalars instead of the payload.
    """
    while isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 0:
        value = value.item()
    return value


def _vibration_from_struct(value: object) -> np.ndarray | None:
    """Search a loaded PU struct for a channel named *vibration*.

    The official Paderborn .mat files store channels as a structured array in
    the ``Y`` field of a struct whose variable name equals the file stem.
    Channel entries expose ``Name`` and ``Data`` fields; the raw acceleration
    channel is ``vibration_1`` (and sometimes ``vibration_2``).

    Args:
        value: A loaded MATLAB variable (struct or array).

    Returns:
        The first matching vibration channel's data as a 1-D float64 array,
        or ``None`` if the value is not such a struct.
    """
    arr = np.asarray(value)
    if getattr(arr, "dtype", None) is None or arr.dtype.names is None:
        return None
    if "Y" not in arr.dtype.names:
        return None

    y_table = _unwrap(arr["Y"])
    y_arr = np.asarray(y_table)
    if getattr(y_arr, "dtype", None) is None or y_arr.dtype.names is None:
        return None
    if "Name" not in y_arr.dtype.names or "Data" not in y_arr.dtype.names:
        return None

    names = np.asarray(y_arr["Name"]).reshape(-1)
    data = y_arr["Data"]
    best: np.ndarray | None = None
    best_size = -1
    for name, channel in zip(names, data, strict=False):
        if not isinstance(name, str) or _VIBRATION_CHANNEL_RE.search(name) is None:
            continue
        channel_arr = np.asarray(channel).ravel()
        if channel_arr.size >= MIN_VIBRATION_SAMPLES and channel_arr.size > best_size:
            best = channel_arr
            best_size = channel_arr.size

    if best is None:
        return None
    return np.asarray(best, dtype=np.float64)


def _iter_numeric_arrays(value: object, depth: int = 0):
    """Yield candidate 1-D numeric arrays from a (possibly nested) MATLAB value.

    Recursively descends dicts, structs (field access), object arrays, and
    sequences, bounding depth to avoid pathological nesting. Yields arrays with
    at least ``MIN_VIBRATION_SAMPLES`` samples.
    """
    if depth > _MAX_STRUCT_DEPTH:
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_numeric_arrays(item, depth + 1)
        return
    value = _unwrap(value)
    arr = np.asarray(value)
    if getattr(arr, "dtype", None) is None:
        return
    if arr.dtype.names is not None:
        for field in arr.dtype.names:
            yield from _iter_numeric_arrays(arr[field], depth + 1)
    elif arr.dtype == object:
        for item in arr.ravel()[:64]:
            yield from _iter_numeric_arrays(item, depth + 1)
    elif arr.ndim >= 1:
        flat = arr.ravel()
        if flat.size >= MIN_VIBRATION_SAMPLES and np.issubdtype(flat.dtype, np.number):
            yield flat


def bearing_id_from_filename(stem: str) -> str | None:
    """Extract the bearing catalog ID from a Paderborn file name.

    Real PU files are named like ``N15_M07_F10_K001_4.mat`` (bearing ID
    ``K001``) or ``N15_M07_F10_KI04_1.mat`` (bearing ID ``KI04``). The unified
    catalog keys are ``K001``/``KI04``/``KA04``/etc.

    Args:
        stem: The file stem (e.g. ``"N15_M07_F10_K001_4"``).

    Returns:
        The catalog ID (e.g. ``"K001"``) or ``None`` when the name carries none.
    """
    match = re.search(r"(?:^|_)(K(?:A|I|O|B)?\d+)", stem)
    return match.group(1) if match else None


def rpm_from_filename(stem: str) -> float | None:
    """Parse shaft speed from the PU ``N<rpm/100>`` filename convention.

    ``N15`` -> 1500 rpm, ``N09`` -> 900 rpm.

    Args:
        stem: The file stem.

    Returns:
        RPM value, or ``None`` if the stem carries no speed marker.
    """
    match = re.search(r"N(\d{2})", stem)
    return float(int(match.group(1)) * 100) if match else None


def torque_from_filename(stem: str) -> float | None:
    """Parse load torque (Nm) from the PU ``M<torque*10>`` filename convention.

    ``M07`` -> 0.7 Nm, ``M01`` -> 0.1 Nm.

    Args:
        stem: The file stem.

    Returns:
        Torque in Nm, or ``None`` if the stem carries no load marker.
    """
    match = re.search(r"M(\d{2})", stem)
    return float(int(match.group(1)) / 10.0) if match else None


def parse_mat(filepath: Path) -> dict:
    """Load a Paderborn .mat file.

    Uses ``scipy.io.loadmat`` with ``squeeze_me=True`` and ``mat_dtype=True``
    so that ``(1, N)`` arrays collapse to ``(N,)`` and struct fields degrade
    gracefully. Returns the raw dict of variables with MATLAB metadata keys
    (``__header__``, ``__version__``, ``__globals__``) stripped.

    Args:
        filepath: Path to a Paderborn ``.mat`` file.

    Returns:
        Dict mapping variable name -> numpy array.

    Raises:
        ImportError: If scipy is not installed.
        FileNotFoundError: If ``filepath`` does not exist.
    """
    if sio is None:
        raise ImportError("scipy is required to read .mat files: pip install scipy")
    mat = sio.loadmat(str(filepath), squeeze_me=True, mat_dtype=True)
    return {key: value for key, value in mat.items() if not key.startswith("__")}


def extract_vibration(mat: dict) -> np.ndarray:
    """Extract the vibration signal from a Paderborn .mat file.

    Extraction order:
    1. Top-level ``vibration`` key (plain-variable .mat files).
    2. A channel named ``vibration*`` inside a PU struct's ``Y`` table
       (the official .mat distribution).
    3. The largest 1-D numeric array found anywhere in the file
       (length >= ``MIN_VIBRATION_SAMPLES``).

    Args:
        mat: Raw variable dict from :func:`parse_mat`.

    Returns:
        1-D float64 array of raw acceleration samples.

    Raises:
        ValueError: If no vibration array (or sufficiently large 1-D numeric
            array) can be found.
    """
    if "vibration" in mat:
        arr = np.asarray(mat["vibration"])
        if arr.ndim == 0:
            raise ValueError(
                "Paderborn .mat 'vibration' variable is a scalar; expected a signal array."
            )
        return np.asarray(arr.ravel(), dtype=np.float64)

    # Official PU struct layout: variable named after the file with a 'Y'
    # channel table containing vibration_1 / vibration_2.
    for value in mat.values():
        vib = _vibration_from_struct(value)
        if vib is not None:
            return vib

    # Robust fallback: deepest/largest numeric signal anywhere in the file.
    best: np.ndarray | None = None
    best_size = -1
    for candidate in _iter_numeric_arrays(mat):
        if candidate.size > best_size:
            best = candidate
            best_size = candidate.size

    if best is None:
        raise ValueError(
            "Could not find a vibration signal in Paderborn .mat: no 'vibration' "
            f"key, no vibration channel, and no 1-D numeric array with >= "
            f"{MIN_VIBRATION_SAMPLES} samples."
        )
    return np.asarray(best, dtype=np.float64)


def _speed_from_struct(mat: dict) -> float | None:
    """Read the mean of the ``speed`` channel from a PU struct, if present."""
    for value in mat.values():
        arr = np.asarray(value)
        if getattr(arr, "dtype", None) is None or arr.dtype.names is None:
            continue
        if "Y" not in arr.dtype.names:
            continue
        y_table = _unwrap(arr["Y"])
        y_arr = np.asarray(y_table)
        if getattr(y_arr, "dtype", None) is None or y_arr.dtype.names is None:
            continue
        if "Name" not in y_arr.dtype.names or "Data" not in y_arr.dtype.names:
            continue
        names = np.asarray(y_arr["Name"]).reshape(-1)
        data = y_arr["Data"]
        for name, channel in zip(names, data, strict=False):
            if isinstance(name, str) and name.lower() == "speed":
                vals = np.asarray(channel).ravel()
                if vals.size and np.all(np.isfinite(vals)):
                    return float(np.mean(vals))
    return None


def extract_rpm(mat: dict) -> float:
    """Extract shaft speed (RPM) from a Paderborn .mat file.

    Looks for the rotational speed variable ``N`` (1500 or 900 rpm in the
    published runs) with a small set of aliases, then falls back to the mean
    of the struct ``speed`` channel when present. Returns the first finite,
    positive value found.

    Args:
        mat: Raw variable dict from :func:`parse_mat`.

    Returns:
        Shaft speed in RPM. Defaults to ``1500.0`` when absent.
    """
    for key in _RPM_KEYS:
        if key not in mat:
            continue
        val = np.asarray(mat[key]).ravel()
        if val.size == 0:
            continue
        speed = float(val[0])
        if np.isfinite(speed) and speed > 0:
            return speed

    struct_speed = _speed_from_struct(mat)
    if struct_speed is not None and struct_speed > 0:
        return struct_speed

    return 1500.0


def extract_torque(mat: dict) -> float:
    """Extract load torque (Nm) from a Paderborn .mat file.

    Paderborn reports load torque (0.1 or 0.7 Nm in the published runs) rather
    than the CWRU horsepower convention. The value is stored in the unified
    ``load_hp`` column for schema compatibility; see :func:`normalize_file`.

    Args:
        mat: Raw variable dict from :func:`parse_mat`.

    Returns:
        Load torque in Nm. Defaults to ``0.0`` when absent.
    """
    for key in _TORQUE_KEYS:
        if key not in mat:
            continue
        val = np.asarray(mat[key]).ravel()
        if val.size == 0:
            continue
        torque = float(val[0])
        if np.isfinite(torque):
            return torque
    return 0.0


def normalize_file(filepath: Path, split: str = "test") -> pd.DataFrame:
    """Process a single Paderborn .mat file into a normalized DataFrame row.

    Uses :data:`PADERBORN_CATALOG` keyed by the bearing ID. The bearing ID is
    the file stem itself for plain files (``K001.mat``) or is parsed from the
    official PU naming convention (``N15_M07_F10_K001_4.mat`` -> ``K001``).
    Healthy bearings map to ``fault_diameter=0.0`` and damaged bearings to
    ``np.nan`` (unknown diameter, documented in the PU docs as such).
    ``split`` defaults to ``"test"`` because Paderborn is the target domain
    for the CWRU -> Paderborn domain shift study.

    Speed and torque are read from the filename convention first (``N15`` ->
    1500 rpm, ``M07`` -> 0.7 Nm), then from the .mat variables themselves.

    Args:
        filepath: Path to a Paderborn ``.mat`` file.
        split: Split label to assign (default ``"test"``).

    Returns:
        A single-row DataFrame with the unified AetherPdM schema.

    Raises:
        ValueError: If the vibration signal cannot be extracted.
    """
    file_id = filepath.stem
    mat = parse_mat(filepath)

    vibration = extract_vibration(mat)
    rpm = rpm_from_filename(file_id) or extract_rpm(mat)
    torque = torque_from_filename(file_id)
    if torque is None:
        torque = extract_torque(mat)

    catalog_key = file_id if file_id in PADERBORN_CATALOG else bearing_id_from_filename(file_id)
    catalog_key = catalog_key or file_id
    fault_type, damage_kind = PADERBORN_CATALOG.get(catalog_key, ("unknown", "unknown"))
    severity = SEVERITY_BY_DAMAGE.get(damage_kind, "unknown")
    fault_diameter = 0.0 if fault_type == "normal" else np.nan

    row = {
        "asset_id": f"paderborn-{file_id}",
        "file_id": file_id,
        "channel": "vibration",
        "sampling_rate": SAMPLING_RATE,
        "rpm": rpm,
        "load_hp": torque,  # Paderborn reports Nm torque; stored for schema parity
        "fault_type": fault_type,
        "fault_diameter": fault_diameter,
        "severity": severity,
        "split": split,
        "waveform": vibration.tolist(),
    }
    return pd.DataFrame([row])


def build_feature_parquet(input_dir: Path, output_dir: Path, split: str = "test") -> Path:
    """Normalize all Paderborn .mat files into a single Parquet dataset.

    Mirrors the CWRU ``build_feature_parquet`` pipeline. Individual file
    failures are logged and skipped so one malformed archive does not kill
    the batch.

    Args:
        input_dir: Directory containing extracted Paderborn ``.mat`` files.
        output_dir: Directory in which the normalized Parquet is written.
        split: Split label to assign to every row (default ``"test"``).

    Returns:
        Path to the written Parquet file.

    Raises:
        FileNotFoundError: If no ``.mat`` files exist under ``input_dir``.
        RuntimeError: If no file could be normalized.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    mat_files = sorted(input_dir.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "paderborn_normalized.parquet"

    all_records: list[pd.DataFrame] = []
    for fp in mat_files:
        try:
            all_records.append(normalize_file(fp, split=split))
        except Exception as e:  # noqa: BLE001 - batch pipeline reports and continues
            print(f"  SKIP {fp.name}: {e}")

    if not all_records:
        raise RuntimeError("No files were successfully normalized")

    result = pd.concat(all_records, ignore_index=True)
    result.to_parquet(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")
    print(f"  Fault distribution:\n{result['fault_type'].value_counts().to_string()}")
    print(f"  Split distribution:\n{result['split'].value_counts().to_string()}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Paderborn .mat to Parquet")
    parser.add_argument("--input", type=Path, default=Path("data/raw/paderborn"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/paderborn"))
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    build_feature_parquet(args.input, args.output, split=args.split)


if __name__ == "__main__":
    main()
