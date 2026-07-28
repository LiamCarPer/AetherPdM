"""
CWRU .mat to normalized Parquet pipeline.

Unified schema:
  asset_id, file_id, channel, sampling_rate, rpm, load_hp,
  fault_type, fault_diameter, severity, split, waveform

Usage:
    python -m aether_pdm.ingest.normalize_cwru [--input data/raw/cwru] [--output data/interim/cwru]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scipy.io as sio
except ImportError:
    sio = None  # type: ignore


# ---------------------------------------------------------------------------
# CWRU file → label mapping
# Format: (file_pattern, fault_type, fault_diameter_inch, location)
# Source: CWRU Bearing Data Center documentation
# ---------------------------------------------------------------------------

CWRU_CATALOG: list[tuple[str, str, float, str]] = [
    # Normal baseline
    ("97", "normal", 0.0, "DE"),
    ("98", "normal", 0.0, "DE"),
    ("99", "normal", 0.0, "DE"),
    ("100", "normal", 0.0, "DE"),
    # Inner race fault — drive end
    ("105", "inner_race", 0.007, "DE"),
    ("106", "inner_race", 0.007, "DE"),
    ("107", "inner_race", 0.007, "DE"),
    ("108", "inner_race", 0.007, "DE"),
    ("109", "inner_race", 0.014, "DE"),
    ("110", "inner_race", 0.014, "DE"),
    ("111", "inner_race", 0.014, "DE"),
    ("112", "inner_race", 0.014, "DE"),
    ("113", "inner_race", 0.021, "DE"),
    ("114", "inner_race", 0.021, "DE"),
    ("115", "inner_race", 0.021, "DE"),
    ("116", "inner_race", 0.021, "DE"),
    # Ball fault — drive end
    ("117", "ball", 0.007, "DE"),
    ("118", "ball", 0.007, "DE"),
    ("119", "ball", 0.007, "DE"),
    ("120", "ball", 0.007, "DE"),
    ("121", "ball", 0.014, "DE"),
    ("122", "ball", 0.014, "DE"),
    ("123", "ball", 0.014, "DE"),
    ("124", "ball", 0.014, "DE"),
    ("125", "ball", 0.021, "DE"),
    ("126", "ball", 0.021, "DE"),
    ("127", "ball", 0.021, "DE"),
    ("128", "ball", 0.021, "DE"),
    # Outer race fault — drive end (centered @ 6:00)
    ("129", "outer_race", 0.007, "DE"),
    ("130", "outer_race", 0.007, "DE"),
    ("131", "outer_race", 0.007, "DE"),
    ("132", "outer_race", 0.007, "DE"),
    ("133", "outer_race", 0.014, "DE"),
    ("134", "outer_race", 0.014, "DE"),
    ("135", "outer_race", 0.014, "DE"),
    ("136", "outer_race", 0.014, "DE"),
    ("137", "outer_race", 0.021, "DE"),
    ("138", "outer_race", 0.021, "DE"),
    ("139", "outer_race", 0.021, "DE"),
    ("140", "outer_race", 0.021, "DE"),
    # Outer race fault — orthogonal positions @ 3:00 and 12:00
    ("144", "outer_race", 0.007, "DE"),
    ("145", "outer_race", 0.007, "DE"),
    ("146", "outer_race", 0.007, "DE"),
    ("147", "outer_race", 0.007, "DE"),
    ("148", "outer_race", 0.014, "DE"),
    ("149", "outer_race", 0.014, "DE"),
    ("150", "outer_race", 0.014, "DE"),
    ("151", "outer_race", 0.014, "DE"),
    ("152", "outer_race", 0.021, "DE"),
    ("153", "outer_race", 0.021, "DE"),
    ("154", "outer_race", 0.021, "DE"),
    ("155", "outer_race", 0.021, "DE"),
]

# Load → RPM mapping for CWRU drive-end bearing
LOAD_RPM: dict[float, float] = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# Channels to extract
CHANNELS = {
    "DE": ("_DE_time", 12000),
    "FE": ("_FE_time", 12000),
    "BA": ("_BA_time", 12000),
}

# Anti-leakage: hold out specific files for test
# These are the most severe faults — harder to generalize to
TEST_FILES: set[str] = {"125", "126", "127", "128"}


def _build_catalog() -> dict[str, tuple[str, float, str]]:
    """Build lookup: file_id → (fault_type, fault_diameter, location)."""
    lookup: dict[str, tuple[str, float, str]] = {}
    for fid, ftype, diam, loc in CWRU_CATALOG:
        lookup[fid] = (ftype, diam, loc)
    return lookup


def parse_mat(filepath: Path) -> dict[str, np.ndarray]:
    """Load a CWRU .mat file and return its variables."""
    if sio is None:
        raise ImportError("scipy is required to read .mat files: pip install scipy")
    try:
        mat = sio.loadmat(str(filepath))
    except NotImplementedError:
        mat = sio.loadmat(str(filepath), mat_dtype=True)
    return mat


def determine_split(file_id: str) -> str:
    """Assign split based on file_id to prevent leakage."""
    if file_id in TEST_FILES:
        return "test"
    return "train"


def determine_load_hp(mat: dict, file_id: str) -> float:
    """Determine load in HP from .mat file metadata."""
    # CWRU encodes load in the variable name pattern or as a separate variable
    for key in mat:
        if key.endswith("RPM"):
            rpm_val = float(mat[key].ravel()[0])
            for hp, rpm in LOAD_RPM.items():
                if abs(rpm_val - rpm) < 10:
                    return hp
    return 0.0


def normalize_file(filepath: Path) -> pd.DataFrame:
    """Process a single CWRU .mat file into a normalized DataFrame row."""
    catalog = _build_catalog()
    file_id = filepath.stem
    mat = parse_mat(filepath)

    fault_type, fault_diameter, location = catalog.get(
        file_id, ("unknown", 0.0, "unknown")
    )
    split = determine_split(file_id)
    load_hp = determine_load_hp(mat, file_id)
    rpm = LOAD_RPM.get(load_hp, 0.0)

    rows = []
    for channel, (suffix, fs) in CHANNELS.items():
        var_name = f"{file_id}{suffix}" if file_id.isdigit() else f"X{file_id}{suffix}"
        if var_name in mat:
            waveform = mat[var_name].ravel()
            rows.append(
                {
                    "asset_id": f"cwru-{file_id}",
                    "file_id": file_id,
                    "channel": channel,
                    "sampling_rate": fs,
                    "rpm": rpm,
                    "load_hp": load_hp,
                    "fault_type": fault_type,
                    "fault_diameter": fault_diameter,
                    "severity": (
                        "severe" if fault_diameter >= 0.021
                        else "moderate" if fault_diameter >= 0.014
                        else "incipient"
                    ),
                    "split": split,
                    "waveform": waveform.tolist(),
                }
            )
    return pd.DataFrame(rows)


def build_feature_parquet(input_dir: Path, output_dir: Path) -> Path:
    """Normalize all CWRU .mat files into a single Parquet dataset."""
    mat_files = sorted(input_dir.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cwru_normalized.parquet"

    all_records: list[pd.DataFrame] = []
    for fp in mat_files:
        try:
            df = normalize_file(fp)
            all_records.append(df)
        except Exception as e:
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
    parser = argparse.ArgumentParser(description="Normalize CWRU .mat to Parquet")
    parser.add_argument("--input", type=Path, default=Path("data/raw/cwru"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/cwru"))
    args = parser.parse_args()

    build_feature_parquet(args.input, args.output)


if __name__ == "__main__":
    main()
