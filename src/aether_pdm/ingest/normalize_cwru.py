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
# Source: CWRU Bearing Data Center 12k Drive End page
#   https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
#
# The host renumbered the 12k DE files (2024+). IDs that now return HTTP 404
# were remapped to their official replacements below (same fault_type,
# fault_diameter, location). Legacy IDs whose content could not be verified
# against the official page (e.g. 135/136, shuffled loads) were dropped in
# favor of the clean official groups.
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
    # 113-116 (IR021) are 404 -> official replacement group 209-212
    ("209", "inner_race", 0.021, "DE"),
    ("210", "inner_race", 0.021, "DE"),
    ("211", "inner_race", 0.021, "DE"),
    ("212", "inner_race", 0.021, "DE"),
    # Ball fault — drive end
    # 117 (B007) is 404; official B007 group is 118-121, but 121 is already
    # catalogued as ball 0.014 (it is alive), so 117 is dropped, not remapped.
    ("118", "ball", 0.007, "DE"),
    ("119", "ball", 0.007, "DE"),
    ("120", "ball", 0.007, "DE"),
    ("121", "ball", 0.014, "DE"),
    ("122", "ball", 0.014, "DE"),
    ("123", "ball", 0.014, "DE"),
    ("124", "ball", 0.014, "DE"),
    # 125-128 (B021) -> official replacement group 222-225
    ("222", "ball", 0.021, "DE"),
    ("223", "ball", 0.021, "DE"),
    ("224", "ball", 0.021, "DE"),
    ("225", "ball", 0.021, "DE"),
    # Outer race fault — drive end (centered @ 6:00)
    # 129 (OR007@6) is 404; official OR007@6 group is 130-133, but 133 is not
    # re-catalogued (it was a legacy OR014 file), so 129 is dropped.
    ("130", "outer_race", 0.007, "DE"),
    ("131", "outer_race", 0.007, "DE"),
    ("132", "outer_race", 0.007, "DE"),
    # 133-136 (OR014@6) -> official replacement group 197-200
    ("197", "outer_race", 0.014, "DE"),
    ("198", "outer_race", 0.014, "DE"),
    ("199", "outer_race", 0.014, "DE"),
    ("200", "outer_race", 0.014, "DE"),
    # 139-140 (OR021@6) are 404 -> official replacements 234-235 (137/138 kept)
    ("137", "outer_race", 0.021, "DE"),
    ("138", "outer_race", 0.021, "DE"),
    ("234", "outer_race", 0.021, "DE"),
    ("235", "outer_race", 0.021, "DE"),
    # Outer race fault — orthogonal positions @ 3:00 and 12:00
    ("144", "outer_race", 0.007, "DE"),
    ("145", "outer_race", 0.007, "DE"),
    ("146", "outer_race", 0.007, "DE"),
    ("147", "outer_race", 0.007, "DE"),
    ("148", "outer_race", 0.014, "DE"),
    ("149", "outer_race", 0.014, "DE"),
    ("150", "outer_race", 0.014, "DE"),
    ("151", "outer_race", 0.014, "DE"),
    # 152-155 (OR021@3) are 404 -> official replacement group 246-249
    ("246", "outer_race", 0.021, "DE"),
    ("247", "outer_race", 0.021, "DE"),
    ("248", "outer_race", 0.021, "DE"),
    ("249", "outer_race", 0.021, "DE"),
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
# These are the most severe faults (ball 0.021", current official group) —
# harder to generalize to.
TEST_FILES: set[str] = {"222", "223", "224", "225"}

# File-level validation split (disjoint from TEST_FILES).
# Severity-representative by design: each fault type spans the severity axis
# (incipient 0.007 / moderate 0.014 / severe 0.021) so the val set measures
# fault-TYPE discrimination across severities, not a severity corner case
# (ops/promote.py evaluates DEFAULT_SPLIT="val").
#   "97","98"   -> normal               (0.000, 0.000)
#   "105","209" -> inner_race           (0.007, 0.021)
#   "118","121" -> ball                 (0.007, 0.014; 0.021 held out in TEST)
#   "130","234" -> outer_race           (0.007, 0.021)
VAL_FILES: set[str] = {"97", "98", "105", "209", "118", "121", "130", "234"}


def _validate_split_sets() -> None:
    """Fail fast if a split set references an unknown catalog id or overlaps."""
    catalog_ids = {fid for fid, _, _, _ in CWRU_CATALOG}
    unknown = (TEST_FILES | VAL_FILES) - catalog_ids
    if unknown:
        raise ValueError(
            f"Split sets reference unknown CWRU catalog ids: {sorted(unknown)}"
        )
    overlap = TEST_FILES & VAL_FILES
    if overlap:
        raise ValueError(
            f"TEST_FILES and VAL_FILES must be disjoint, got {sorted(overlap)}"
        )


_validate_split_sets()


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
    if file_id in VAL_FILES:
        return "val"
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


def _channel_variable_names(file_id: str, suffix: str) -> list[str]:
    """Candidate CWRU channel variable names for ``file_id``, canonical first.

    Official CWRU .mat files zero-pad the numeric ID to three digits and
    prefix it with ``X`` (file ``97.mat`` contains ``X097_DE_time``). Some
    mirrors repackage files without the padding or the prefix, so fall back
    through the unpadded forms too.

    Args:
        file_id: Numeric catalog ID (e.g. ``"97"``).
        suffix: Channel suffix (e.g. ``"_DE_time"``).

    Returns:
        Candidate variable names, most canonical first.
    """
    if not file_id.isdigit():
        return [f"X{file_id}{suffix}", f"{file_id}{suffix}"]
    padded = file_id.zfill(3)
    return [
        f"X{padded}{suffix}",
        f"X{file_id}{suffix}",
        f"{file_id}{suffix}",
    ]


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
        waveform: np.ndarray | None = None
        for var_name in _channel_variable_names(file_id, suffix):
            if var_name in mat:
                waveform = mat[var_name].ravel()
                break
        if waveform is not None:
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
    """Normalize all CWRU .mat files into a single Parquet dataset.

    Raises:
        RuntimeError: If no .mat files exist under ``input_dir`` or if none
            could be normalized (including the case where files exist but
            yield zero rows, e.g. non-numeric stems that do not match
            ``CWRU_CATALOG``). The failure mode is always an actionable
            message, never a ``KeyError`` on an empty result.
    """
    mat_files = sorted(input_dir.rglob("*.mat"))
    if not mat_files:
        raise RuntimeError(
            f"No files were successfully normalized under {input_dir}: "
            "no .mat files found. Run the downloader first: "
            "`uv run python -m aether_pdm.ingest.download_cwru`"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cwru_normalized.parquet"

    all_records: list[pd.DataFrame] = []
    for fp in mat_files:
        try:
            df = normalize_file(fp)
        except Exception as e:
            print(f"  SKIP {fp.name}: {e}")
            continue
        if df.empty:
            print(f"  SKIP {fp.name}: no catalog/channel match produced 0 rows")
            continue
        all_records.append(df)

    if not all_records:
        raise RuntimeError(
            f"No files were successfully normalized under {input_dir}. "
            "Downloaded .mat files must use the numeric file IDs from "
            "CWRU_CATALOG (e.g. '97.mat', '105.mat') so that `filepath.stem` "
            "maps to the catalog. Re-run "
            "`uv run python -m aether_pdm.ingest.download_cwru --file-ids ...` "
            "and try again."
        )

    result = pd.concat(all_records, ignore_index=True)
    if result.empty:  # belt-and-braces: never KeyError on the distribution print
        raise RuntimeError(
            f"No files were successfully normalized under {input_dir}: "
            "concatenated result is empty."
        )
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
