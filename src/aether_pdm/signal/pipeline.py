"""
Signal processing pipeline: raw waveform to windowed features.

Orchestrates windowing + feature extraction to produce feature rows
that can be saved as Parquet or fed directly to models.

Usage:
    python -m aether_pdm.signal.pipeline --input data/interim/cwru/cwru_normalized.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aether_pdm.signal.features import compute_all_features
from aether_pdm.signal.window import sliding_windows

FEATURE_VERSION = "v2"


def process_waveform(
    waveform: np.ndarray,
    sampling_rate: float,
    rpm: float | None = None,
    window_size: int = 2048,
    overlap: float = 0.5,
) -> pd.DataFrame:
    """
    Split a raw waveform into windows and compute features for each.

    Returns a DataFrame with one row per window.
    """
    windows, starts = sliding_windows(waveform, window_size, overlap)
    if windows.shape[0] == 0:
        return pd.DataFrame()

    rows = []
    for i in range(windows.shape[0]):
        feats = compute_all_features(windows[i], sampling_rate, rpm)
        feats["window_id"] = i
        feats["window_start"] = int(starts[i])
        feats["window_end"] = int(starts[i] + window_size)
        rows.append(feats)

    return pd.DataFrame(rows)


def process_dataset(
    input_path: Path,
    output_dir: Path | None = None,
    window_size: int = 2048,
    overlap: float = 0.5,
    max_waveforms: int | None = None,
) -> pd.DataFrame:
    """
    Process a normalized Parquet dataset through the signal pipeline.

    Each row in the input contains a raw waveform; this extracts
    sliding windows and computes the full feature vector.
    """
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} records from {input_path}")

    all_windows: list[pd.DataFrame] = []
    count = 0

    for idx, row in df.iterrows():
        if max_waveforms is not None and count >= max_waveforms:
            break

        waveform = np.array(row["waveform"], dtype=float)
        fs = float(row["sampling_rate"])
        rpm_val = float(row["rpm"]) if pd.notna(row.get("rpm")) else None

        windows_df = process_waveform(waveform, fs, rpm_val, window_size, overlap)
        if windows_df.empty:
            continue

        # Carry over metadata
        meta_cols = ["asset_id", "file_id", "channel", "fault_type",
                      "fault_diameter", "severity", "split", "load_hp"]
        for col in meta_cols:
            if col in row:
                windows_df[col] = row[col]

        windows_df["feature_version"] = FEATURE_VERSION
        all_windows.append(windows_df)
        count += 1

    if not all_windows:
        raise RuntimeError("No features generated — check input data")

    result = pd.concat(all_windows, ignore_index=True)
    print(f"Generated {len(result)} feature windows from {count} waveforms")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"features_{FEATURE_VERSION}.parquet"
        result.to_parquet(output_path, index=False)
        print(f"Wrote features to {output_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal pipeline: waveform to features")
    parser.add_argument("--input", type=Path, required=True, help="Input normalized Parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim/features"))
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max-waveforms", type=int, default=None)
    args = parser.parse_args()

    process_dataset(
        args.input,
        args.output_dir,
        window_size=args.window_size,
        overlap=args.overlap,
        max_waveforms=args.max_waveforms,
    )


if __name__ == "__main__":
    main()
