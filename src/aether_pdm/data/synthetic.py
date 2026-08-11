"""
Synthetic vibration data generator for testing and MLOps.

Generates realistic bearing vibration signals with controlled
fault types, severities, and degradation trajectories.

Usage:
    python -m aether_pdm.data.synthetic --output data/interim/synthetic
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_waveform(
    length: int = 4096,
    sampling_rate: float = 12000.0,
    rpm: float = 1772.0,
    fault_type: str = "normal",
    fault_diameter: float = 0.0,
    noise_level: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic vibration signal for a bearing.

    Returns a 1D numpy array with realistic frequency content.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length) / sampling_rate
    shaft_freq = rpm / 60.0

    # Fundamental shaft rotation + harmonics
    signal = 0.3 * np.sin(2 * np.pi * shaft_freq * t)
    signal += 0.15 * np.sin(2 * np.pi * 2 * shaft_freq * t)
    signal += 0.08 * np.sin(2 * np.pi * 3 * shaft_freq * t)

    if fault_type == "inner_race":
        bpfi = 9 * shaft_freq / 2 * (1 + 0.3126 / 1.537)
        amplitude = 0.4 + 0.8 * (fault_diameter / 0.021)
        signal += amplitude * np.sin(2 * np.pi * bpfi * t)
        signal += 0.3 * amplitude * np.sin(2 * np.pi * 2 * bpfi * t)
    elif fault_type == "outer_race":
        bpfo = 9 * shaft_freq / 2 * (1 - 0.3126 / 1.537)
        amplitude = 0.35 + 0.7 * (fault_diameter / 0.021)
        signal += amplitude * np.sin(2 * np.pi * bpfo * t)
        signal += 0.25 * amplitude * np.sin(2 * np.pi * 2 * bpfo * t)
    elif fault_type == "ball":
        bsf = 1.537 / 0.3126 * shaft_freq * (1 - (0.3126 / 1.537) ** 2)
        amplitude = 0.2 + 0.5 * (fault_diameter / 0.021)
        signal += amplitude * np.sin(2 * np.pi * bsf * t)
        signal += 0.2 * amplitude * np.sin(2 * np.pi * 2 * bsf * t)

    # Add random noise
    signal += rng.normal(0, noise_level, length)

    # Add occasional impulses (simulating bearing impacts)
    if fault_type != "normal":
        n_impulses = rng.integers(3, 8)
        for _ in range(n_impulses):
            idx = rng.integers(0, length)
            width = rng.integers(5, 15)
            if idx + width < length:
                signal[idx : idx + width] += 3.0 * fault_diameter / 0.021

    return signal


def degradation_ramp(
    n_points: int = 24,
    span_hours: float = 480.0,
    fault_type: str = "outer_race",
    fault_diameter_max: float = 0.0230,
    reference_diameter: float = 0.028,
    length: int = 4096,
    sampling_rate: float = 12000.0,
    rpm: float = 1772.0,
    noise_level: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic degradation trajectory for RUL demos.

    Emits one row per inspection point: a vibration waveform whose fault
    diameter grows linearly from 0 to ``fault_diameter_max`` across
    ``span_hours``. The degradation index is the waveform RMS normalized by
    the healthy baseline, anchored so the ramp starts at 0 (pristine) and
    reaches 1 at the ``reference_diameter`` (default 0.028, the dataset's
    maximum severity)::

        index_i = clip((rms_i / rms_baseline - ratio_0) / (ratio_ref - ratio_0), 0, 1)

    Columns: ``hours``, ``fault_diameter``, ``rms``, ``degradation_index``,
    ``health_score`` (health = 1 - index).

    NOTE: this is a *simulated* trajectory for demo/testing only — there is
    no run-to-failure ground truth behind it. RUL numbers derived from it
    must carry the honesty caveats documented in
    ``aether_pdm.models.rul`` and ``docs/model-cards/rul-v1.md``.
    """
    diameters = np.linspace(0.0, fault_diameter_max, n_points)
    hours = np.linspace(0.0, span_hours, n_points)

    def _rms(fault_diameter: float, wave_seed: int) -> float:
        waveform = synthetic_waveform(
            length=length,
            sampling_rate=sampling_rate,
            rpm=rpm,
            fault_type=fault_type,
            fault_diameter=fault_diameter,
            noise_level=noise_level,
            seed=wave_seed,
        )
        return float(np.sqrt(np.mean(np.square(waveform))))

    rms_baseline = _rms(0.0, seed)  # 'normal' waveform = pristine baseline
    rms_values = [_rms(float(d), seed + i) for i, d in enumerate(diameters)]
    rms_reference = _rms(reference_diameter, seed + n_points + 1)

    ratios = np.asarray(rms_values) / rms_baseline
    ratio_0 = ratios[0]
    ratio_ref = rms_reference / rms_baseline
    if ratio_ref - ratio_0 <= 1e-12:
        ratio_ref = ratio_0 + 1.0  # degenerate ramp -> index stays 0
    degradation_index = np.clip((ratios - ratio_0) / (ratio_ref - ratio_0), 0.0, 1.0)

    return pd.DataFrame(
        {
            "hours": hours,
            "fault_diameter": diameters,
            "rms": rms_values,
            "degradation_index": degradation_index,
            "health_score": 1.0 - degradation_index,
        }
    )


def generate_dataset(
    output_dir: Path,
    n_normal: int = 20,
    n_faulty: int = 30,
    length: int = 4096,
    seed: int = 42,
    n_val_normal: int = 3,
    n_val_faulty: int = 5,
    n_test_faulty: int = 5,
) -> Path:
    """Generate a full synthetic dataset with metadata and waveforms.

    Split assignment is deterministic and index-based (no extra RNG draws):
      - Normal rows: first ``n_val_normal`` -> ``"val"``, remainder -> ``"train"``.
      - Faulty rows: first ``n_test_faulty`` -> ``"test"``, next
        ``n_val_faulty`` -> ``"val"``, remainder -> ``"train"``.

    This guarantees a ``val`` split containing BOTH normal and faulty samples,
    which the promotion gate (``aether_pdm.ops.promote``, ``DEFAULT_SPLIT="val"``)
    requires for anomaly and fault candidate evaluation. Counts are clamped to
    the available row counts so tiny datasets degrade gracefully instead of
    producing empty splits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rows = []

    fault_types = ["inner_race", "outer_race", "ball"]
    severities = [0.007, 0.014, 0.021]
    loads = [0, 1, 2, 3]

    # Clamp so small datasets never produce empty/negative split ranges.
    n_val_normal = min(n_val_normal, n_normal)
    n_test_faulty = min(n_test_faulty, n_faulty)
    n_val_faulty = min(n_val_faulty, n_faulty - n_test_faulty)

    for i in range(n_normal):
        load = rng.choice(loads)
        rpm = {0: 1797, 1: 1772, 2: 1750, 3: 1730}[load]
        waveform = synthetic_waveform(
            length=length,
            rpm=float(rpm),
            fault_type="normal",
            seed=seed + i,
        )
        rows.append({
            "asset_id": f"synth-{i:04d}",
            "file_id": f"synth-normal-{i:04d}",
            "channel": "DE",
            "sampling_rate": 12000.0,
            "rpm": float(rpm),
            "load_hp": float(load),
            "fault_type": "normal",
            "fault_diameter": 0.0,
            "severity": "none",
            "split": "val" if i < n_val_normal else "train",
            "waveform": waveform.tolist(),
        })

    for i in range(n_faulty):
        ft = rng.choice(fault_types)
        sev = rng.choice(severities)
        load = rng.choice(loads)
        rpm = {0: 1797, 1: 1772, 2: 1750, 3: 1730}[load]
        waveform = synthetic_waveform(
            length=length,
            rpm=float(rpm),
            fault_type=ft,
            fault_diameter=sev,
            seed=seed + n_normal + i,
        )
        if i < n_test_faulty:
            split = "test"
        elif i < n_test_faulty + n_val_faulty:
            split = "val"
        else:
            split = "train"
        rows.append({
            "asset_id": f"synth-{n_normal + i:04d}",
            "file_id": f"synth-{ft}-{i:04d}",
            "channel": "DE",
            "sampling_rate": 12000.0,
            "rpm": float(rpm),
            "load_hp": float(load),
            "fault_type": ft,
            "fault_diameter": sev,
            "severity": "severe" if sev >= 0.021 else "moderate" if sev >= 0.014 else "incipient",
            "split": split,
            "waveform": waveform.tolist(),
        })

    df = pd.DataFrame(rows)
    output_path = output_dir / "synthetic_normalized.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Generated {len(df)} synthetic waveforms -> {output_path}")
    print(f"  Normal: {n_normal}, Faulty: {n_faulty}")
    print(f"  Distribution:\n{df['fault_type'].value_counts().to_string()}")
    print(f"  Split distribution:\n{df['split'].value_counts().to_string()}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic vibration dataset")
    parser.add_argument("--output", type=Path, default=Path("data/interim/synthetic"))
    parser.add_argument("--n-normal", type=int, default=20)
    parser.add_argument("--n-faulty", type=int, default=30)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-val-normal", type=int, default=3)
    parser.add_argument("--n-val-faulty", type=int, default=5)
    parser.add_argument("--n-test-faulty", type=int, default=5)
    args = parser.parse_args()

    generate_dataset(
        args.output,
        args.n_normal,
        args.n_faulty,
        args.length,
        args.seed,
        args.n_val_normal,
        args.n_val_faulty,
        args.n_test_faulty,
    )


if __name__ == "__main__":
    main()
