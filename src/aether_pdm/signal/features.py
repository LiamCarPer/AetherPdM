"""
Time-domain and frequency-domain feature computation for bearing diagnostics.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert


def time_features(signal: np.ndarray) -> dict[str, float]:
    """Compute time-domain features from a 1D vibration window."""
    rms_val = np.sqrt(np.mean(signal**2))
    peak_val = np.max(np.abs(signal))
    crest_val = peak_val / rms_val if rms_val > 0 else 0.0
    kurtosis_val = float(np.mean((signal - np.mean(signal)) ** 4) / (np.std(signal) ** 4 + 1e-12))
    skew_val = float(np.mean((signal - np.mean(signal)) ** 3) / (np.std(signal) ** 3 + 1e-12))
    return {
        "rms": float(rms_val),
        "peak": float(peak_val),
        "crest": float(crest_val),
        "kurtosis": float(kurtosis_val),
        "skew": float(skew_val),
    }


def envelope_spectrum(signal: np.ndarray) -> np.ndarray:
    """Compute the envelope spectrum via Hilbert transform."""
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    spectrum = np.abs(rfft(envelope))
    return spectrum


def frequency_features(
    signal: np.ndarray,
    sampling_rate: float,
    bands: list[tuple[float, float]] | None = None,
) -> dict[str, float]:
    """
    Compute frequency-domain features:
    - FFT peak magnitudes in each band
    - Total energy in predefined frequency bands
    """
    n = len(signal)
    fft_vals = np.abs(rfft(signal))
    fft_freqs = rfftfreq(n, d=1.0 / sampling_rate)

    if bands is None:
        bands = [
            (0, 500),
            (500, 1000),
            (1000, 2000),
            (2000, 4000),
            (4000, sampling_rate / 2),
        ]

    features = {}
    for low, high in bands:
        mask = (fft_freqs >= low) & (fft_freqs < high)
        band_energy = float(np.sum(fft_vals[mask] ** 2))
        band_peak = float(np.max(fft_vals[mask])) if np.any(mask) else 0.0
        features[f"band_power_{low}_{high}"] = band_energy
        features[f"fft_peak_{low}_{high}"] = band_peak

    return features


def bearing_fault_frequencies(
    rpm: float,
    n_balls: int = 9,
    ball_diameter: float = 0.3126,
    pitch_diameter: float = 1.537,
    contact_angle: float = 0.0,
) -> dict[str, float]:
    """
    Calculate theoretical bearing fault frequencies.

    Defaults match SKF 6205 bearing (CWRU drive-end).
    """
    fr = rpm / 60.0
    cos_angle = np.cos(np.deg2rad(contact_angle))
    bpfo = n_balls * fr / 2 * (1 - ball_diameter / pitch_diameter * cos_angle)
    bpfi = n_balls * fr / 2 * (1 + ball_diameter / pitch_diameter * cos_angle)
    bsf = pitch_diameter / ball_diameter * fr * (1 - (ball_diameter / pitch_diameter * cos_angle) ** 2)
    ftf = fr / 2 * (1 - ball_diameter / pitch_diameter * cos_angle)
    return {
        "bpfo": float(bpfo),
        "bpfi": float(bpfi),
        "bsf": float(bsf),
        "ftf": float(ftf),
    }


def compute_all_features(
    signal: np.ndarray,
    sampling_rate: float,
    rpm: float | None = None,
    bands: list[tuple[float, float]] | None = None,
) -> dict[str, float]:
    """
    Compute the full feature vector for a vibration window.

    Combines time-domain, frequency-domain, and envelope features.
    """
    feats = {}
    feats.update(time_features(signal))
    feats.update(frequency_features(signal, sampling_rate, bands))

    env_spec = envelope_spectrum(signal)
    env_peak = float(np.max(env_spec))
    env_energy = float(np.sum(env_spec**2))
    feats["envelope_peak"] = env_peak
    feats["envelope_energy"] = env_energy

    if rpm is not None:
        bff = bearing_fault_frequencies(rpm)
        # Map envelope peaks near bearing fault frequencies
        fft_freqs = rfftfreq(len(signal), d=1.0 / sampling_rate)
        for name, freq in bff.items():
            idx = np.argmin(np.abs(fft_freqs - freq))
            feats[f"{name}_amp"] = float(env_spec[idx])
            # 2nd and 3rd harmonics
            for h in [2, 3]:
                idx_h = np.argmin(np.abs(fft_freqs - freq * h))
                feats[f"{name}_h{h}_amp"] = float(env_spec[idx_h])

    return feats
