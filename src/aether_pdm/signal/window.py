"""
Signal windowing utilities.

Splits raw vibration waveforms into fixed-length windows with optional overlap.
"""

import numpy as np


def sliding_windows(
    signal: np.ndarray,
    window_size: int,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a 1D signal into overlapping windows.

    Args:
        signal: 1D array of vibration samples.
        window_size: Number of samples per window.
        overlap: Fraction of overlap between consecutive windows (0..1).

    Returns:
        windows: 2D array (n_windows, window_size).
        start_indices: 1D array of start indices for each window.
    """
    n = len(signal)
    if n < window_size:
        return np.empty((0, window_size), dtype=signal.dtype), np.array([], dtype=int)
    step = int(window_size * (1 - overlap))
    if step < 1:
        step = 1
    starts = np.arange(0, n - window_size + 1, step)
    windows = np.lib.stride_tricks.sliding_window_view(signal, window_size)[::step]
    return np.ascontiguousarray(windows), starts
