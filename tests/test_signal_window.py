"""Tests for signal windowing."""

import numpy as np

from aether_pdm.signal.window import sliding_windows


def test_sliding_windows_no_overlap():
    """Non-overlapping windows should partition the signal exactly."""
    signal = np.arange(100, dtype=float)
    windows, starts = sliding_windows(signal, window_size=20, overlap=0.0)
    assert windows.shape == (5, 20)
    assert list(starts) == [0, 20, 40, 60, 80]


def test_sliding_windows_overlap_50():
    """50% overlap should produce ~2x windows vs no overlap."""
    signal = np.arange(200, dtype=float)
    w_no_overlap = sliding_windows(signal, window_size=40, overlap=0.0)[0]
    w_overlap = sliding_windows(signal, window_size=40, overlap=0.5)[0]
    assert len(w_overlap) == 2 * len(w_no_overlap) - 1


def test_sliding_windows_window_content():
    """Each window should contain the correct signal slice."""
    signal = np.sin(np.linspace(0, 4 * np.pi, 100))
    windows, starts = sliding_windows(signal, window_size=20, overlap=0.5)
    for i, start in enumerate(starts):
        np.testing.assert_array_almost_equal(windows[i], signal[start : start + 20])


def test_sliding_windows_short_signal():
    """Signal shorter than window should return no windows."""
    signal = np.array([1.0, 2.0, 3.0])
    windows, starts = sliding_windows(signal, window_size=10, overlap=0.5)
    assert windows.shape[0] == 0
    assert len(starts) == 0
