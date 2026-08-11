"""Tests for the degradation-trend RUL estimator (honest scope).

Fast tests: only numpy/scipy/pytest + the local modules — no heavy imports.
"""

import numpy as np
import pytest

from aether_pdm.data.synthetic import degradation_ramp
from aether_pdm.models.rul import (
    DISCLAIMER,
    DegradationTrendRUL,
    estimate_rul_from_scores,
)


def _fit(x: np.ndarray, y: np.ndarray, **kwargs):
    est = DegradationTrendRUL(**kwargs)
    summary = est.fit(x, y)
    return est, summary


def test_degradation_index_from_scores():
    """Healthy scores -> 0 degradation, failed scores -> 1, clipped outside [0,1]."""
    est = DegradationTrendRUL()
    idx = est.degradation_index_from_scores(np.array([1.0, 0.9, 0.5, 0.0, -0.5, 1.5]))
    np.testing.assert_allclose(idx, [0.0, 0.1, 0.5, 1.0, 1.0, 0.0])


def test_fit_returns_slope_intercept_r2():
    """A clean linear ramp should produce slope ~0.1, intercept ~0, R^2 ~1."""
    x = np.arange(10, dtype=np.float64)
    y = 0.1 * x
    _, summary = _fit(x, y)
    assert summary["slope"] == pytest.approx(0.1, abs=1e-9)
    assert summary["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert summary["r_squared"] > 0.99
    assert summary["n_points"] == 10


def test_predict_rul_positive_slope():
    """Degrading series -> positive RUL, tight CI band around the extrapolation."""
    x = np.arange(20, dtype=np.float64)
    y = 0.05 * x
    y[3] += 0.02  # tiny deterministic wiggle so the CI is non-degenerate
    est, _ = _fit(x, y)
    pred = est.predict_remaining_life(hours_elapsed=10.0, current_index=0.5)
    # RUL = (0.9 - 0.5) / slope ~= 8.0 hours
    assert pred["status"] == "ok"
    assert pred["rul_hours"] == pytest.approx(8.0, abs=0.05)
    assert pred["ci_low_hours"] < pred["rul_hours"] < pred["ci_high_hours"]
    assert pred["confidence"] == "medium"


def test_predict_rul_no_trend():
    """Flat (non-degrading) series -> RUL None with honest status."""
    x = np.arange(10, dtype=np.float64)
    y = np.full(10, 0.3)
    est, summary = _fit(x, y)
    assert summary["slope"] <= 0.0
    pred = est.predict_remaining_life(hours_elapsed=9.0, current_index=0.3)
    assert pred["rul_hours"] is None
    assert pred["status"] == "no_detectable_degradation_trend"
    assert pred["confidence"] is None


def test_predict_rul_insufficient_points():
    """Fewer than min_points -> RUL None with 'insufficient data' status."""
    history = [
        {"timestamp": float(t), "health_score": float(h)}
        for t, h in zip((0.0, 1.0, 2.0), (0.95, 0.9, 0.85))
    ]
    result = estimate_rul_from_scores("motor-1", history, now_hours=2.0, min_points=5)
    assert result["rul_hours"] is None
    assert result["status"] == "insufficient_data"
    assert "Insufficient data" in result["reason"]

    with pytest.raises(ValueError, match="min_points"):
        DegradationTrendRUL(min_points=5).fit(np.arange(3.0), np.arange(3.0) * 0.1)


def test_low_r2_flagged():
    """A noisy series still reports a number but flags confidence 'low'."""
    rng = np.random.default_rng(7)
    x = np.arange(20, dtype=np.float64)
    y = 0.05 * x + rng.normal(0.0, 0.4, 20)
    est, summary = _fit(x, y)
    assert summary["r_squared"] < 0.5
    pred = est.predict_remaining_life(hours_elapsed=19.0, current_index=0.6)
    assert pred["status"] == "ok"
    assert pred["rul_hours"] is not None and pred["rul_hours"] > 0
    assert pred["confidence"] == "low"


def test_synthetic_demo_ramp():
    """Degradation ramp via synthetic_waveform -> positive RUL, sane magnitude."""
    ramp = degradation_ramp(n_points=24, span_hours=480.0, seed=42)
    history = [
        {"timestamp": float(row.hours), "health_score": float(row.health_score)}
        for row in ramp.itertuples()
    ]
    result = estimate_rul_from_scores(
        "demo-motor", history, now_hours=float(ramp["hours"].iloc[-1])
    )
    assert result["status"] == "ok"
    assert result["rul_hours"] is not None and result["rul_hours"] > 0
    assert result["rul_hours"] < 480.0  # sane: shorter than the ramp span
    assert result["r_squared"] > 0.9  # near-linear synthetic ramp
    assert result["confidence"] == "medium"
    assert DISCLAIMER in result["disclaimer"]


def test_estimate_rul_sorts_and_drops_nan():
    """estimate_rul_from_scores sorts by timestamp and drops dropout rows."""
    history = [
        {"timestamp": 3.0, "health_score": 0.85},
        {"timestamp": 1.0, "health_score": 0.95},
        {"timestamp": 2.0, "health_score": float("nan")},  # sensor dropout
        {"timestamp": 4.0, "health_score": 0.75},
        {"timestamp": 5.0, "health_score": 0.7},
        {"timestamp": 6.0, "health_score": 0.65},
        {"timestamp": 7.0, "health_score": 0.6},
    ]
    result = estimate_rul_from_scores("motor-2", history, now_hours=7.0)
    assert result["status"] == "ok"
    assert result["n_points"] == 6
    assert result["rul_hours"] is not None and result["rul_hours"] > 0


def test_estimate_rul_empty_history():
    """Empty history -> RUL None with 'insufficient data' status."""
    result = estimate_rul_from_scores("motor-3", [], now_hours=0.0)
    assert result["rul_hours"] is None
    assert result["status"] == "insufficient_data"
