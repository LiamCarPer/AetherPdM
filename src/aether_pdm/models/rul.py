"""
Degradation-trend remaining-useful-life (RUL) estimation.

HONEST SCOPE
------------
This module does **not** predict calibrated time-to-failure. The AetherPdM
datasets (CWRU, Paderborn, synthetic) carry no run-to-failure ground truth,
so a "RUL in hours" number can never be validated here. Instead we fit a
linear trend of a degradation index (0 = pristine, 1 = failure) over
elapsed hours and extrapolate that trend to a user-supplied failure
threshold:

    RUL = (failure_threshold - current_index) / slope

The extrapolation is reported with explicit caveats:

* ``RUL = None`` when the trend is flat or improving (slope <= 0) —
  status ``"no_detectable_degradation_trend"``.
* ``RUL = 0.0`` when the current index already sits at/above the failure
  threshold — status ``"failure_threshold_reached"``.
* A 95% confidence band on the extrapolated time-to-threshold, derived from
  the slope standard error of :func:`scipy.stats.linregress`.
* A confidence flag that drops to ``"low"`` when R² < 0.5 (or when the CI
  is unbounded). The flag is never ``"high"``: even a perfect in-sample fit
  is an unvalidated extrapolation.
* A standing disclaimer (``DISCLAIMER``) echoed in every result dict:
  this is a degradation-trend extrapolation, not a calibrated
  time-to-failure model. Validation on run-to-failure datasets
  (NASA IMS, XJTU-SY) is future work.

Public API
----------
* :class:`DegradationTrendRUL` — stateful fit + predict estimator.
* :func:`estimate_rul_from_scores` — convenience wrapper over a persisted
  health-score history (e.g. rows from ``score_records``).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "Degradation-trend extrapolation, NOT a calibrated time-to-failure model. "
    "Validation on run-to-failure datasets (NASA IMS / XJTU-SY) is future work."
)

_STATUS_OK = "ok"
_STATUS_INSUFFICIENT = "insufficient_data"
_STATUS_NO_TREND = "no_detectable_degradation_trend"
_STATUS_AT_THRESHOLD = "failure_threshold_reached"

_LOW_R2 = 0.5  # R² below this flags the trend fit as "low confidence"
_MIN_POINTS_FOR_MEDIUM_CONFIDENCE = 6  # 5 points can overfit a line


class DegradationTrendRUL:
    """Linear degradation-trend RUL estimator (honest scope, see module docstring).

    Parameters
    ----------
    failure_threshold : float
        Degradation index (0-1) at which the asset is considered failed.
        Default 0.9.
    min_points : int
        Minimum number of (hours, index) observations required for a trend
        fit. Fewer observations → ``fit`` raises and the convenience wrapper
        reports status ``"insufficient_data"``. Default 5.
    """

    def __init__(self, failure_threshold: float = 0.9, min_points: int = 5) -> None:
        if not 0.0 < failure_threshold <= 1.0:
            raise ValueError(
                f"failure_threshold must be in (0, 1], got {failure_threshold}"
            )
        if min_points < 2:
            raise ValueError(
                f"min_points must be >= 2 (a linear fit needs at least 2 points), got {min_points}"
            )
        self.failure_threshold = float(failure_threshold)
        self.min_points = int(min_points)
        self._slope: float | None = None
        self._intercept: float | None = None
        self._r_squared: float | None = None
        self._slope_stderr: float | None = None
        self._n_points: int = 0

    # ------------------------------------------------------------------
    #  Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        timestamps_hours: np.ndarray,
        degradation_index: np.ndarray,
    ) -> dict[str, Any]:
        """Fit a linear trend ``index = slope * hours + intercept``.

        Parameters
        ----------
        timestamps_hours : (N,) ndarray of float
            Elapsed hours at each observation.
        degradation_index : (N,) ndarray of float
            Degradation in [0, 1] (0 = pristine, 1 = failure).

        Returns
        -------
        dict
            Keys: ``slope``, ``intercept``, ``r_squared``, ``slope_stderr``,
            ``intercept_stderr``, ``n_points``, ``x_min_hours``,
            ``x_max_hours``.

        Raises
        ------
        ValueError
            Fewer than ``min_points`` observations, shape mismatch,
            non-finite values, or a constant time axis.
        """
        x = np.asarray(timestamps_hours, dtype=np.float64)
        y = np.asarray(degradation_index, dtype=np.float64)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError(
                "timestamps_hours and degradation_index must be 1-D arrays of "
                f"equal length, got {x.shape} and {y.shape}"
            )
        if x.size < self.min_points:
            raise ValueError(
                f"Need at least min_points={self.min_points} observations for "
                f"a trend fit, got {x.size}"
            )
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError(
                "timestamps_hours and degradation_index must be finite — drop "
                "NaN/Inf (sensor dropout) rows before fitting"
            )
        if x.max() == x.min():
            raise ValueError(
                "timestamps_hours is constant — a linear trend cannot be fit "
                "on a single time point"
            )

        result = stats.linregress(x, y)
        self._slope = float(result.slope)
        self._intercept = float(result.intercept)
        rvalue = float(result.rvalue) if np.isfinite(result.rvalue) else 0.0
        self._r_squared = rvalue**2
        self._slope_stderr = (
            float(result.stderr) if np.isfinite(result.stderr) else None
        )
        self._n_points = int(x.size)

        return {
            "slope": self._slope,
            "intercept": self._intercept,
            "r_squared": self._r_squared,
            "slope_stderr": self._slope_stderr,
            "intercept_stderr": (
                float(result.intercept_stderr)
                if np.isfinite(result.intercept_stderr)
                else None
            ),
            "n_points": self._n_points,
            "x_min_hours": float(x.min()),
            "x_max_hours": float(x.max()),
        }

    # ------------------------------------------------------------------
    #  Predict
    # ------------------------------------------------------------------

    def predict_remaining_life(
        self,
        hours_elapsed: float,
        current_index: float,
    ) -> dict[str, Any]:
        """Extrapolate the fitted trend to the failure threshold.

        .. math::

            RUL = (failure\\_threshold - current\\_index) / slope

        Returns ``rul_hours = None`` when the trend is flat or improving
        (slope <= 0) — "no detectable degradation trend" — and ``0.0`` when
        the current index already meets/exceeds the failure threshold.

        Parameters
        ----------
        hours_elapsed : float
            Elapsed hours at the current observation (x-coordinate of the
            current point; used only for context and now-consistency checks).
        current_index : float
            Current degradation index in [0, 1].

        Returns
        -------
        dict
            Keys: ``rul_hours`` (float or None), ``status``, ``reason``,
            ``slope``, ``r_squared``, ``confidence`` (``"low"``/``"medium"``
            or None), ``ci_low_hours``, ``ci_high_hours`` (float or None;
            None = unbounded).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not completed successfully first.
        ValueError
            If inputs are not finite.
        """
        if self._slope is None:
            raise RuntimeError(
                "DegradationTrendRUL.fit() must complete successfully before "
                "predict_remaining_life()"
            )
        if not (np.isfinite(hours_elapsed) and np.isfinite(current_index)):
            raise ValueError("hours_elapsed and current_index must be finite")

        slope = self._slope
        r_squared = self._r_squared if self._r_squared is not None else 0.0

        if slope <= 0.0:
            return {
                "rul_hours": None,
                "status": _STATUS_NO_TREND,
                "reason": (
                    "No detectable degradation trend (slope <= 0); RUL not "
                    "estimated. Revisit when the health score starts "
                    "declining consistently."
                ),
                "slope": slope,
                "r_squared": r_squared,
                "confidence": None,
                "ci_low_hours": None,
                "ci_high_hours": None,
            }

        if current_index >= self.failure_threshold:
            return {
                "rul_hours": 0.0,
                "status": _STATUS_AT_THRESHOLD,
                "reason": (
                    "Current degradation index already at/above the failure "
                    "threshold; remaining life is 0."
                ),
                "slope": slope,
                "r_squared": r_squared,
                "confidence": "low",
                "ci_low_hours": None,
                "ci_high_hours": None,
            }

        horizon = (self.failure_threshold - current_index) / slope
        ci_low, ci_high = self._time_to_threshold_ci(current_index)
        return {
            "rul_hours": float(horizon),
            "status": _STATUS_OK,
            "reason": "Degradation-trend extrapolation to the failure threshold.",
            "slope": slope,
            "r_squared": r_squared,
            "confidence": self._confidence(r_squared, self._slope_stderr, ci_high),
            "ci_low_hours": ci_low,
            "ci_high_hours": ci_high,
        }

    def _time_to_threshold_ci(
        self, current_index: float
    ) -> tuple[float | None, float | None]:
        """95% CI on the extrapolated time-to-threshold (slope uncertainty only).

        t* = (threshold - current) / slope, evaluated at
        ``slope ± z * stderr`` (z = 1.96). If the lower slope bound is
        <= 0 (trend could be flat/improving) the upper RUL bound is
        unbounded (``None``). Intercept/model uncertainty is not included —
        the R² flag and confidence label carry that caveat.
        """
        if self._slope is None or self._slope_stderr is None:
            return None, None
        z = float(stats.norm.ppf(0.975))
        slope_lo = self._slope - z * self._slope_stderr
        slope_hi = self._slope + z * self._slope_stderr
        horizon_num = self.failure_threshold - current_index
        if horizon_num <= 0.0:
            return None, None
        if slope_lo <= 0.0:
            return horizon_num / slope_hi, None  # upper bound unbounded
        return horizon_num / slope_hi, horizon_num / slope_lo

    def _confidence(
        self,
        r_squared: float,
        slope_stderr: float | None,
        ci_high_hours: float | None,
    ) -> str:
        """Confidence label. Never ``"high"``: the trend→failure mapping is
        unvalidated extrapolation, so a perfect fit still only earns
        ``"medium"`` at best."""
        if r_squared < _LOW_R2:
            return "low"
        if slope_stderr is None or not np.isfinite(slope_stderr):
            return "low"
        if ci_high_hours is None:
            return "low"
        if self._n_points < _MIN_POINTS_FOR_MEDIUM_CONFIDENCE:
            return "low"
        return "medium"

    # ------------------------------------------------------------------
    #  Degradation index
    # ------------------------------------------------------------------

    @staticmethod
    def degradation_index_from_scores(health_scores: np.ndarray) -> np.ndarray:
        """Map 0-1 health scores to a 0-1 degradation index.

        ``index = 1 - clip(health, 0, 1)``: healthy (1.0) → 0.0 (pristine),
        failed (0.0) → 1.0 (failure). Values outside [0, 1] are clipped;
        NaN propagates — callers should drop dropout rows before fitting.

        Parameters
        ----------
        health_scores : (N,) ndarray of float
            Health scores in [0, 1] (1 = perfectly healthy).

        Returns
        -------
        (N,) ndarray of float
            Degradation index in [0, 1].
        """
        return 1.0 - np.clip(np.asarray(health_scores, dtype=np.float64), 0.0, 1.0)


def estimate_rul_from_scores(
    asset_id: str,
    score_history: list[dict[str, Any]],
    now_hours: float,
    failure_threshold: float = 0.9,
    min_points: int = 5,
) -> dict[str, Any]:
    """Build a degradation series from a health-score history and estimate RUL.

    This is the convenient entrypoint over persisted score records. Entries
    with missing/non-finite ``timestamp`` or ``health_score`` are dropped
    (sensor dropouts, logged at WARNING); the remainder is sorted by
    timestamp. RUL is measured from the latest observation: ``now_hours``
    earlier than the latest timestamp is clamped to it.

    Parameters
    ----------
    asset_id : str
        Asset identifier, echoed in the result.
    score_history : list of dict
        Each entry: ``{"timestamp": float hours, "health_score": float 0-1}``.
    now_hours : float
        Elapsed-hours clock at the current inspection.
    failure_threshold : float
        Degradation index treated as failure (default 0.9).
    min_points : int
        Minimum observations required for a trend fit (default 5).

    Returns
    -------
    dict
        Keys: ``asset_id``, ``rul_hours``, ``status``, ``reason``, ``slope``,
        ``intercept``, ``r_squared``, ``confidence``, ``ci_low_hours``,
        ``ci_high_hours``, ``failure_threshold``, ``n_points``,
        ``current_degradation_index``, ``disclaimer``.
    """
    estimator = DegradationTrendRUL(
        failure_threshold=failure_threshold, min_points=min_points
    )

    timestamps: list[float] = []
    scores: list[float] = []
    dropped = 0
    for entry in score_history:
        try:
            ts = float(entry["timestamp"])
            hs = float(entry["health_score"])
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if not (np.isfinite(ts) and np.isfinite(hs)):
            dropped += 1
            continue
        timestamps.append(ts)
        scores.append(hs)
    if dropped:
        logger.warning(
            "estimate_rul_from_scores(%s): dropped %d malformed or non-finite "
            "score entries (sensor dropouts)",
            asset_id,
            dropped,
        )

    if len(timestamps) < min_points:
        return _rul_result(
            estimator=estimator,
            asset_id=asset_id,
            status=_STATUS_INSUFFICIENT,
            reason=(
                f"Insufficient data: need at least min_points={min_points} "
                f"finite (timestamp, health_score) observations, got {len(timestamps)}."
            ),
            n_points=len(timestamps),
            current_index=None,
        )

    order = np.argsort(timestamps)
    hours = np.asarray(timestamps, dtype=np.float64)[order]
    index = estimator.degradation_index_from_scores(
        np.asarray(scores, dtype=np.float64)[order]
    )

    try:
        fit_summary = estimator.fit(hours, index)
    except ValueError as exc:
        return _rul_result(
            estimator=estimator,
            asset_id=asset_id,
            status=_STATUS_INSUFFICIENT,
            reason=str(exc),
            n_points=int(len(hours)),
            current_index=None,
        )

    current_hours = float(max(now_hours, hours[-1]))
    current_index = float(index[-1])
    prediction = estimator.predict_remaining_life(current_hours, current_index)

    return {
        "asset_id": asset_id,
        "rul_hours": prediction["rul_hours"],
        "status": prediction["status"],
        "reason": prediction["reason"],
        "slope": prediction["slope"],
        "intercept": fit_summary["intercept"],
        "r_squared": prediction["r_squared"],
        "confidence": prediction["confidence"],
        "ci_low_hours": prediction["ci_low_hours"],
        "ci_high_hours": prediction["ci_high_hours"],
        "failure_threshold": estimator.failure_threshold,
        "n_points": int(len(hours)),
        "current_degradation_index": current_index,
        "disclaimer": DISCLAIMER,
    }


def _rul_result(
    estimator: DegradationTrendRUL,
    asset_id: str,
    status: str,
    reason: str,
    n_points: int,
    current_index: float | None,
) -> dict[str, Any]:
    """Build the full result schema for the no-RUL guard branches."""
    return {
        "asset_id": asset_id,
        "rul_hours": None,
        "status": status,
        "reason": reason,
        "slope": None,
        "intercept": None,
        "r_squared": None,
        "confidence": None,
        "ci_low_hours": None,
        "ci_high_hours": None,
        "failure_threshold": estimator.failure_threshold,
        "n_points": n_points,
        "current_degradation_index": current_index,
        "disclaimer": DISCLAIMER,
    }
