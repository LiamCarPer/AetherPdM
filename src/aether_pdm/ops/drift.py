"""
Drift monitoring for production feature and score distributions.

Detects when live (production) data diverges from a training reference
distribution using:

1. Population Stability Index (PSI) — binned distribution shift
2. Kolmogorov-Smirnov (KS) two-sample test — CDF distance
3. Score distribution monitoring — spike detection on anomaly scores

Interpretation (industry standard):
    PSI < 0.10  -> no significant drift
    PSI 0.10-0.25 -> moderate drift (investigate)
    PSI > 0.25  -> severe drift (alert)
"""

import json
import warnings
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from scipy import stats

META_COLS = {
    "window_id", "window_start", "window_end", "asset_id", "file_id",
    "channel", "fault_type", "fault_diameter", "severity", "split",
    "load_hp", "feature_version", "rpm", "sampling_rate", "waveform",
}

# Minimum count fraction used to avoid log(0) inside the PSI summation.
_PSI_EPSILON = 1e-4


def _clean_array(values: np.ndarray, name: str) -> np.ndarray:
    """Validate and sanitize a 1-D numeric array for drift math.

    Drops NaN/inf values with a warning and raises ``ValueError`` when the
    array is empty (or becomes empty after sanitization).

    Args:
        values: Input array (any numeric dtype).
        name: Human-readable label used in error/warning messages.

    Returns:
        Float64 1-D array containing only finite values.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty; cannot compute drift metrics.")
    if not np.all(np.isfinite(arr)):
        n_dropped = int(np.count_nonzero(~np.isfinite(arr)))
        warnings.warn(
            f"Dropped {n_dropped} non-finite value(s) (NaN/inf) from {name}.",
            stacklevel=2,
        )
        arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{name} contains only non-finite values; cannot compute drift metrics.")
    return arr


def psi(
    reference: np.ndarray,
    production: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index between two distributions.

    Bins the reference distribution, applies same edges to production,
    computes PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct)).

    Uses percentile-based bin edges from the reference to guarantee coverage.
    Clips zero counts to 1e-4 to avoid log(0).

    Returns a float. PSI < 0.10 no drift, 0.10-0.25 moderate, > 0.25 severe.

    Args:
        reference: Training/reference distribution.
        production: Live/production distribution.
        n_bins: Number of bins for the PSI histogram.

    Returns:
        PSI value. Constant (zero-variance) references return 0.0.

    Raises:
        ValueError: If either input array is empty.
    """
    ref = _clean_array(reference, "reference")
    prod = _clean_array(production, "production")

    edges = np.percentile(ref, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        # Reference is constant (zero variance): no meaningful binning.
        return 0.0

    # Expand outer edges so production values outside the reference
    # percentile range are still counted. numpy.histogram EXCLUDES values
    # outside a supplied bin-edge sequence, which would zero the production
    # histogram and break the PSI sum (industry standard places out-of-range
    # observations in the boundary bins).
    edges = edges.astype(np.float64, copy=False)
    edges[0] = min(float(edges[0]), float(prod.min()))
    edges[-1] = max(float(edges[-1]), float(prod.max()))

    ref_hist, _ = np.histogram(ref, bins=edges)
    prod_hist, _ = np.histogram(prod, bins=edges)

    ref_pct = ref_hist / ref_hist.sum()
    prod_pct = prod_hist / prod_hist.sum()

    ref_pct = np.clip(ref_pct, _PSI_EPSILON, None)
    prod_pct = np.clip(prod_pct, _PSI_EPSILON, None)

    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


def ks_drift(
    reference: np.ndarray,
    production: np.ndarray,
) -> dict:
    """
    Kolmogorov-Smirnov two-sample test.

    Returns dict:
    - statistic (float)
    - p_value (float)
    - drifted (bool): p_value < 0.05

    Args:
        reference: Training/reference distribution.
        production: Live/production distribution.

    Returns:
        Dict with ``statistic``, ``p_value`` and ``drifted`` keys.

    Raises:
        ValueError: If either input array is empty.
    """
    ref = _clean_array(reference, "reference")
    prod = _clean_array(production, "production")

    result = stats.ks_2samp(ref, prod)
    p_value = float(result.pvalue)
    return {
        "statistic": float(result.statistic),
        "p_value": p_value,
        "drifted": bool(p_value < 0.05),
    }


def drift_status(psi_value: float) -> str:
    """Map PSI to status: 'none', 'moderate', 'severe'."""
    if psi_value < 0.10:
        return "none"
    if psi_value < 0.25:
        return "moderate"
    return "severe"


def feature_drift_report(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-feature drift report comparing reference vs production distributions.

    For each numeric column in reference_df (excluding metadata columns),
    computes PSI, KS statistic, KS p-value, and drift status.

    Uses the intersection of numeric columns present in both dataframes so
    that differing feature sets degrade gracefully. String/timestamp columns
    are skipped.

    Returns DataFrame with columns:
    - feature
    - psi
    - ks_statistic
    - ks_p_value
    - status  ('none', 'moderate', 'severe')

    Args:
        reference_df: Reference (training) features.
        production_df: Production (live) features.

    Returns:
        Drift report DataFrame (one row per shared numeric feature).
    """
    if not isinstance(reference_df, pd.DataFrame) or not isinstance(production_df, pd.DataFrame):
        raise TypeError("reference_df and production_df must be pandas DataFrames.")

    common = sorted(set(reference_df.columns) & set(production_df.columns))
    rows: list[dict[str, Any]] = []

    for col in common:
        if col in META_COLS:
            continue
        ref_series = reference_df[col]
        prod_series = production_df[col]
        if not (
            pd.api.types.is_numeric_dtype(ref_series)
            and pd.api.types.is_numeric_dtype(prod_series)
        ):
            continue

        ref_arr = _clean_array(ref_series.to_numpy(), f"reference['{col}']")
        prod_arr = _clean_array(prod_series.to_numpy(), f"production['{col}']")

        psi_value = psi(ref_arr, prod_arr)
        ks = ks_drift(ref_arr, prod_arr)
        rows.append({
            "feature": col,
            "psi": psi_value,
            "ks_statistic": ks["statistic"],
            "ks_p_value": ks["p_value"],
            "status": drift_status(psi_value),
        })

    return pd.DataFrame(rows, columns=["feature", "psi", "ks_statistic", "ks_p_value", "status"])


def score_distribution_monitor(
    reference_scores: np.ndarray,
    production_scores: np.ndarray,
    spike_z_threshold: float = 3.0,
) -> dict:
    """
    Detect spikes in production anomaly scores vs reference.

    Computes:
    - reference mean/std
    - production mean/std
    - z-score of production mean vs reference distribution
    - PSI of score distributions
    - ratio of production scores exceeding reference 95th percentile

    Returns dict:
    - reference_mean, reference_std
    - production_mean, production_std
    - z_score (float)
    - psi (float)
    - p95_violation_rate (fraction of production scores above ref p95)
    - spike_detected (bool): z_score > threshold OR p95_violation_rate > 0.10

    Args:
        reference_scores: Training/reference anomaly scores.
        production_scores: Live/production anomaly scores.
        spike_z_threshold: Z-score threshold above which a spike is declared.

    Returns:
        Dict of score-distribution drift metrics.

    Raises:
        ValueError: If either input array is empty.
    """
    ref = _clean_array(reference_scores, "reference_scores")
    prod = _clean_array(production_scores, "production_scores")

    ref_mean = float(np.mean(ref))
    ref_std = float(np.std(ref))
    prod_mean = float(np.mean(prod))
    prod_std = float(np.std(prod))

    z_score = (prod_mean - ref_mean) / (ref_std + 1e-12)

    psi_value = psi(ref, prod)

    p95 = float(np.percentile(ref, 95))
    p95_violation_rate = float(np.mean(prod > p95))

    spike_detected = bool(z_score > spike_z_threshold or p95_violation_rate > 0.10)

    return {
        "reference_mean": ref_mean,
        "reference_std": ref_std,
        "production_mean": prod_mean,
        "production_std": prod_std,
        "z_score": z_score,
        "psi": psi_value,
        "p95_violation_rate": p95_violation_rate,
        "spike_detected": spike_detected,
    }


def detect_drift(
    features_path: Path,
    reference_split: str = "train",
    production_split: str = "test",
) -> dict:
    """
    Full drift detection pipeline on a features Parquet file.

    Loads features, splits by reference/production splits, computes:
    - feature_drift_report (reference vs production)
    - overall PSI (mean of feature PSIs)
    - worst feature (highest PSI)
    - n_features_drifted (severe count)
    - drift_fired (bool): any feature severe OR mean PSI >= 0.25

    Returns dict with keys above.

    Args:
        features_path: Path to features Parquet file with a ``split`` column.
        reference_split: Split label used as the reference distribution.
        production_split: Split label used as the production distribution.

    Returns:
        Drift summary dict.

    Raises:
        FileNotFoundError: If the features file does not exist.
        ValueError: If the file has no ``split`` column or either split is empty.
    """
    features_path = Path(features_path)
    if not features_path.exists():
        raise FileNotFoundError(f"Features file '{features_path}' not found.")

    df = pd.read_parquet(features_path)
    if "split" not in df.columns:
        raise ValueError(
            f"Features file '{features_path}' has no 'split' column. "
            "Cannot detect drift."
        )

    reference_df = df[df["split"] == reference_split]
    if reference_df.empty:
        raise ValueError(
            f"No samples found with split='{reference_split}' in '{features_path}'. "
            "Cannot detect drift."
        )

    production_df = df[df["split"] == production_split]
    if production_df.empty:
        raise ValueError(
            f"No samples found with split='{production_split}' in '{features_path}'. "
            "Cannot detect drift."
        )

    report = feature_drift_report(reference_df, production_df)

    if report.empty:
        warnings.warn(
            "No common numeric features found between reference and production splits.",
            stacklevel=2,
        )
        mean_psi = 0.0
        worst_feature: str | None = None
        worst_psi = 0.0
        n_features_drifted = 0
        drift_fired = False
    else:
        mean_psi = float(report["psi"].mean())
        worst_idx = int(report["psi"].idxmax())
        worst_feature = str(report.loc[worst_idx, "feature"])
        worst_psi = float(report.loc[worst_idx, "psi"])
        n_features_drifted = int((report["status"] == "severe").sum())
        drift_fired = bool(n_features_drifted > 0 or mean_psi >= 0.25)

    return {
        "feature_drift_report": report,
        "mean_psi": mean_psi,
        "worst_feature": worst_feature,
        "worst_psi": worst_psi,
        "n_features_drifted": n_features_drifted,
        "drift_fired": drift_fired,
    }


def log_drift_to_mlflow(
    drift_result: dict,
    mlflow_uri: str | None = None,
    run_name: str = "drift_monitor",
) -> None:
    """
    Log drift summary to MLflow: mean_psi, n_features_drifted, drift_fired
    as metrics, and feature-level PSIs as params (JSON string).

    Args:
        drift_result: Output of :func:`detect_drift`.
        mlflow_uri: MLflow tracking URI (defaults to ``mlruns``).
        run_name: MLflow run name.
    """
    mlflow.set_tracking_uri(mlflow_uri or "mlruns")

    report = drift_result.get("feature_drift_report")
    feature_psi: dict[str, float] = {}
    if isinstance(report, pd.DataFrame) and not report.empty:
        if "feature" in report.columns and "psi" in report.columns:
            feature_psi = {
                str(feature): float(value)
                for feature, value in zip(report["feature"], report["psi"], strict=False)
            }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics({
            "mean_psi": float(drift_result.get("mean_psi", 0.0)),
            "n_features_drifted": float(drift_result.get("n_features_drifted", 0)),
            "worst_psi": float(drift_result.get("worst_psi", 0.0)),
        })
        mlflow.log_param("feature_drift", json.dumps(feature_psi))
        mlflow.log_param("drift_fired", str(drift_result.get("drift_fired", False)))
