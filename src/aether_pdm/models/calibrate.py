"""
Model calibration: anomaly thresholds, health scores, and fault probability scaling.

Provides:

* :func:`optimize_anomaly_threshold` — grid-search for the best IsolationForest
  decision threshold that balances detection rate vs false-alarm rate.
* :func:`calibrate_health_score` — sigmoid mapping from raw anomaly score to a
  0–1 health score.
* :func:`calibrate_anomaly_model` — end-to-end calibration pipeline (load val data,
  find threshold, log to MLflow).
* :func:`calibrate_fault_model` — Platt / sigmoid probability calibration for the
  RandomForest fault classifier using ``CalibratedClassifierCV``.
"""

import logging
import math
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import LabelEncoder

from aether_pdm.eval.metrics import classification_report_dict
from aether_pdm.models.anomaly import META_COLS as ANOMALY_META_COLS
from aether_pdm.models.fault import FAULT_LABELS
from aether_pdm.models.fault import META_COLS as FAULT_META_COLS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Low-level threshold & health-score helpers
# ---------------------------------------------------------------------------


def optimize_anomaly_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    target_recall: float = 0.90,
    max_far: float = 0.10,
) -> dict:
    """
    Grid-search over possible thresholds to maximise detection rate while
    keeping the false-alarm rate ≤ **max_far**.

    ``scores`` are raw IsolationForest ``decision_function`` outputs
    (positive = inlier, negative / low = anomaly).

    ``y_true``: 0 = normal, 1 = fault.

    At each candidate threshold *t*::

        pred = (scores < t).astype(int)
        far  = false_alarm_rate(y_true, pred)
        dr   = detection_rate(y_true, pred)

    Parameters
    ----------
    scores : (N,) ndarray
    y_true : (N,) ndarray
    target_recall : float
        Minimum desired detection rate (recall for the fault class).
    max_far : float
        Maximum tolerable false-alarm rate.

    Returns
    -------
    dict
        Keys: ``threshold``, ``false_alarm_rate``, ``detection_rate``, ``f1``,
        ``all_thresholds``, ``all_fars``, ``all_detection_rates``.
    """
    if scores.size == 0:
        return {
            "threshold": 0.0,
            "false_alarm_rate": 0.0,
            "detection_rate": 0.0,
            "f1": 0.0,
            "all_thresholds": np.array([]),
            "all_fars": np.array([]),
            "all_detection_rates": np.array([]),
        }

    thresholds = np.linspace(scores.min(), scores.max(), 200)

    # Vectorised grid: (N, 1) < (T,) → (N, T)
    preds = (scores[:, np.newaxis] < thresholds).astype(int)
    yt = y_true[:, np.newaxis]  # (N, 1)

    # Per-threshold counts (sum over samples, axis=0)
    fp = ((yt == 0) & (preds == 1)).sum(axis=0).astype(np.float64)
    tn = ((yt == 0) & (preds == 0)).sum(axis=0).astype(np.float64)
    tp = ((yt == 1) & (preds == 1)).sum(axis=0).astype(np.float64)
    fn_arr = ((yt == 1) & (preds == 0)).sum(axis=0).astype(np.float64)

    far_arr = fp / (fp + tn + 1e-12)
    dr_arr = tp / (tp + fn_arr + 1e-12)

    # Precision & F1 at each threshold
    precision_arr = np.where((tp + fp) > 0, tp / (tp + fp + 1e-12), 0.0)
    f1_arr = 2.0 * precision_arr * dr_arr / (precision_arr + dr_arr + 1e-12)

    # --- Selection ---
    mask = (far_arr <= max_far) & (dr_arr >= target_recall)

    if mask.any():
        valid_dr = dr_arr[mask]
        best_local = np.argmax(valid_dr)
        mask_indices = np.flatnonzero(mask)
        best_idx = mask_indices[best_local]
    else:
        cost = (1.0 - dr_arr) + far_arr
        best_idx = int(np.argmin(cost))
        logger.warning(
            "No threshold satisfies target_recall=%.2f AND max_far=%.2f. "
            "Returning best-available threshold (t=%.4f, far=%.4f, dr=%.4f).",
            target_recall,
            max_far,
            float(thresholds[best_idx]),
            float(far_arr[best_idx]),
            float(dr_arr[best_idx]),
        )

    return {
        "threshold": float(thresholds[best_idx]),
        "false_alarm_rate": float(far_arr[best_idx]),
        "detection_rate": float(dr_arr[best_idx]),
        "f1": float(f1_arr[best_idx]),
        "all_thresholds": thresholds,
        "all_fars": far_arr,
        "all_detection_rates": dr_arr,
    }


def calibrate_health_score(
    anomaly_score: float,
    threshold: float,
    steepness: float = 2.0,
) -> float:
    """
    Convert a raw IsolationForest ``decision_function`` score to a 0–1
    health score using a sigmoid mapping.

    .. math::

        health = \\frac{1}{1 + \\exp(-s \\cdot (score - threshold))}

    - ``score >> threshold`` → health ≈ 1.0  (very healthy)
    - ``score = threshold``  → health = 0.5  (boundary)
    - ``score << threshold`` → health ≈ 0.0  (very anomalous)

    Parameters
    ----------
    anomaly_score : float or ndarray
        Raw decision_function output(s).
    threshold : float
        Calibrated anomaly threshold.
    steepness : float
        Controls the sigmoid slope (higher = sharper transition).

    Returns
    -------
    float or ndarray
        Health score(s) in [0, 1].
    """
    return float(1.0 / (1.0 + math.exp(-steepness * (anomaly_score - threshold))))


# ---------------------------------------------------------------------------
#  End-to-end calibration pipelines
# ---------------------------------------------------------------------------


def calibrate_anomaly_model(
    model: IsolationForest,
    features_path: Path,
    mlflow_uri: str | None = None,
    target_recall: float = 0.90,
    max_far: float = 0.10,
    split: str = "val",
) -> dict:
    """
    Load the validation split, run threshold optimisation, and log results to MLflow.

    Steps
    -----
    1. Read features Parquet, filter to ``split``.
    2. Exclude ``META_COLS`` (from ``anomaly.py``) to get the feature matrix.
    3. Run ``model.decision_function`` to get raw scores.
    4. Build ``y_true`` from the ``fault_type`` column (normal=0, faulty=1).
    5. Call :func:`optimize_anomaly_threshold`.
    6. Log threshold + metrics to MLflow.

    Returns the dict from :func:`optimize_anomaly_threshold`.

    Raises
    ------
    ValueError
        If the filtered DataFrame is empty or contains only normal samples.
    """
    df = pd.read_parquet(features_path)
    df = df[df["split"] == split].copy()
    if df.empty:
        raise ValueError(f"No samples found with split='{split}'. Cannot calibrate anomaly model.")

    # Sanity: need at least some fault samples to calibrate against
    fault_mask = df["fault_type"] != "normal"
    if not fault_mask.any():
        raise ValueError(
            f"All val samples are healthy (split='{split}'). "
            "Need at least some fault samples to calibrate anomaly threshold."
        )

    # Feature matrix
    feature_cols = [c for c in df.columns if c not in ANOMALY_META_COLS]
    x_val = df[feature_cols].values.astype(np.float64)

    # Raw anomaly scores
    scores = model.decision_function(x_val)

    # Binary ground-truth: 0 = normal, 1 = faulty
    y_true = np.where(df["fault_type"] == "normal", 0, 1).astype(int)

    # Find best threshold
    result = optimize_anomaly_threshold(
        scores, y_true,
        target_recall=target_recall,
        max_far=max_far,
    )

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri or "mlruns")
    with mlflow.start_run(run_name="calibrate_anomaly") as run:
        mlflow.log_params({
            "threshold": result["threshold"],
            "target_recall": target_recall,
            "max_far": max_far,
            "steepness": 2.0,
        })
        mlflow.log_metrics({
            "calibrated_far": result["false_alarm_rate"],
            "calibrated_dr": result["detection_rate"],
            "calibrated_f1": result["f1"],
        })
        run_id = run.info.run_id
        print(f"Anomaly calibration complete. MLflow run: {run_id}")
        print(f"  Threshold={result['threshold']:.4f}, "
              f"FAR={result['false_alarm_rate']:.4f}, "
              f"DR={result['detection_rate']:.4f}, "
              f"F1={result['f1']:.4f}")

    return result


def calibrate_fault_model(
    model: RandomForestClassifier,
    le: LabelEncoder,
    features_path: Path,
    mlflow_uri: str | None = None,
    split: str = "val",
) -> tuple[RandomForestClassifier, dict]:
    """
    Calibrate RandomForest probabilities using Platt / sigmoid scaling.

    Uses ``CalibratedClassifierCV`` with a ``FrozenEstimator`` wrapper and
    ``PredefinedSplit`` — the sklearn-1.9+ equivalent of ``cv="prefit"`` —
    to fit a sigmoid regressor on the validation set without refitting the
    underlying model.

    Parameters
    ----------
    model : RandomForestClassifier
        Already-trained model (fitted on train split).
    le : LabelEncoder
        Fitted label encoder mapping class names ↔ integer indices.
    features_path : Path
        Parquet file containing windowed features + metadata.
    mlflow_uri : str or None
        MLflow tracking URI.
    split : str
        Data split to use for calibration (default ``"val"``).

    Returns
    -------
    tuple[RandomForestClassifier, dict]
        ``(calibrated_model, metrics_dict)`` where ``calibrated_model``
        is the ``CalibratedClassifierCV`` wrapper and ``metrics_dict``
        contains classification metrics on the validation set.

    Raises
    ------
    ValueError
        If the filtered DataFrame is empty or contains only a single class.
    """
    df = pd.read_parquet(features_path)
    df = df[df["split"] == split].copy()
    if df.empty:
        raise ValueError(
            f"No samples found with split='{split}'. Cannot calibrate fault model."
        )

    df = df[df["fault_type"].isin(FAULT_LABELS)]
    if df.empty:
        raise ValueError(
            f"No labeled fault samples found in split='{split}'."
        )

    # Feature matrix & labels
    feature_cols = [c for c in df.columns if c not in FAULT_META_COLS]
    x_val = df[feature_cols].values.astype(np.float64)
    y_val = le.transform(df["fault_type"])

    # Check for single class → sigmoid calibration would be degenerate
    unique_classes = np.unique(y_val)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Only one class present in split='{split}'. "
            "Probability calibration requires at least two classes."
        )

    calibrator = CalibratedClassifierCV(
        FrozenEstimator(model),
        cv=PredefinedSplit(test_fold=np.zeros(len(x_val), dtype=int)),
        method="sigmoid",
    )
    calibrator.fit(x_val, y_val)

    # Metrics on calibrated predictions
    y_pred = calibrator.predict(x_val)
    y_proba = calibrator.predict_proba(x_val)
    metrics = classification_report_dict(y_val, y_pred, labels=list(le.classes_))

    # Add log-loss for calibrated probabilities
    metrics["log_loss"] = float(log_loss(y_val, y_proba, labels=np.arange(len(le.classes_))))

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_uri or "mlruns")
    with mlflow.start_run(run_name="calibrate_fault") as run:
        mlflow.log_params({
            "calibration_method": "sigmoid",
            "cv": "prefit",
            "n_cal_samples": len(x_val),
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            calibrator, "calibrated_model",
            registered_model_name="aether-fault-clf-calibrated",
            serialization_format="pickle",
        )
        print(f"Fault calibration complete. MLflow run: {run.info.run_id}")
        print(f"  Metrics: {metrics}")

    return calibrator, metrics
