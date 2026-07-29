"""
Evaluation metrics for anomaly detection and fault classification.

Focus on B2B-relevant metrics: false alarm rate, lead-time proxy,
balanced accuracy, not just raw accuracy.
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    False alarm rate = FP / (FP + TN).
    Only meaningful when y_true has a 'normal' (negative) class.
    For anomaly detection: normal=0, anomaly=1.
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(fp / (fp + tn + 1e-12))


def detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Detection rate (recall for fault class).
    For anomaly: how many actual faults were caught.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn + 1e-12))


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str] | None = None
) -> dict:
    """
    Return a dict of classification metrics suitable for MLflow logging.
    """
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
    }


def compute_anomaly_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Compute anomaly-specific metrics: FAR, detection rate, F1."""
    return {
        "false_alarm_rate": false_alarm_rate(y_true, y_pred),
        "detection_rate": detection_rate(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def threshold_scan(
    scores: np.ndarray,
    y_true: np.ndarray,
    n_thresholds: int = 200,
) -> dict:
    """
    Scan all candidate thresholds vectorized and return FAR + detection_rate arrays.

    Treats ``scores < threshold`` as anomalous (pred=1).  For IsolationForest
    ``decision_function`` scores this is correct: negative/very-low = anomaly.

    Parameters
    ----------
    scores : (N,) ndarray
        Raw anomaly scores from the model.
    y_true : (N,) ndarray
        Binary labels (0 = normal, 1 = fault).
    n_thresholds : int
        Number of evenly-spaced candidate thresholds.

    Returns
    -------
    dict with keys ``thresholds``, ``false_alarm_rates``, ``detection_rates``.
    Each value is an ndarray of length ``n_thresholds``.
    """
    if scores.size == 0 or y_true.size == 0:
        return {
            "thresholds": np.array([]),
            "false_alarm_rates": np.array([]),
            "detection_rates": np.array([]),
        }

    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)

    # Vectorized grid: (N, 1) < (T,) → (N, T) boolean → int
    preds = (scores[:, np.newaxis] < thresholds).astype(int)
    yt = y_true[:, np.newaxis]  # (N, 1)

    # Per-threshold class counts (sum over samples, axis=0)
    fp = ((yt == 0) & (preds == 1)).sum(axis=0).astype(np.float64)
    tn = ((yt == 0) & (preds == 0)).sum(axis=0).astype(np.float64)
    tp = ((yt == 1) & (preds == 1)).sum(axis=0).astype(np.float64)
    fn_arr = ((yt == 1) & (preds == 0)).sum(axis=0).astype(np.float64)

    far = fp / (fp + tn + 1e-12)
    dr = tp / (tp + fn_arr + 1e-12)

    return {
        "thresholds": thresholds,
        "false_alarm_rates": far,
        "detection_rates": dr,
    }


def find_optimal_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    target_recall: float = 0.90,
    max_far: float = 0.10,
    n_thresholds: int = 200,
) -> dict:
    """
    Find the best threshold that satisfies recall and FAR constraints.

    Scans ``n_thresholds`` candidates, filters to those where
    ``false_alarm_rate <= max_far`` and ``detection_rate >= target_recall``,
    then returns the threshold that maximises detection rate.

    When no candidate satisfies both constraints the threshold that
    minimises ``(1 - detection_rate) + false_alarm_rate`` is returned.

    Parameters
    ----------
    scores : (N,) ndarray
    y_true : (N,) ndarray
    target_recall : float
        Minimum acceptable detection rate (recall for fault class).
    max_far : float
        Maximum acceptable false-alarm rate.
    n_thresholds : int
        Number of candidate thresholds.

    Returns
    -------
    dict
        All keys from :func:`threshold_scan` plus ``best_threshold``,
        ``best_far``, ``best_detection_rate``.
    """
    result = threshold_scan(scores, y_true, n_thresholds=n_thresholds)

    thresholds = result["thresholds"]
    far = result["false_alarm_rates"]
    dr = result["detection_rates"]

    if thresholds.size == 0:
        return {
            **result,
            "best_threshold": 0.0,
            "best_far": 0.0,
            "best_detection_rate": 0.0,
        }

    # Valid candidates: FAR <= max_far AND DR >= target_recall
    mask = (far <= max_far) & (dr >= target_recall)

    if mask.any():
        # Pick the candidate with the highest detection rate
        valid_dr = dr[mask]
        best_local = np.argmax(valid_dr)
        mask_indices = np.flatnonzero(mask)
        best_idx = mask_indices[best_local]
    else:
        # Fallback: minimise (1 - DR) + FAR
        cost = (1.0 - dr) + far
        best_idx = int(np.argmin(cost))

    return {
        **result,
        "best_threshold": float(thresholds[best_idx]),
        "best_far": float(far[best_idx]),
        "best_detection_rate": float(dr[best_idx]),
    }
