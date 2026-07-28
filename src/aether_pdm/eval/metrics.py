"""
Evaluation metrics for anomaly detection and fault classification.

Focus on B2B-relevant metrics: false alarm rate, lead-time proxy,
balanced accuracy, not just raw accuracy.
"""

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
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
