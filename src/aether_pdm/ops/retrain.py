"""
Retrain pipeline: drift-triggered model refresh.

Flow:
    1. Optional drift check (if drift_fired OR force=True)
    2. Retrain anomaly detector on current data (train split)
    3. Retrain fault classifier on current data (train split)
    4. Calibrate anomaly threshold on val split
    5. Run promotion gate on both models (evaluate on val)
    6. If gate rejects, previous production stays active (implicit rollback)

This module orchestrates the existing train_* and promote_* functions.
It does NOT re-implement training or evaluation logic.
"""

from pathlib import Path
from typing import Any

from aether_pdm.models.anomaly import train_anomaly
from aether_pdm.models.fault import train_fault_classifier
from aether_pdm.ops.drift import detect_drift
from aether_pdm.ops.promote import promote_anomaly, promote_fault

_DRIFT_SUMMARY_KEYS = ("mean_psi", "worst_feature", "n_features_drifted")


def _drift_summary(drift: dict[str, Any]) -> dict[str, Any]:
    """Extract the summary subset of a ``detect_drift`` result."""
    return {key: drift.get(key) for key in _DRIFT_SUMMARY_KEYS}


def _promotion_outcome(anomaly: dict[str, Any], fault: dict[str, Any]) -> str:
    """Classify the combined promotion outcome of both gates."""
    anomaly_promoted = anomaly.get("decision") == "promoted"
    fault_promoted = fault.get("decision") == "promoted"
    if anomaly_promoted and fault_promoted:
        return "promoted"
    if anomaly_promoted or fault_promoted:
        return "partial"
    return "rejected"


def _run_promotion_gate(
    promote_fn: Any,
    features_path: Path,
    mlflow_uri: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Run one promotion gate, recording (not crashing on) ``ValueError``.

    On success returns ``(result, None)``. When the gate raises ``ValueError``
    (e.g. no registered candidate), returns an error result with
    ``decision="error"`` plus the original message so the pipeline can keep
    running and report the failure.
    """
    try:
        return promote_fn(features_path, mlflow_uri), None
    except ValueError as exc:
        return (
            {
                "candidate_version": None,
                "decision": "error",
                "reason": f"promotion error: {exc}",
                "metrics": {},
            },
            str(exc),
        )


def should_retrain(
    features_path: Path,
    drift_threshold: float = 0.25,
    force: bool = False,
) -> dict[str, Any]:
    """
    Decide whether retraining is needed.

    Runs ``detect_drift``. If ``drift_fired`` (or the mean PSI meets/exceeds
    ``drift_threshold``) OR ``force`` is True, returns ``retrain=True``.
    Otherwise returns ``retrain=False``.

    Returns dict:
    - retrain (bool)
    - reason (str): "drift_fired" | "forced" | "no_drift"
    - drift (dict): the full detect_drift result

    Raises:
        FileNotFoundError: If the features file does not exist.
        ValueError: If drift detection fails (e.g. missing split column);
            re-raised with additional context.
    """
    features_path = Path(features_path)
    try:
        drift = detect_drift(features_path)
    except ValueError as exc:
        raise ValueError(
            f"Drift check failed for features file '{features_path}': {exc}"
        ) from exc

    mean_psi = drift.get("mean_psi", 0.0)
    drift_fired = bool(drift.get("drift_fired", False) or mean_psi >= drift_threshold)

    if force:
        return {"retrain": True, "reason": "forced", "drift": drift}
    if drift_fired:
        return {"retrain": True, "reason": "drift_fired", "drift": drift}
    return {"retrain": False, "reason": "no_drift", "drift": drift}


def retrain_models(
    features_path: Path,
    mlflow_uri: str | None = None,
    train_split: str = "train",
    **train_kwargs: Any,
) -> dict[str, Any]:
    """
    Retrain both models on the current data.

    Calls ``train_anomaly`` and ``train_fault_classifier`` on the
    ``train_split`` split, passing through ``train_kwargs`` (contamination,
    n_estimators, etc.). The ``split`` and ``mlflow_uri`` keys are always
    controlled by the explicit ``train_split`` / ``mlflow_uri`` parameters.

    Returns dict:
    - anomaly_trained (bool)
    - fault_trained (bool)
    - anomaly_model_type (str)
    - fault_model_type (str)

    Raises:
        ValueError: If either trainer fails (e.g. no healthy samples);
            errors propagate, never swallowed.
    """
    features_path = Path(features_path)
    kwargs = dict(train_kwargs)
    kwargs.pop("split", None)
    kwargs.pop("mlflow_uri", None)

    anomaly_model = train_anomaly(
        features_path, mlflow_uri=mlflow_uri, split=train_split, **kwargs
    )
    fault_model, _ = train_fault_classifier(
        features_path, mlflow_uri=mlflow_uri, split=train_split, **kwargs
    )

    return {
        "anomaly_trained": True,
        "fault_trained": True,
        "anomaly_model_type": type(anomaly_model).__name__,
        "fault_model_type": type(fault_model).__name__,
    }


def run_retrain_pipeline(
    features_path: Path,
    mlflow_uri: str | None = None,
    force: bool = False,
    drift_threshold: float = 0.25,
    **train_kwargs: Any,
) -> dict[str, Any]:
    """
    End-to-end retrain pipeline.

    Steps:
    1. should_retrain(features_path, drift_threshold, force)
       - if not retrain: return {"skipped": True, "reason": reason, "drift": drift}
    2. retrain_models(features_path, mlflow_uri, **train_kwargs)
    3. promote_anomaly(features_path, mlflow_uri)
    4. promote_fault(features_path, mlflow_uri)
    5. Compile results.

    Returns dict:
    - skipped (bool)
    - skip_reason (str | None)
    - drift_summary (dict): mean_psi, worst_feature, n_features_drifted
    - retrained (bool)
    - anomaly: promote_anomaly result
    - fault: promote_fault result
    - outcome (str): "promoted" if both promoted, "partial" if one,
      "rejected" if none, "skipped" if not retrained
    - anomaly_error (str | None): promotion error for the anomaly gate, if any
    - fault_error (str | None): promotion error for the fault gate, if any
    """
    features_path = Path(features_path)
    decision = should_retrain(
        features_path, drift_threshold=drift_threshold, force=force
    )
    drift = decision.get("drift", {})
    summary = _drift_summary(drift)

    if not decision.get("retrain", False):
        return {
            "skipped": True,
            "skip_reason": decision.get("reason"),
            "drift_summary": summary,
            "retrained": False,
            "anomaly": None,
            "fault": None,
            "outcome": "skipped",
            "anomaly_error": None,
            "fault_error": None,
        }

    retrain_result = retrain_models(
        features_path, mlflow_uri=mlflow_uri, **train_kwargs
    )
    retrained = bool(
        retrain_result.get("anomaly_trained") and retrain_result.get("fault_trained")
    )

    anomaly, anomaly_error = _run_promotion_gate(
        promote_anomaly, features_path, mlflow_uri
    )
    fault, fault_error = _run_promotion_gate(
        promote_fault, features_path, mlflow_uri
    )

    return {
        "skipped": False,
        "skip_reason": None,
        "drift_summary": summary,
        "retrained": retrained,
        "anomaly": anomaly,
        "fault": fault,
        "outcome": _promotion_outcome(anomaly, fault),
        "anomaly_error": anomaly_error,
        "fault_error": fault_error,
    }
