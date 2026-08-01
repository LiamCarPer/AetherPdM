"""
Promotion gate for MLflow model registry.

Evaluates a candidate model (staging stage, or latest if no staging exists)
against validation data. Promotes to 'production' only if metrics pass
the configured thresholds (e.g., FAR <= max_far, detection_rate >= min_recall).
Logs the decision and metrics to MLflow.

The decision itself is delegated to the GatedOps gate engine: AetherPdM
releases pass through the same contract that promotes GatedOps' reference
models, and every promotion is recorded as a GatedOps lineage manifest on the
model version.
"""

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from gatedops.gate.engine import evaluate_gate
from gatedops.gate.rules import GateConfig, ThresholdRule
from gatedops.manifest.builder import build_manifest
from gatedops.manifest.hashing import sha256_file
from gatedops.registry.mlflow_ import MlflowRegistry
from mlflow.exceptions import MlflowException
from sklearn.preprocessing import LabelEncoder

from aether_pdm.eval.metrics import (
    classification_report_dict,
    detection_rate,
    false_alarm_rate,
    find_optimal_threshold,
)
from aether_pdm.models.anomaly import META_COLS as ANOMALY_META_COLS
from aether_pdm.models.fault import META_COLS as FAULT_META_COLS

DEFAULT_SPLIT = "val"


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Keep only numeric values for ``mlflow.log_metrics`` compatibility."""
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def anomaly_gate(max_far: float = 0.10, min_recall: float = 0.80) -> GateConfig:
    """The anomaly promotion gate, expressed as a GatedOps ``GateConfig``."""
    return GateConfig(
        thresholds=[
            ThresholdRule(metric="detection_rate", op=">=", value=min_recall),
            ThresholdRule(metric="false_alarm_rate", op="<=", value=max_far),
        ]
    )


def fault_gate(
    min_f1_macro: float = 0.90,
    min_balanced_accuracy: float = 0.90,
) -> GateConfig:
    """The fault classifier promotion gate, as a GatedOps ``GateConfig``."""
    return GateConfig(
        thresholds=[
            ThresholdRule(metric="f1_macro", op=">=", value=min_f1_macro),
            ThresholdRule(
                metric="balanced_accuracy", op=">=", value=min_balanced_accuracy
            ),
        ]
    )


def load_promote_gates(path: str | Path = "configs/promote.yaml") -> dict[str, GateConfig]:
    """Load the canonical promotion gates from ``configs/promote.yaml``."""
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return {
        name: GateConfig(
            thresholds=[ThresholdRule(**threshold) for threshold in spec["thresholds"]]
        )
        for name, spec in raw.items()
        if isinstance(spec, dict) and "thresholds" in spec
    }


def _build_anomaly_reason(
    recall: float,
    min_recall: float,
    far: float,
    max_far: float,
) -> str:
    """Human-readable reason for the promotion decision."""
    if recall >= min_recall and far <= max_far:
        return f"Metrics pass gate: DR={recall:.4f} >= {min_recall}, FAR={far:.4f} <= {max_far}"
    failures: list[str] = []
    if recall < min_recall:
        failures.append(f"DR={recall:.4f} < {min_recall}")
    if far > max_far:
        failures.append(f"FAR={far:.4f} > {max_far}")
    return "Thresholds not met: " + ", ".join(failures)


def _build_fault_reason(
    f1_macro: float,
    min_f1_macro: float,
    balanced_accuracy: float,
    min_balanced_accuracy: float,
) -> str:
    """Human-readable reason for the fault promotion decision."""
    if f1_macro >= min_f1_macro and balanced_accuracy >= min_balanced_accuracy:
        return (
            f"Metrics pass gate: f1_macro={f1_macro:.4f} >= {min_f1_macro}, "
            f"balanced_accuracy={balanced_accuracy:.4f} >= {min_balanced_accuracy}"
        )
    failures: list[str] = []
    if f1_macro < min_f1_macro:
        failures.append(f"f1_macro={f1_macro:.4f} < {min_f1_macro}")
    if balanced_accuracy < min_balanced_accuracy:
        failures.append(
            f"balanced_accuracy={balanced_accuracy:.4f} < {min_balanced_accuracy}"
        )
    return "Thresholds not met: " + ", ".join(failures)


def evaluate_anomaly_candidate(
    model: Any,
    features_path: Path,
    split: str = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """
    Evaluate an anomaly model on val split.

    Loads features Parquet, filters to split, computes decision_function scores,
    builds y_true (normal=0, fault=1), computes metrics via threshold scan.

    Returns dict with keys:
    - n_samples
    - false_alarm_rate (at default threshold 0)
    - detection_rate (at default threshold 0)
    - best_threshold (from find_optimal_threshold)
    - best_far
    - best_detection_rate
    - n_faults
    - n_normal

    Raises
    ------
    ValueError
        If the split is empty, missing, contains only healthy samples
        (no fault samples to measure detection rate against), or contains
        only fault samples (no normal samples to measure false alarm rate
        against).
    """
    df = pd.read_parquet(features_path)
    if "split" not in df.columns:
        raise ValueError(
            f"Features file '{features_path}' has no 'split' column. "
            "Cannot evaluate anomaly candidate."
        )
    df = df[df["split"] == split].copy()
    if df.empty:
        raise ValueError(
            f"No samples found with split='{split}' in '{features_path}'. "
            "Cannot evaluate anomaly candidate."
        )

    fault_mask = df["fault_type"] != "normal"
    if not fault_mask.any():
        raise ValueError(
            f"All val samples are healthy (split='{split}'). "
            "Cannot compute detection rate (need at least one fault sample)."
        )

    feature_cols = [c for c in df.columns if c not in ANOMALY_META_COLS]
    x_val = df[feature_cols].values.astype(np.float64)
    scores = model.decision_function(x_val)
    y_true = np.where(df["fault_type"] == "normal", 0, 1).astype(int)

    n_normal = int((y_true == 0).sum())
    if n_normal == 0:
        raise ValueError(
            f"Validation split has no normal samples (split='{split}'); "
            "cannot compute false alarm rate."
        )

    y_pred = (scores < 0).astype(int)  # default threshold 0 (matches predict_anomaly)

    threshold_result = find_optimal_threshold(scores, y_true)

    return {
        "n_samples": int(len(y_true)),
        "n_faults": int(fault_mask.sum()),
        "n_normal": n_normal,
        "false_alarm_rate": false_alarm_rate(y_true, y_pred),
        "detection_rate": detection_rate(y_true, y_pred),
        "best_threshold": threshold_result["best_threshold"],
        "best_far": threshold_result["best_far"],
        "best_detection_rate": threshold_result["best_detection_rate"],
    }


def evaluate_fault_candidate(
    model: Any,
    le: LabelEncoder,
    features_path: Path,
    split: str = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """
    Evaluate fault classifier on val split.

    Returns dict:
    - n_samples
    - f1_macro
    - balanced_accuracy
    - classes (list)

    Raises
    ------
    ValueError
        If the split is empty, contains no known classes, or contains only
        a single class (f1_macro would be degenerate).
    """
    df = pd.read_parquet(features_path)
    if "split" not in df.columns:
        raise ValueError(
            f"Features file '{features_path}' has no 'split' column. "
            "Cannot evaluate fault candidate."
        )
    df = df[df["split"] == split].copy()
    if df.empty:
        raise ValueError(
            f"No samples found with split='{split}' in '{features_path}'. "
            "Cannot evaluate fault candidate."
        )

    # Restrict to the classes the model was trained on (le.classes_).
    df = df[df["fault_type"].isin(le.classes_)]
    if df.empty:
        raise ValueError(
            f"No labeled fault samples found in split='{split}' for model classes "
            f"{list(le.classes_)}. Cannot evaluate fault candidate."
        )
    if df["fault_type"].nunique() < 2:
        raise ValueError(
            f"Only one class present in split='{split}' after filtering to model classes. "
            "Cannot evaluate fault candidate (f1_macro would be degenerate)."
        )

    feature_cols = [c for c in df.columns if c not in FAULT_META_COLS]
    x_val = df[feature_cols].values.astype(np.float64)
    y_true = le.transform(df["fault_type"])
    y_pred = model.predict(x_val)

    metrics = classification_report_dict(y_true, y_pred, labels=le.classes_.tolist())

    return {
        "n_samples": int(len(y_true)),
        "f1_macro": metrics["f1_macro"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "classes": le.classes_.tolist(),
    }


def _attach_manifest(
    client: Any,
    model_name: str,
    model_version: int,
    metrics: dict[str, Any],
    gate: Any,
    features_path: Path,
) -> Any:
    """Attach a GatedOps lineage manifest to a model version.

    The manifest uses the same schema GatedOps emits for its reference models,
    so every AetherPdM release carries the identical lineage contract. It
    records the evaluated metrics, the gate verdict, the artifact hash of the
    served model bytes, and a hash of the evaluation features.
    """
    version = client.get_model_version(model_name, str(model_version))
    registry = MlflowRegistry(tracking_uri=mlflow.get_tracking_uri())
    artifact = registry.version_artifact(model_name, str(model_version))
    manifest = build_manifest(
        model_name=model_name,
        model_version=str(model_version),
        artifact=artifact,
        run_id=version.run_id,
        data_hash=sha256_file(Path(features_path)),
        metrics=_numeric_metrics(metrics),
        gate=gate,
        promote_stage="Production" if gate.status == "PASS" else "None",
    )
    client.set_model_version_tag(
        model_name, str(model_version), "gatedops.manifest", manifest.model_dump_json()
    )
    return manifest


def _load_candidate_model(name: str, client: Any) -> tuple[Any, int]:
    """
    Find candidate version: prefer 'staging' stage, else latest any-stage.
    Returns (model, version).

    Raises
    ------
    ValueError
        If no model version exists for ``name``.
    """
    staging: list[Any] = []
    try:
        staging = client.get_latest_versions(name, stages=["Staging"])
    except MlflowException:
        # Model registry missing entirely (or store-specific error) -> no staging.
        staging = []

    if staging:
        version = int(staging[0].version)
        model = mlflow.sklearn.load_model(staging[0].source)
        return model, version

    latest = client.search_model_versions(
        f"name='{name}'",
        order_by=["version_number DESC"],
        max_results=1,
    )
    if not latest:
        raise ValueError(
            f"No model versions found for '{name}'. "
            "Train and register a model before running the promotion gate."
        )

    version = int(latest[0].version)
    model = mlflow.sklearn.load_model(latest[0].source)
    return model, version


def _load_fault_label_encoder(client: Any, model_name: str, version: int) -> LabelEncoder:
    """Rebuild the fault LabelEncoder from the 'classes' param logged at training."""
    versions = client.search_model_versions(
        f"name='{model_name}' and version_number={version}"
    )
    if not versions:
        raise ValueError(f"Cannot find version {version} of '{model_name}'.")
    run = client.get_run(versions[0].run_id)
    classes_param = run.data.params.get("classes", "")
    if not classes_param:
        raise ValueError(
            f"No 'classes' param recorded for '{model_name}' v{version}. "
            "Re-train with train_fault_classifier to record class labels."
        )
    return LabelEncoder().fit(classes_param.split(","))


def promote_anomaly(
    features_path: Path,
    mlflow_uri: str | None = None,
    max_far: float = 0.10,
    min_recall: float = 0.80,
    model_name: str = "aether-anomaly",
    gate: GateConfig | None = None,
) -> dict[str, Any]:
    """
    Load latest candidate (staging, else latest any-stage), evaluate on val,
    and promote to production if the GatedOps gate passes.

    The decision is made by ``gatedops.gate.engine.evaluate_gate`` against the
    anomaly thresholds (detection_rate >= min_recall, false_alarm_rate
    <= max_far), and the promotion is recorded as a GatedOps lineage manifest.

    Returns dict:
    - candidate_version
    - decision: "promoted" | "rejected"
    - reason
    - metrics (from evaluate_anomaly_candidate)
    - gate (GatedOps GateReport)

    Raises
    ------
    ValueError
        If no candidate model exists or the val split is empty / has no faults.
    """
    mlflow.set_tracking_uri(mlflow_uri or "mlruns")
    client = mlflow.tracking.MlflowClient()
    model, candidate_version = _load_candidate_model(model_name, client)

    metrics = evaluate_anomaly_candidate(model, features_path, split=DEFAULT_SPLIT)

    recall = float(metrics["detection_rate"])
    far = float(metrics["false_alarm_rate"])
    report = evaluate_gate(
        gate or anomaly_gate(max_far, min_recall),
        {"detection_rate": recall, "false_alarm_rate": far},
        model_name=model_name,
    )
    decision = "promoted" if report.status == "PASS" else "rejected"
    reason = _build_anomaly_reason(recall, min_recall, far, max_far)

    if decision == "promoted":
        client.transition_model_version_stage(
            name=model_name,
            version=str(candidate_version),
            stage="Production",
        )

    _attach_manifest(client, model_name, candidate_version, metrics, report, features_path)

    with mlflow.start_run(run_name="promote_anomaly"):
        mlflow.log_params({
            "candidate_version": str(candidate_version),
            "decision": decision,
            "reason": reason,
            "gate_status": report.status,
        })
        mlflow.log_metrics(_numeric_metrics(metrics))

    return {
        "candidate_version": candidate_version,
        "decision": decision,
        "reason": reason,
        "metrics": metrics,
        "gate": report,
    }


def promote_fault(
    features_path: Path,
    mlflow_uri: str | None = None,
    min_f1_macro: float = 0.90,
    min_balanced_accuracy: float = 0.90,
    model_name: str = "aether-fault-clf",
    gate: GateConfig | None = None,
) -> dict[str, Any]:
    """
    Load latest candidate, evaluate on val, and promote if the GatedOps gate
    passes (f1_macro >= min_f1_macro, balanced_accuracy >= min_balanced_accuracy).

    Returns dict:
    - candidate_version
    - decision: "promoted" | "rejected"
    - reason
    - metrics (from evaluate_fault_candidate)
    - gate (GatedOps GateReport)

    Raises
    ------
    ValueError
        If no candidate model exists or the val split is empty / degenerate.
    """
    mlflow.set_tracking_uri(mlflow_uri or "mlruns")
    client = mlflow.tracking.MlflowClient()
    model, candidate_version = _load_candidate_model(model_name, client)
    le = _load_fault_label_encoder(client, model_name, candidate_version)

    metrics = evaluate_fault_candidate(model, le, features_path, split=DEFAULT_SPLIT)

    f1_macro = float(metrics["f1_macro"])
    balanced_accuracy = float(metrics["balanced_accuracy"])
    report = evaluate_gate(
        gate or fault_gate(min_f1_macro, min_balanced_accuracy),
        {"f1_macro": f1_macro, "balanced_accuracy": balanced_accuracy},
        model_name=model_name,
    )
    decision = "promoted" if report.status == "PASS" else "rejected"
    reason = _build_fault_reason(
        f1_macro, min_f1_macro, balanced_accuracy, min_balanced_accuracy
    )

    if decision == "promoted":
        client.transition_model_version_stage(
            name=model_name,
            version=str(candidate_version),
            stage="Production",
        )

    _attach_manifest(client, model_name, candidate_version, metrics, report, features_path)

    with mlflow.start_run(run_name="promote_fault"):
        mlflow.log_params({
            "candidate_version": str(candidate_version),
            "decision": decision,
            "reason": reason,
            "gate_status": report.status,
        })
        mlflow.log_metrics(_numeric_metrics(metrics))

    return {
        "candidate_version": candidate_version,
        "decision": decision,
        "reason": reason,
        "metrics": metrics,
        "gate": report,
    }
