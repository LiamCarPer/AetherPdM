"""
Domain shift evaluation: train on CWRU (source), evaluate on Paderborn (target).

Quantifies how well a CWRU-trained anomaly/fault model transfers to real
Paderborn bearing data. Produces metrics comparing source vs target performance
and a drift report between the two feature distributions.

Models are loaded from the MLflow registry (staging, else latest) and evaluated
with :mod:`aether_pdm.ops.promote` on both the CWRU source split and the
Paderborn target split. Missing models degrade gracefully: the study returns a
``models_missing`` flag plus notes instead of crashing.

Usage (via scripts/run_domain_shift.py):
    python -m aether_pdm.ops.domain_shift \
        --cwru-features data/interim/cwru/features_v1.parquet \
        --paderborn-features data/interim/paderborn/features_v1.parquet \
        --output reports/domain-shift.md
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import aether_pdm.ops.promote as promote_mod
from aether_pdm.ops.drift import feature_drift_report

ANOMALY_MODEL_NAME = "aether-anomaly"
FAULT_MODEL_NAME = "aether-fault-clf"

# Splits used for source (CWRU) and target (Paderborn) evaluation.
SOURCE_SPLIT = "test"
TARGET_SPLIT = "test"

# Project-wide MLflow tracking URI (see serve/app.py, serve/inference.py).
DEFAULT_MLFLOW_URI = "sqlite:///mlflow.db"

# Evaluation keys tracked for the performance delta summary.
_ANOMALY_METRIC_KEYS = ("false_alarm_rate", "detection_rate", "best_detection_rate")
_FAULT_METRIC_KEYS = ("f1_macro", "balanced_accuracy")


def evaluate_on_target(
    model: Any,
    features_path: Path,
    target: str,
    labels: list[str] | LabelEncoder | None = None,
) -> dict:
    """Evaluate a model on a target domain's features.

    Anomaly models (``labels=None``) are evaluated with
    :func:`aether_pdm.ops.promote.evaluate_anomaly_candidate`. Fault models
    (``labels`` = class list or a fitted ``LabelEncoder``) are evaluated with
    :func:`aether_pdm.ops.promote.evaluate_fault_candidate`.

    Evaluation errors (empty split, single-class target, etc.) are captured
    into a dict with an ``error`` key so the study report can display ``N/A``
    instead of crashing.

    Args:
        model: Trained sklearn model (anomaly or fault classifier).
        features_path: Parquet features file with a ``split`` column.
        target: Split label to evaluate on (e.g. ``"test"``).
        labels: For fault models — the class list or fitted ``LabelEncoder``.
            ``None`` selects anomaly-model evaluation.

    Returns:
        Metrics dict from the underlying evaluator, or
        ``{"error": <message>, "split": target}`` when evaluation is impossible.
    """
    try:
        if labels is not None:
            le = labels if isinstance(labels, LabelEncoder) else LabelEncoder().fit(labels)
            return promote_mod.evaluate_fault_candidate(model, le, features_path, split=target)
        return promote_mod.evaluate_anomaly_candidate(model, features_path, split=target)
    except ValueError as exc:
        return {"error": str(exc), "split": target}


def compute_domain_shift(cwru_features: Path, paderborn_features: Path) -> dict:
    """Compute domain shift metrics between CWRU and Paderborn feature distributions.

    Uses :func:`aether_pdm.ops.drift.feature_drift_report` with CWRU as the
    reference distribution and Paderborn as the production distribution.

    Args:
        cwru_features: CWRU (source) features Parquet.
        paderborn_features: Paderborn (target) features Parquet.

    Returns:
        Dict with ``feature_drift_report`` (DataFrame), ``n_features_drifted``,
        ``mean_psi``, ``worst_feature`` and ``worst_psi``.
    """
    cwru_df = pd.read_parquet(Path(cwru_features))
    paderborn_df = pd.read_parquet(Path(paderborn_features))

    report = feature_drift_report(reference_df=cwru_df, production_df=paderborn_df)

    if report.empty:
        return {
            "feature_drift_report": report,
            "n_features_drifted": 0,
            "mean_psi": 0.0,
            "worst_feature": None,
            "worst_psi": 0.0,
        }

    n_features_drifted = int((report["status"] == "severe").sum())
    mean_psi = float(report["psi"].mean())
    worst_idx = int(report["psi"].idxmax())
    worst_feature = str(report.loc[worst_idx, "feature"])
    worst_psi = float(report.loc[worst_idx, "psi"])

    return {
        "feature_drift_report": report,
        "n_features_drifted": n_features_drifted,
        "mean_psi": mean_psi,
        "worst_feature": worst_feature,
        "worst_psi": worst_psi,
    }


def _performance_delta(source: dict, target: dict, metric_keys: tuple[str, ...]) -> dict:
    """Target-minus-source delta per metric; ``None`` when either side errored."""
    deltas: dict[str, float | None] = {}
    for key in metric_keys:
        if (
            isinstance(source, dict)
            and isinstance(target, dict)
            and "error" not in source
            and "error" not in target
            and key in source
            and key in target
        ):
            deltas[key] = float(target[key]) - float(source[key])
        else:
            deltas[key] = None
    return deltas


def run_domain_shift_study(
    cwru_features: Path,
    paderborn_features: Path,
    mlflow_uri: str | None = None,
) -> dict:
    """Run the full CWRU -> Paderborn domain shift study.

    Steps:
    1. Load the production anomaly + fault models from MLflow (staging, else latest).
    2. Evaluate them on CWRU (source) features.
    3. Evaluate them on Paderborn (target) features.
    4. Compute feature drift (CWRU reference vs Paderborn production).
    5. Assemble the report data with a performance-delta summary.

    Missing models are not fatal: the returned dict contains
    ``models_missing=True`` and a ``notes`` list describing what could not be
    loaded, allowing the report to document that models must be trained first.

    Args:
        cwru_features: CWRU (source) features Parquet.
        paderborn_features: Paderborn (target) features Parquet.
        mlflow_uri: MLflow tracking URI (default ``sqlite:///mlflow.db``).

    Returns:
        Dict with ``anomaly``/``fault`` source+target metrics, ``drift``,
        ``performance_delta``, ``models_missing`` and ``notes``.
    """
    mlflow.set_tracking_uri(mlflow_uri or DEFAULT_MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    notes: list[str] = []
    anomaly_model: Any = None
    fault_model: Any = None
    fault_le: LabelEncoder | None = None

    try:
        anomaly_model, _ = promote_mod._load_candidate_model(ANOMALY_MODEL_NAME, client)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully for the report
        notes.append(f"Anomaly model unavailable: {exc}")

    try:
        fault_model, fault_version = promote_mod._load_candidate_model(FAULT_MODEL_NAME, client)
        fault_le = promote_mod._load_fault_label_encoder(client, FAULT_MODEL_NAME, fault_version)
    except Exception as exc:  # noqa: BLE001
        # Atomic: without the label encoder the fault model cannot be evaluated,
        # so treat the whole fault side as unavailable rather than letting the
        # model fall through to the anomaly evaluator.
        fault_model = None
        fault_le = None
        notes.append(f"Fault model unavailable: {exc}")

    models_missing = anomaly_model is None and fault_model is None

    anomaly_source = (
        evaluate_on_target(anomaly_model, cwru_features, SOURCE_SPLIT)
        if anomaly_model is not None
        else {"error": "anomaly model not loaded", "split": SOURCE_SPLIT}
    )
    anomaly_target = (
        evaluate_on_target(anomaly_model, paderborn_features, TARGET_SPLIT)
        if anomaly_model is not None
        else {"error": "anomaly model not loaded", "split": TARGET_SPLIT}
    )
    fault_source = (
        evaluate_on_target(fault_model, cwru_features, SOURCE_SPLIT, labels=fault_le)
        if fault_model is not None
        else {"error": "fault model not loaded", "split": SOURCE_SPLIT}
    )
    fault_target = (
        evaluate_on_target(fault_model, paderborn_features, TARGET_SPLIT, labels=fault_le)
        if fault_model is not None
        else {"error": "fault model not loaded", "split": TARGET_SPLIT}
    )

    drift = compute_domain_shift(cwru_features, paderborn_features)

    return {
        "source_features": str(cwru_features),
        "target_features": str(paderborn_features),
        "models_missing": models_missing,
        "notes": notes,
        "anomaly": {
            "source": anomaly_source,
            "target": anomaly_target,
            "delta": _performance_delta(anomaly_source, anomaly_target, _ANOMALY_METRIC_KEYS),
        },
        "fault": {
            "source": fault_source,
            "target": fault_target,
            "delta": _performance_delta(fault_source, fault_target, _FAULT_METRIC_KEYS),
        },
        "drift": drift,
        "performance_delta": {
            "anomaly": _performance_delta(anomaly_source, anomaly_target, _ANOMALY_METRIC_KEYS),
            "fault": _performance_delta(fault_source, fault_target, _FAULT_METRIC_KEYS),
        },
    }


def _metric_table(label: str, result: dict) -> str:
    """Render a source-vs-target metric table for one model family."""
    source = result.get("source", {})
    target = result.get("target", {})

    keys: tuple[str, ...]
    if label == "Anomaly":
        keys = _ANOMALY_METRIC_KEYS
    else:
        keys = _FAULT_METRIC_KEYS

    rows = ["| Metric | Source (CWRU) | Target (Paderborn) | Delta |", "|---|---|---|---|"]
    for key in keys:
        src_val = f"{source[key]:.4f}" if isinstance(source, dict) and key in source else "N/A"
        tgt_val = f"{target[key]:.4f}" if isinstance(target, dict) and key in target else "N/A"
        delta = result.get("delta", {}).get(key)
        delta_val = f"{delta:+.4f}" if delta is not None else "N/A"
        rows.append(f"| {key} | {src_val} | {tgt_val} | {delta_val} |")

    errors = []
    if isinstance(source, dict) and "error" in source:
        errors.append(f"- Source: {source['error']}")
    if isinstance(target, dict) and "error" in target:
        errors.append(f"- Target: {target['error']}")
    if errors:
        rows.append("")
        rows.append("**Notes:**")
        rows.extend(errors)
    return "\n".join(rows)


def _drift_table(drift: dict, top_n: int = 10) -> str:
    """Render the top-N PSI feature drift table."""
    report = drift.get("feature_drift_report")
    if not isinstance(report, pd.DataFrame) or report.empty:
        return "_No common numeric features between source and target._"

    top = report.sort_values("psi", ascending=False).head(top_n)
    rows = ["| Feature | PSI | KS stat | Status |", "|---|---|---|---|"]
    for _, row in top.iterrows():
        rows.append(
            f"| {row['feature']} | {float(row['psi']):.4f} "
            f"| {float(row['ks_statistic']):.4f} | {row['status']} |"
        )
    return "\n".join(rows)


def _interpretation(result: dict) -> str:
    """Auto-generate a prose interpretation from drift + performance deltas."""
    paragraphs: list[str] = []

    drift = result.get("drift", {})
    mean_psi = float(drift.get("mean_psi", 0.0))
    n_drifted = int(drift.get("n_features_drifted", 0))
    worst = drift.get("worst_feature")
    worst_psi = float(drift.get("worst_psi", 0.0))

    if mean_psi >= 0.25:
        paragraphs.append(
            f"Severe feature drift detected (mean PSI = {mean_psi:.3f}, "
            f"{n_drifted} feature(s) severe). CWRU-trained models are unlikely to "
            "transfer to Paderborn without retraining or domain adaptation."
        )
    elif mean_psi >= 0.10:
        paragraphs.append(
            f"Moderate feature drift detected (mean PSI = {mean_psi:.3f}). "
            "Investigate the top drifting features before relying on source-domain thresholds."
        )
    else:
        paragraphs.append(
            f"No significant feature drift detected (mean PSI = {mean_psi:.3f}). "
            "CWRU and Paderborn feature distributions are broadly aligned."
        )

    if worst is not None:
        paragraphs.append(
            f"The worst-drifting feature is **{worst}** (PSI = {worst_psi:.3f})."
        )

    if not result.get("models_missing", True):
        fault_delta = result.get("performance_delta", {}).get("fault", {})
        anomaly_delta = result.get("performance_delta", {}).get("anomaly", {})

        f1_delta = fault_delta.get("f1_macro")
        dr_delta = anomaly_delta.get("detection_rate")
        far_delta = anomaly_delta.get("false_alarm_rate")

        if f1_delta is not None and f1_delta < -0.10:
            paragraphs.append(
                f"Fault classifier macro-F1 degrades by {abs(f1_delta):.3f} on Paderborn, "
                "confirming that sensor/system differences matter beyond raw feature drift."
            )
        elif f1_delta is not None and f1_delta > 0.10:
            paragraphs.append(
                f"Fault classifier macro-F1 improves by {f1_delta:.3f} on Paderborn, "
                "suggesting the source model generalizes well to the target domain."
            )

        if dr_delta is not None and far_delta is not None:
            paragraphs.append(
                f"Anomaly detection rate changes by {dr_delta:+.3f} and false-alarm rate "
                f"by {far_delta:+.3f} on Paderborn."
            )
    else:
        paragraphs.append(
            "No trained models were found in the MLflow registry; train and register "
            "`aether-anomaly` and `aether-fault-clf` before interpreting performance deltas."
        )

    return "\n\n".join(paragraphs)


def write_domain_shift_report(result: dict, output_path: Path) -> Path:
    """Write the domain shift study as a Markdown report.

    Includes a summary table (source vs target metrics), the performance delta,
    a top-10 feature drift table by PSI, and an auto-generated interpretation.

    Args:
        result: Output of :func:`run_domain_shift_study`.
        output_path: Destination Markdown path (parent dirs created).

    Returns:
        The ``output_path`` that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    drift = result.get("drift", {})
    mean_psi = float(drift.get("mean_psi", 0.0))
    n_drifted = int(drift.get("n_features_drifted", 0))
    worst = drift.get("worst_feature") or "N/A"
    worst_psi = float(drift.get("worst_psi", 0.0))

    lines = [
        "# Domain Shift Report: CWRU to Paderborn",
        "",
        f"- **Source features**: `{result.get('source_features', 'N/A')}`",
        f"- **Target features**: `{result.get('target_features', 'N/A')}`",
        f"- **Generated**: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- **Models loaded**: {'no' if result.get('models_missing', True) else 'yes'}",
        "",
        "## Summary",
        "",
        f"- **Mean PSI**: {mean_psi:.3f}",
        f"- **Severely drifted features**: {n_drifted}",
        f"- **Worst feature**: {worst} (PSI = {worst_psi:.3f})",
        "",
        "## Anomaly Model Performance",
        "",
        _metric_table("Anomaly", result.get("anomaly", {})),
        "",
        "## Fault Classifier Performance",
        "",
        _metric_table("Fault", result.get("fault", {})),
        "",
        "## Feature Drift (top 10 by PSI)",
        "",
        _drift_table(drift),
        "",
        "## Interpretation",
        "",
        _interpretation(result),
        "",
    ]

    if result.get("notes"):
        lines.append("## Notes")
        lines.append("")
        for note in result["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CWRU -> Paderborn domain shift study")
    parser.add_argument("--cwru-features", type=Path, required=True)
    parser.add_argument("--paderborn-features", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/domain-shift-cwru-to-paderborn.md"),
    )
    parser.add_argument("--mlflow-uri", type=str, default=None)
    args = parser.parse_args()

    result = run_domain_shift_study(
        args.cwru_features,
        args.paderborn_features,
        mlflow_uri=args.mlflow_uri,
    )
    write_domain_shift_report(result, args.output)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
