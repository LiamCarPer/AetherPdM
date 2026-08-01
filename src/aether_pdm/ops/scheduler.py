"""
Scheduled ops pipeline: the autonomous monitoring loop.

Runs, in order:
1. BATCH SCORE — score all assets (org-scoped) with hysteresis + cooldown rules.
2. DRIFT CHECK — compare train (reference) vs test (production) feature distributions.
3. RETRAIN DECISION — if drift fired, retrain anomaly + fault models.
4. PROMOTE — run promotion gates; new models promoted only if metrics pass
   (previous production stays active if rejected = implicit rollback).

This is the capstone orchestration. It does NOT re-implement batch scoring,
drift detection, retraining, or promotion — it calls the existing modules.

Cron-compatible: exits 0 on success, non-zero on failure. Designed to be run
by cron or a scheduled container (see docs/scheduling.md).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from aether_pdm.db.database import get_session, init_db
from aether_pdm.ops.batch_scorer import BatchScorer
from aether_pdm.ops.drift import detect_drift
from aether_pdm.ops.retrain import run_retrain_pipeline

logger = logging.getLogger(__name__)

DEFAULT_FEATURES_PATH = Path("data/interim/features/features_v1.parquet")
_DEFAULT_HYSTERESIS = 3
_DEFAULT_COOLDOWN_MIN = 30
_DEFAULT_DRIFT_THRESHOLD = 0.25
_DEFAULT_BATCH_LIMIT = 100


def run_scheduled_pipeline(
    features_path: Path,
    org: str | None = None,
    mlflow_uri: str | None = None,
    hysteresis: int = 3,
    cooldown_min: int = 30,
    drift_threshold: float = 0.25,
    retrain: bool = True,
    batch_limit: int = 100,
    **retrain_kwargs: Any,
) -> dict[str, Any]:
    """
    Run the full scheduled ops loop.

    Steps:
    1. init_db()
    2. Batch score: with get_session() as db:
         scorer = BatchScorer(
             mlflow_uri=mlflow_uri, hysteresis=hysteresis, cooldown_min=cooldown_min
         )
         batch = scorer.run(db, org=org, limit=batch_limit)
    3. If features_path exists:
         drift = detect_drift(features_path)  # train vs test
       else:
         drift = {"error": "features file not found"} (non-fatal, batch still ran)
     4. Retrain stage (delegated entirely to run_retrain_pipeline when
        retrain=True AND drift was computed successfully). Its internal
        should_retrain() honors drift_threshold — not the drift_fired hardcode
        in detect_drift (0.25) — so a custom threshold below 0.25 still
        triggers retraining. When there is no drift it returns
        {"skipped": True, "skip_reason": ...} instead.
        Else (retrain disabled or drift errored):
          retrain_result = {"skipped": True, "reason": "no_drift_or_retrain_disabled"}

    Returns dict:
    - batch (dict from BatchScorer.run)
    - drift (dict from detect_drift or error dict)
    - retrain (dict from run_retrain_pipeline or skipped dict)
    - summary (dict): assets_scored, alerts_raised, drift_fired, retrained, promoted

    Args:
        features_path: Path to the features Parquet used for drift + retrain.
        org: Only score assets belonging to this org. ``None`` scores all
            tenants (admin operation).
        mlflow_uri: MLflow tracking URI (defaults to the BatchScorer default).
        hysteresis: Consecutive non-healthy scores required before an alert.
        cooldown_min: Minutes to suppress re-alerting the same asset+level.
        drift_threshold: Mean PSI at/above which drift forces retraining.
        retrain: Whether the retrain/promote stage may run at all.
        batch_limit: Max assets scored per batch run.
        **retrain_kwargs: Passed through to run_retrain_pipeline
            (e.g. ``force=True`` to retrain even without drift).

    Returns:
        Pipeline result dict with ``batch``, ``drift``, ``retrain`` and
        ``summary`` keys.
    """
    features_path = Path(features_path)

    # 1. Batch score: score all assets (org-scoped) with alert rules.
    init_db()
    with get_session() as db:
        scorer = BatchScorer(
            mlflow_uri=mlflow_uri,
            hysteresis=hysteresis,
            cooldown_min=cooldown_min,
        )
        batch = scorer.run(db, org=org, limit=batch_limit)

    # 2. Drift check: train (reference) vs test (production) distributions.
    #    A missing or unreadable features file is non-fatal — the batch still
    #    ran and was persisted; drift is recorded as an error dict instead.
    if features_path.exists():
        try:
            drift = detect_drift(features_path)
        except Exception as exc:  # noqa: BLE001 - drift failure is non-fatal
            logger.error("Drift detection failed: %s", exc)
            drift = {"error": str(exc)}
    else:
        logger.warning(
            "Features file not found: %s (drift check + retrain skipped)", features_path
        )
        drift = {"error": "features file not found"}

    # 3 + 4. Retrain decision + promote.
    #    Delegated entirely to run_retrain_pipeline when retrain=True and drift
    #    was computed successfully: its internal should_retrain() honors the
    #    user-supplied drift_threshold (e.g. mean_psi >= 0.20 even when the
    #    detect_drift drift_fired hardcode is 0.25), so no local drift_fired
    #    gate is duplicated here. force= in retrain_kwargs is forwarded too.
    #    When retrain is disabled or drift errored (features file missing or
    #    unreadable), run_retrain_pipeline cannot be invoked safely; skip.
    drift_fired = bool(drift.get("drift_fired", False))
    if retrain and "error" not in drift:
        retrain_result = run_retrain_pipeline(
            features_path,
            mlflow_uri=mlflow_uri,
            drift_threshold=drift_threshold,
            **retrain_kwargs,
        )
    else:
        retrain_result = {"skipped": True, "reason": "no_drift_or_retrain_disabled"}

    retrained = not bool(retrain_result.get("skipped", True))
    summary = {
        "assets_scored": int(batch.get("scored", 0)),
        "alerts_raised": int(batch.get("alerts_raised", 0)),
        "drift_fired": drift_fired,
        "retrained": retrained,
        "promoted": bool(retrain_result.get("outcome") == "promoted"),
    }

    return {
        "batch": batch,
        "drift": drift,
        "retrain": retrain_result,
        "summary": summary,
    }


def main() -> None:
    """CLI entrypoint.

    Usage:
        python -m aether_pdm.ops.scheduler [--features data/interim/features/features_v1.parquet]
            [--org acme] [--mlflow-uri sqlite:///mlflow.db]
            [--hysteresis 3] [--cooldown-min 30] [--no-retrain]

    Exits 0 on success, 1 on failure (cron-safe).
    """
    parser = argparse.ArgumentParser(
        prog="aether_pdm.ops.scheduler",
        description=(
            "Run the scheduled ops pipeline: "
            "batch score -> drift check -> retrain -> promote."
        ),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to the features Parquet (default: %(default)s)",
    )
    parser.add_argument(
        "--org",
        default=None,
        help="Only score assets belonging to this org (default: all tenants)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=os.environ.get("AETHER_MLFLOW_TRACKING_URI"),
        help=(
            "MLflow tracking URI (default: $AETHER_MLFLOW_TRACKING_URI "
            "or sqlite:///mlflow.db)"
        ),
    )
    parser.add_argument(
        "--hysteresis",
        type=int,
        default=_DEFAULT_HYSTERESIS,
        help="Consecutive non-healthy scores required before an alert",
    )
    parser.add_argument(
        "--cooldown-min",
        type=int,
        default=_DEFAULT_COOLDOWN_MIN,
        help="Minutes to suppress re-alerting the same asset+level",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=_DEFAULT_DRIFT_THRESHOLD,
        help="Mean PSI at/above which drift forces retraining",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip the retrain/promote stage even if drift fired",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=_DEFAULT_BATCH_LIMIT,
        help="Max assets scored per batch run",
    )
    args = parser.parse_args()

    try:
        result = run_scheduled_pipeline(
            features_path=args.features,
            org=args.org,
            mlflow_uri=args.mlflow_uri,
            hysteresis=args.hysteresis,
            cooldown_min=args.cooldown_min,
            drift_threshold=args.drift_threshold,
            retrain=not args.no_retrain,
            batch_limit=args.batch_limit,
        )
    except Exception as exc:  # noqa: BLE001 - cron-compatible failure exit
        print(f"ERROR: scheduled ops pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = result["summary"]
    print("=" * 64)
    print("Scheduled ops pipeline complete")
    print(f"  assets_scored : {summary['assets_scored']}")
    print(f"  alerts_raised : {summary['alerts_raised']}")
    print(f"  drift_fired   : {summary['drift_fired']}")
    print(f"  retrained     : {summary['retrained']}")
    print(f"  promoted      : {summary['promoted']}")
    if "error" in result["drift"]:
        print(f"  drift_error   : {result['drift']['error']}")
    retrain_result = result["retrain"]
    if retrain_result.get("skipped"):
        reason = retrain_result.get("skip_reason") or retrain_result.get("reason")
        print(f"  retrain       : skipped ({reason})")
    else:
        print(f"  retrain       : outcome={retrain_result.get('outcome')}")
    print("=" * 64)
    sys.exit(0)


if __name__ == "__main__":
    main()
