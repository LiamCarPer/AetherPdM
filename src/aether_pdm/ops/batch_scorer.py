"""
Batch scoring + alert rules engine.

Scores every registered asset (or a subset) on a schedule, applies
production alert rules, and persists scores + alerts.

Alert rules:
1. HYSTERESIS — require N consecutive non-healthy scores before raising
   an alert (suppresses transient blips).
2. COOLDOWN — do not re-alert the same asset/level within T minutes of
   the previous alert of that level (prevents alert fatigue).

Usage (library):
    from aether_pdm.ops.batch_scorer import BatchScorer
    scorer = BatchScorer(mlflow_uri="sqlite:///mlflow.db")
    results = scorer.run(org="acme")

Usage (CLI):
    uv run python scripts/run_batch_scorer.py --org acme --hysteresis 3 --cooldown-min 30
"""

import argparse
from datetime import UTC, datetime, timedelta
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from aether_pdm.data.synthetic import synthetic_waveform
from aether_pdm.db.database import get_session, init_db
from aether_pdm.db.models import Asset, ScoreRecord
from aether_pdm.db.repository import (
    list_alerts,
    list_assets,
    list_scores,
    save_alert,
    save_score,
)
from aether_pdm.serve.inference import InferenceEngine

DEFAULT_HYSTERESIS = 3  # consecutive non-healthy scores before alert
DEFAULT_COOLDOWN_MIN = 30  # minutes between alerts of the same level per asset
DEFAULT_LIMIT = 100  # max assets per batch run

_HEALTHY_THRESHOLD = 0.5  # health_score < threshold counts as "non-healthy"
_SYNTH_WAVEFORM_LENGTH = 4096
_DEFAULT_RPM = 1772.0
_DEFAULT_SAMPLING_RATE = 12000.0


def _ensure_aware(value: datetime | None) -> datetime | None:
    """Coerce a datetime to timezone-aware UTC.

    SQLAlchemy's SQLite dialect drops the timezone when round-tripping
    ``DateTime`` columns, so values read from the DB may come back naive.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class BatchScorer:
    """
    Scores registered assets and applies alert rules.

    Parameters
    ----------
    mlflow_uri : str | None — MLflow tracking URI for the inference engine.
    engine : InferenceEngine | None — optional pre-built engine (injectable for tests).
    hysteresis : int — consecutive non-healthy scores required before alerting (>= 1).
    cooldown_min : int — minutes to suppress re-alerting the same asset+level.
    """

    def __init__(
        self,
        mlflow_uri: str | None = None,
        engine: InferenceEngine | None = None,
        hysteresis: int = DEFAULT_HYSTERESIS,
        cooldown_min: int = DEFAULT_COOLDOWN_MIN,
    ) -> None:
        if engine is not None:
            self.engine = engine
        else:
            self.engine = InferenceEngine(mlflow_uri=mlflow_uri or "sqlite:///mlflow.db")
        self.hysteresis = max(1, hysteresis)
        self.cooldown_min = max(0, cooldown_min)

    def score_asset(
        self,
        db: Session,
        asset: Asset,
        waveform: NDArray[np.float64] | None = None,
    ) -> dict:
        """
        Score a single asset.

        If waveform is None, synthesize a diagnostic waveform from the asset's
        nominal RPM using aether_pdm.data.synthetic.synthetic_waveform
        (length=4096, rpm=asset.rpm_nominal or 1772, fault_type="normal").

        Returns the engine.score() result dict.
        Raises RuntimeError if engine has no models loaded.
        """
        if not getattr(self.engine, "model_available", False):
            raise RuntimeError(
                "Models not loaded. Train models first or check MLflow connection."
            )
        rpm = float(asset.rpm_nominal or _DEFAULT_RPM)
        if waveform is None:
            waveform = synthetic_waveform(
                length=_SYNTH_WAVEFORM_LENGTH,
                rpm=rpm,
                fault_type="normal",
            )
        sampling_rate = float(asset.sampling_rate or _DEFAULT_SAMPLING_RATE)
        return self.engine.score(
            waveform=waveform,
            sampling_rate=sampling_rate,
            rpm=rpm,
        )

    def _recent_scores(self, db: Session, asset_id: str, n: int) -> list[ScoreRecord]:
        """Fetch the most recent n score records for an asset (oldest→newest).

        ``list_scores`` orders by ``created_at DESC`` with no tiebreaker, so
        records sharing an identical timestamp could be returned in arbitrary
        order. Sorting by ``(created_at, id)`` guarantees deterministic,
        insertion-stable ordering even for timestamp ties, ensuring the
        just-persisted record is always included in the last-N window.
        """
        records = list_scores(db, asset_id=asset_id, limit=n)
        records.sort(key=lambda r: (r.created_at, r.id))
        return records

    def _hysteresis_satisfied(
        self,
        db: Session,
        asset_id: str,
        new_health_score: float,
        threshold: float = _HEALTHY_THRESHOLD,
    ) -> bool:
        """
        Hysteresis rule: require self.hysteresis CONSECUTIVE non-healthy
        scores (health_score < threshold) including the current one.

        Uses recent score records (including the just-persisted one).
        Returns True if the last `hysteresis` scores are all non-healthy.
        """
        if new_health_score >= threshold:
            return False
        recent = self._recent_scores(db, asset_id, self.hysteresis)
        if len(recent) < self.hysteresis:
            return False
        return all(record.health_score < threshold for record in recent)

    def _cooldown_active(
        self,
        db: Session,
        asset_id: str,
        level: str,
        now: datetime | None = None,
    ) -> bool:
        """
        Cooldown rule: suppress a new alert of `level` for this asset if a
        prior alert of the same level was created within cooldown_min minutes.

        now: datetime (UTC) — injectable for tests.
        """
        if now is None:
            now = datetime.now(UTC)
        now_aware = _ensure_aware(now)
        if now_aware is None:
            return False
        if self.cooldown_min <= 0:
            return False
        alerts = list_alerts(db, asset_id=asset_id, level=level, limit=1)
        if not alerts:
            return False
        created_at = _ensure_aware(cast(datetime | None, alerts[0].created_at))
        if created_at is None:
            return False
        cutoff = now_aware - timedelta(minutes=self.cooldown_min)
        return created_at > cutoff

    def run(
        self,
        db: Session,
        org: str | None = None,
        limit: int = DEFAULT_LIMIT,
        now: datetime | None = None,
    ) -> dict:
        """
        Score all assets (optionally org-scoped) and apply alert rules.

        For each asset:
        1. score_asset()
        2. save_score(db, asset.asset_id, result)
        3. Determine alert level from result["alert"]["level"]
        4. If level != "healthy":
           a. Check hysteresis: _hysteresis_satisfied(...)
           b. Check cooldown: _cooldown_active(...)
           c. If hysteresis satisfied AND not in cooldown: save_alert(...)
           d. If in cooldown: skip (record in suppressed list)
        5. If level == "healthy": reset (no alert; consecutive counter implicitly resets
           because hysteresis checks the last N scores)

        Returns dict:
        - scored (int)
        - alerts_raised (int)
        - alerts_suppressed_by_cooldown (int)
        - alerts_suppressed_by_hysteresis (int)   # non-healthy but not enough consecutive
        - errors (list[str])                      # per-asset errors (e.g. engine 503)
        - results (list[dict])                    # per-asset: asset_id,
                                                  #   health_score, alert_level, alert_raised
        """
        assets = list_assets(db, org=org)
        if limit and limit > 0:
            assets = assets[:limit]

        scored = 0
        alerts_raised = 0
        alerts_suppressed_by_cooldown = 0
        alerts_suppressed_by_hysteresis = 0
        errors: list[str] = []
        results: list[dict] = []

        for asset in assets:
            entry: dict = {
                "asset_id": asset.asset_id,
                "health_score": None,
                "alert_level": None,
                "alert_raised": False,
            }
            try:
                result = self.score_asset(db, asset)
                asset_id = str(asset.asset_id)
                save_score(db, asset_id, result)

                alert = result.get("alert") or {}
                level = str(alert.get("level", "healthy"))
                health_score = float(result.get("health_score", 1.0))
                entry["health_score"] = health_score
                entry["alert_level"] = level
                scored += 1

                if level != "healthy":
                    if not self._hysteresis_satisfied(db, asset_id, health_score):
                        entry["suppressed_by"] = "hysteresis"
                        alerts_suppressed_by_hysteresis += 1
                    elif self._cooldown_active(db, asset_id, level, now=now):
                        entry["suppressed_by"] = "cooldown"
                        alerts_suppressed_by_cooldown += 1
                    else:
                        fault = result.get("fault") or {}
                        save_alert(
                            db,
                            asset_id,
                            level,
                            alert.get("reason"),
                            health_score,
                            fault.get("class"),
                        )
                        alerts_raised += 1
                        entry["alert_raised"] = True
            except Exception as exc:  # noqa: BLE001 - per-asset isolation, never abort batch
                entry["error"] = str(exc)
                errors.append(f"{asset.asset_id}: {exc}")
            results.append(entry)

        return {
            "scored": scored,
            "alerts_raised": alerts_raised,
            "alerts_suppressed_by_cooldown": alerts_suppressed_by_cooldown,
            "alerts_suppressed_by_hysteresis": alerts_suppressed_by_hysteresis,
            "errors": errors,
            "results": results,
        }


def main() -> None:
    """CLI: score assets and apply alert rules.

    Usage:
        python -m aether_pdm.ops.batch_scorer \
            [--org acme] [--hysteresis 3] [--cooldown-min 30] [--limit 100]
    """
    parser = argparse.ArgumentParser(description="Score assets and apply alert rules")
    parser.add_argument(
        "--org",
        default=None,
        help="Only score assets belonging to this org (default: all)",
    )
    parser.add_argument(
        "--hysteresis",
        type=int,
        default=DEFAULT_HYSTERESIS,
        help="Consecutive non-healthy scores required before alert",
    )
    parser.add_argument(
        "--cooldown-min",
        type=int,
        default=DEFAULT_COOLDOWN_MIN,
        help="Minutes to suppress re-alerting the same asset+level",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max assets per batch run",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="MLflow tracking URI (default: sqlite:///mlflow.db)",
    )
    args = parser.parse_args()

    init_db()
    with get_session() as session:
        scorer = BatchScorer(
            mlflow_uri=args.mlflow_uri,
            hysteresis=args.hysteresis,
            cooldown_min=args.cooldown_min,
        )
        summary = scorer.run(session, org=args.org, limit=args.limit)

    print(
        f"Batch scoring complete: scored={summary['scored']} "
        f"alerts_raised={summary['alerts_raised']}"
    )
    print(
        "  suppressed_by_hysteresis="
        f"{summary['alerts_suppressed_by_hysteresis']} "
        "suppressed_by_cooldown="
        f"{summary['alerts_suppressed_by_cooldown']}"
    )
    if summary["errors"]:
        print(f"  errors={len(summary['errors'])}")
        for err in summary["errors"]:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
