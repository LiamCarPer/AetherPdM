"""Repository layer for database operations."""

import json
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from aether_pdm.db.models import Alert, Asset, ScoreRecord

# --- Assets ---


def get_asset(db: Session, asset_id: str) -> Asset | None:
    return db.query(Asset).filter(Asset.asset_id == asset_id).first()


def list_assets(db: Session, org: str | None = None) -> list[Asset]:
    q = db.query(Asset)
    if org:
        q = q.filter(Asset.org == org)
    return q.all()


def upsert_asset(db: Session, asset_id: str, **kwargs: Any) -> Asset:
    asset = get_asset(db, asset_id)
    if asset:
        for k, v in kwargs.items():
            setattr(asset, k, v)
    else:
        asset = Asset(asset_id=asset_id, **kwargs)
        db.add(asset)
    db.flush()
    return asset


# --- Scores ---


def save_score(db: Session, asset_id: str, score_result: dict[str, Any]) -> ScoreRecord:
    top_features_json = json.dumps(score_result.get("top_features", []))
    fault = score_result.get("fault") or {}
    record = ScoreRecord(
        asset_id=asset_id,
        health_score=score_result["health_score"],
        anomaly_score=score_result["anomaly_score"],
        fault_class=fault.get("class"),
        fault_confidence=fault.get("confidence"),
        model_version_anomaly=score_result.get("model_versions", {}).get("anomaly"),
        model_version_fault=score_result.get("model_versions", {}).get("fault"),
        window_size=score_result.get("window_size"),
        top_features=top_features_json,
    )
    db.add(record)
    db.flush()
    return record


def get_latest_score(db: Session, asset_id: str) -> ScoreRecord | None:
    return (
        db.query(ScoreRecord)
        .filter(ScoreRecord.asset_id == asset_id)
        .order_by(desc(ScoreRecord.created_at))
        .first()
    )


def list_scores(db: Session, asset_id: str | None = None, limit: int = 50) -> list[ScoreRecord]:
    q = db.query(ScoreRecord)
    if asset_id:
        q = q.filter(ScoreRecord.asset_id == asset_id)
    return q.order_by(desc(ScoreRecord.created_at)).limit(limit).all()


# --- Alerts ---


def save_alert(
    db: Session,
    asset_id: str,
    level: str,
    reason: str | None,
    health_score: float,
    fault_class: str | None = None,
) -> Alert:
    alert = Alert(
        asset_id=asset_id,
        level=level,
        reason=reason,
        health_score=health_score,
        fault_class=fault_class,
    )
    db.add(alert)
    db.flush()
    return alert


def list_alerts(
    db: Session,
    asset_id: str | None = None,
    level: str | None = None,
    limit: int = 50,
) -> list[Alert]:
    q = db.query(Alert)
    if asset_id:
        q = q.filter(Alert.asset_id == asset_id)
    if level:
        q = q.filter(Alert.level == level)
    return q.order_by(desc(Alert.created_at)).limit(limit).all()


def acknowledge_alert(db: Session, alert_id: int) -> Alert | None:
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if alert:
        alert.acknowledged = 1  # type: ignore[assignment]
        db.flush()
    return alert
