"""Repository layer for database operations."""

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from aether_pdm.db.models import Alert, ApiKey, Asset, Organization, Plant, ScoreRecord

# --- Organizations ---


def get_organization(db: Session, org_id: str) -> Organization | None:
    """Fetch an organization by its org_id (e.g. "acme")."""
    return db.query(Organization).filter(Organization.org_id == org_id).first()


def upsert_organization(db: Session, org_id: str, name: str | None = None) -> Organization:
    """Create or update an organization."""
    org = get_organization(db, org_id)
    if org:
        if name is not None:
            org.name = name  # type: ignore[assignment]
    else:
        org = Organization(org_id=org_id, name=name or org_id)
        db.add(org)
    db.flush()
    return org


def list_organizations(db: Session) -> list[Organization]:
    """List all organizations, ordered by org_id."""
    return db.query(Organization).order_by(Organization.org_id).all()


# --- Plants ---


def get_plant(db: Session, plant_id: str) -> Plant | None:
    """Fetch a plant by its plant_id (e.g. "plant-1")."""
    return db.query(Plant).filter(Plant.plant_id == plant_id).first()


def upsert_plant(db: Session, plant_id: str, org_id: str, name: str | None = None) -> Plant:
    """Create or update a plant, scoped to an org.

    Raises:
        ValueError: If an existing plant is re-assigned to a different org.
    """
    plant = get_plant(db, plant_id)
    if plant:
        if plant.org_id != org_id:
            raise ValueError(
                f"Plant '{plant_id}' belongs to org '{plant.org_id}', not '{org_id}'"
            )
        if name is not None:
            plant.name = name  # type: ignore[assignment]
    else:
        plant = Plant(plant_id=plant_id, org_id=org_id, name=name or plant_id)
        db.add(plant)
    db.flush()
    return plant


def list_plants(db: Session, org_id: str | None = None) -> list[Plant]:
    """List plants, optionally filtered by org."""
    q = db.query(Plant)
    if org_id:
        q = q.filter(Plant.org_id == org_id)
    return q.order_by(Plant.plant_id).all()


# --- Assets ---


def get_asset(db: Session, asset_id: str, org: str | None = None) -> Asset | None:
    """Scoped asset lookup: if org provided, the asset must belong to it."""
    q = db.query(Asset).filter(Asset.asset_id == asset_id)
    if org:
        q = q.filter(Asset.org == org)
    return q.first()


def list_assets(db: Session, org: str | None = None) -> list[Asset]:
    q = db.query(Asset)
    if org:
        q = q.filter(Asset.org == org)
    return q.all()


def upsert_asset(
    db: Session,
    asset_id: str,
    expected_org: str | None = None,
    **kwargs: Any,
) -> Asset:
    """Create or update an asset, optionally guarding cross-org ownership.

    When ``expected_org`` is provided and an existing asset belongs to a
    different org, a ``ValueError`` is raised and the asset row is left
    untouched. This prevents a tenant from hijacking another org's asset by
    overwriting its ``org`` column. When ``expected_org`` is None (or the
    asset does not exist), the behavior is backward compatible.

    Args:
        db: SQLAlchemy session.
        asset_id: Unique asset identifier.
        expected_org: Org the asset must already belong to, if any.

    Raises:
        ValueError: If the asset exists and belongs to a different org than
            ``expected_org``.
    """
    asset = get_asset(db, asset_id)
    if asset:
        if expected_org is not None and asset.org != expected_org:
            raise ValueError(
                f"Asset '{asset_id}' belongs to org '{asset.org}', not '{expected_org}'"
            )
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


def list_scores(
    db: Session,
    asset_id: str | None = None,
    limit: int = 50,
    org: str | None = None,
) -> list[ScoreRecord]:
    """List scores, optionally filtered by asset and org-scoped via Asset join."""
    q = db.query(ScoreRecord)
    if org:
        q = q.join(Asset, ScoreRecord.asset_id == Asset.asset_id).filter(Asset.org == org)
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
    org: str | None = None,
) -> list[Alert]:
    """List alerts, optionally filtered by asset/level and org-scoped via Asset join."""
    q = db.query(Alert)
    if org:
        # Join Asset so we can scope alerts to assets belonging to the tenant org.
        q = q.join(Asset, Alert.asset_id == Asset.asset_id).filter(Asset.org == org)
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


# --- API Keys ---


def create_api_key(
    db: Session,
    name: str,
    key_prefix: str,
    key_hash: str,
    org: str = "default",
) -> ApiKey:
    """Persist a new API key record (prefix + hash only — never the plaintext)."""
    key = ApiKey(name=name, key_prefix=key_prefix, key_hash=key_hash, org=org)
    db.add(key)
    db.flush()
    return key


def get_api_key_by_prefix(db: Session, key_prefix: str) -> ApiKey | None:
    """Fetch key record by its 8-12 char prefix (indexed lookup)."""
    return db.query(ApiKey).filter(ApiKey.key_prefix == key_prefix).first()


def list_api_keys(db: Session, include_revoked: bool = False) -> list[ApiKey]:
    """List keys, optionally including revoked ones."""
    q = db.query(ApiKey).order_by(ApiKey.id)
    if not include_revoked:
        q = q.filter(ApiKey.revoked_at.is_(None))
    return q.all()


def revoke_api_key(db: Session, key_id: int) -> ApiKey | None:
    """Set revoked_at = now. Returns the key or None if not found."""
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if key:
        key.revoked_at = datetime.now(UTC)  # type: ignore[assignment]
        db.flush()
    return key
