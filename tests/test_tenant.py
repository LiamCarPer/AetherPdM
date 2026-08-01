"""Tests for multi-tenant isolation (org -> plant -> asset)."""

import os
from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aether_pdm.db.database import Base
from aether_pdm.db.models import Alert
from aether_pdm.db.repository import (
    acknowledge_alert,
    get_asset,
    get_latest_score,
    list_alerts,
    list_organizations,
    list_plants,
    list_scores,
    save_alert,
    save_score,
    upsert_asset,
    upsert_organization,
    upsert_plant,
)
from aether_pdm.ops.apikeys import create_key
from aether_pdm.serve.app import app, get_db
from aether_pdm.serve.auth import _get_db as auth_get_db

# In-memory SQLite for tests
_test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
_test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)

_MOCK_RESULT = {
    "model_versions": {"anomaly": "1", "fault": "1"},
    "health_score": 0.85,
    "anomaly_score": 0.15,
    "fault": {"class": "normal", "confidence": 0.92},
    "alert": {"level": "healthy", "reason": None},
    "top_features": [],
}


def _override_get_db():
    session = _test_session_local()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _seed_asset(db, asset_id, org):
    upsert_asset(db, asset_id, org=org, plant="plant-1")


def _metrics_series_value(body: str, series: str) -> float:
    """Parse the latest sample value for ``series`` (0.0 if absent).

    Prometheus only emits a series after its first ``inc()``/``set()``, so
    callers must treat "absent" as 0.0 rather than fail.
    """
    for line in body.splitlines():
        if line.startswith(series) and not line.startswith("#"):
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                continue
    return 0.0


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)


@pytest.fixture(autouse=True)
def db_override():
    """Route the app DB dependency AND the auth dependency to the in-memory DB.

    Saves/restores any previous overrides so sibling test modules (test_api.py)
    are not clobbered when the whole suite runs in one process.
    """
    previous_app_db = app.dependency_overrides.get(get_db)
    previous_auth_db = app.dependency_overrides.get(auth_get_db)
    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[auth_get_db] = _override_get_db
    yield
    if previous_app_db is None:
        app.dependency_overrides.pop(get_db, None)
    else:
        app.dependency_overrides[get_db] = previous_app_db
    if previous_auth_db is None:
        app.dependency_overrides.pop(auth_get_db, None)
    else:
        app.dependency_overrides[auth_get_db] = previous_auth_db


@pytest.fixture(autouse=True)
def enable_auth():
    """Force auth ON for these tests (restore after)."""
    old = os.environ.get("AETHER_API_KEY_AUTH_ENABLED")
    os.environ["AETHER_API_KEY_AUTH_ENABLED"] = "true"
    yield
    if old is None:
        os.environ.pop("AETHER_API_KEY_AUTH_ENABLED", None)
    else:
        os.environ["AETHER_API_KEY_AUTH_ENABLED"] = old


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def db_session():
    session = _test_session_local()
    try:
        yield session
    finally:
        session.close()


@pytest.mark.asyncio
async def test_tenant_org_scopes_asset_list(client, db_session):
    """An API key for org 'acme' only sees acme's assets."""
    _seed_asset(db_session, "acme-motor", org="acme")
    _seed_asset(db_session, "other-motor", org="other")
    db_session.commit()

    acme_key = create_key(db_session, name="acme-key", org="acme")

    resp = await client.get("/v1/assets", headers={"X-API-Key": acme_key["api_key"]})
    assert resp.status_code == 200
    data = resp.json()
    assert [a["asset_id"] for a in data] == ["acme-motor"]


@pytest.mark.asyncio
async def test_tenant_cannot_access_other_org_assets(client, db_session):
    """Cross-org asset reads via /v1/orgs/{org_id}/assets -> 403 for real tenants."""
    acme_key = create_key(db_session, name="acme-key", org="acme")

    resp = await client.get(
        "/v1/orgs/other/assets",
        headers={"X-API-Key": acme_key["api_key"]},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_tenant_default_org_dev_mode(client):
    """With auth off (default org tenant), cross-org reads are allowed for dev."""
    os.environ.pop("AETHER_API_KEY_AUTH_ENABLED", None)
    resp = await client.get("/v1/orgs/other/assets")
    assert resp.status_code == 200


def test_org_and_plant_crud(db_session):
    """Upsert/list organizations and org-scoped plants."""
    upsert_organization(db_session, "acme", name="Acme Corp")
    upsert_organization(db_session, "other", name="Other Inc")
    orgs = list_organizations(db_session)
    assert {o.org_id for o in orgs} == {"acme", "other"}
    assert next(o for o in orgs if o.org_id == "acme").name == "Acme Corp"

    upsert_plant(db_session, "plant-1", org_id="acme", name="Plant One")
    upsert_plant(db_session, "plant-9", org_id="other", name="Plant Nine")

    acme_plants = list_plants(db_session, org_id="acme")
    assert [p.plant_id for p in acme_plants] == ["plant-1"]
    assert len(list_plants(db_session)) == 2


@pytest.mark.asyncio
async def test_scored_asset_belongs_to_tenant_org(client, db_session):
    """Scoring with an acme key upserts the asset with org='acme'."""
    acme_key = create_key(db_session, name="acme-key", org="acme")

    payload = {"waveform": [0.1] * 2048, "sampling_rate": 12000}
    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = _MOCK_RESULT

        resp = await client.post(
            "/v1/assets/acme-motor/score",
            json=payload,
            headers={"X-API-Key": acme_key["api_key"]},
        )
        assert resp.status_code == 200

    asset = get_asset(db_session, "acme-motor")
    assert asset is not None
    assert asset.org == "acme"
    assert asset.plant == "default"


@pytest.mark.asyncio
async def test_alerts_scoped_by_org(client, db_session):
    """Alerts for an acme asset are not visible to another org's key."""
    upsert_asset(db_session, "acme-motor", org="acme", plant="plant-1")
    save_alert(
        db_session,
        "acme-motor",
        level="critical",
        reason="vibration",
        health_score=0.2,
    )
    db_session.commit()

    acme_key = create_key(db_session, name="acme-key", org="acme")
    other_key = create_key(db_session, name="other-key", org="other")

    resp = await client.get("/v1/alerts", headers={"X-API-Key": acme_key["api_key"]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["asset_id"] == "acme-motor"

    resp_other = await client.get("/v1/alerts", headers={"X-API-Key": other_key["api_key"]})
    assert resp_other.status_code == 200
    assert resp_other.json() == []


@pytest.mark.asyncio
async def test_auth_off_uses_default_tenant(client):
    """With auth off, /v1/assets works using the default org tenant."""
    os.environ.pop("AETHER_API_KEY_AUTH_ENABLED", None)
    resp = await client.get("/v1/assets")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Security regression: cross-org asset ownership hijack via the score endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tenant_cannot_hijack_other_org_asset(client, db_session):
    """Org B scoring org A's asset must NOT steal ownership (403)."""
    # seed asset owned by "other"
    upsert_asset(db_session, "motor-1", org="other", plant="plant-1")
    db_session.commit()

    # org A key scores it
    key_a = create_key(db_session, name="acme-key", org="acme")

    payload = {"waveform": [0.1] * 2048, "sampling_rate": 12000}
    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = _MOCK_RESULT

        resp = await client.post(
            "/v1/assets/motor-1/score",
            json=payload,
            headers={"X-API-Key": key_a["api_key"]},
        )
        assert resp.status_code == 403

    # asset still belongs to "other"
    asset = get_asset(db_session, "motor-1")
    assert asset is not None
    assert asset.org == "other"


@pytest.mark.asyncio
async def test_hijack_probe_full_closure(client, db_session):
    """Full exploit-closure probe for the cross-org asset hijack.

    Verifies ALL of:
      1) acme key scoring org 'other''s asset -> 403
      2) asset.org stays 'other' (ownership never rewritten)
      3) NO orphan score rows are persisted after the 403 (rollback)
      4) NO orphan alert rows are persisted after the 403 (rollback)
      5) acme's org-scoped alert list contains no rows for the foreign asset
    """
    upsert_asset(db_session, "motor-1", org="other", plant="plant-1")
    db_session.commit()

    acme_key = create_key(db_session, name="acme-key", org="acme")
    db_session.commit()

    payload = {"waveform": [0.1] * 2048, "sampling_rate": 12000}
    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = _MOCK_RESULT
        resp = await client.post(
            "/v1/assets/motor-1/score",
            json=payload,
            headers={"X-API-Key": acme_key["api_key"]},
        )
        assert resp.status_code == 403

    # ownership untouched
    asset = get_asset(db_session, "motor-1")
    assert asset is not None
    assert asset.org == "other"

    # rollback: no orphan score/alert rows from the rejected request
    assert list_scores(db_session, asset_id="motor-1") == []
    assert list_alerts(db_session, asset_id="motor-1") == []

    # acme's org-scoped alert feed must not expose the foreign asset
    assert list_alerts(db_session, org="acme") == []


@pytest.mark.asyncio
async def test_403_does_not_increment_predictions(client, db_session):
    """A 403 cross-org score must NOT inflate business telemetry.

    Regression for ops advisory A-3: prediction/alert counters were updated
    BEFORE the org guard, so rejected requests inflated counters and created
    a phantom health_score series for a foreign asset. Counters must only
    move on a fully authorized (org-owned) scoring.
    """
    upsert_asset(db_session, "motor-1", org="other")
    db_session.commit()
    acme_key = create_key(db_session, name="acme-key", org="acme")

    predictions = 'aetherpdm_predictions_total{class="normal"}'
    alerts = 'aetherpdm_alerts_total{level="healthy"}'
    health = 'aetherpdm_health_score{asset_id="motor-1"}'

    before_p = _metrics_series_value((await client.get("/metrics")).text, predictions)
    before_a = _metrics_series_value((await client.get("/metrics")).text, alerts)
    before_h = _metrics_series_value((await client.get("/metrics")).text, health)

    payload = {"waveform": [0.1] * 2048, "sampling_rate": 12000}
    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = _MOCK_RESULT
        resp = await client.post(
            "/v1/assets/motor-1/score",
            json=payload,
            headers={"X-API-Key": acme_key["api_key"]},
        )
        assert resp.status_code == 403

    after_p = _metrics_series_value((await client.get("/metrics")).text, predictions)
    after_a = _metrics_series_value((await client.get("/metrics")).text, alerts)
    after_h = _metrics_series_value((await client.get("/metrics")).text, health)

    assert after_p == before_p
    assert after_a == before_a
    assert after_h == before_h


@pytest.mark.asyncio
async def test_score_own_asset_ok_for_tenant(client, db_session):
    """A tenant scoring its own asset still succeeds (org guard is a no-op)."""
    upsert_asset(db_session, "motor-1", org="acme", plant="plant-1")
    db_session.commit()

    acme_key = create_key(db_session, name="acme-key", org="acme")
    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = _MOCK_RESULT
        resp = await client.post(
            "/v1/assets/motor-1/score",
            json={"waveform": [0.1] * 2048, "sampling_rate": 12000},
            headers={"X-API-Key": acme_key["api_key"]},
        )
        assert resp.status_code == 200

    asset = get_asset(db_session, "motor-1")
    assert asset.org == "acme"


@pytest.mark.asyncio
async def test_score_dev_mode_skips_org_guard(client, db_session):
    """In dev mode (default tenant) the org guard is disabled: no 403."""
    upsert_asset(db_session, "motor-1", org="other", plant="plant-1")
    db_session.commit()

    os.environ.pop("AETHER_API_KEY_AUTH_ENABLED", None)
    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = _MOCK_RESULT
        resp = await client.post(
            "/v1/assets/motor-1/score",
            json={"waveform": [0.1] * 2048, "sampling_rate": 12000},
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_dev_mode_score_foreign_asset_persists_side_effects(client, db_session):
    """Dev convenience preserved: default tenant may score any asset.

    The org guard is skipped for the DEFAULT_ORG tenant, so a dev-mode score of
    another org's asset returns 200, rewrites the asset row to 'default', and
    persists its score/alert rows (intended dev behavior, auth disabled).
    """
    upsert_asset(db_session, "motor-1", org="other", plant="plant-1")
    db_session.commit()

    os.environ.pop("AETHER_API_KEY_AUTH_ENABLED", None)
    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = _MOCK_RESULT
        resp = await client.post(
            "/v1/assets/motor-1/score",
            json={"waveform": [0.1] * 2048, "sampling_rate": 12000},
        )
        assert resp.status_code == 200

    # dev rewrite to default + side-effects persisted
    asset = get_asset(db_session, "motor-1")
    assert asset is not None
    assert asset.org == "default"
    assert len(list_scores(db_session, asset_id="motor-1")) == 1
    assert len(list_alerts(db_session, asset_id="motor-1")) == 1


# ---------------------------------------------------------------------------
# Repository coverage: upsert update branches + org guards
# ---------------------------------------------------------------------------


def test_upsert_organization_update_branch(db_session):
    """Updating an existing org keeps the row id and replaces the name."""
    org = upsert_organization(db_session, "acme", name="Acme Corp")
    assert org.name == "Acme Corp"

    updated = upsert_organization(db_session, "acme", name="Acme Renamed")
    assert updated.id == org.id
    assert updated.name == "Acme Renamed"

    # no name passed -> name is left untouched
    untouched = upsert_organization(db_session, "acme")
    assert untouched.name == "Acme Renamed"


def test_upsert_plant_update_branch_and_org_guard(db_session):
    """Existing plants update in place; cross-org re-assignment is rejected."""
    created = upsert_plant(db_session, "plant-1", org_id="acme", name="Plant One")
    assert created.org_id == "acme"

    updated = upsert_plant(db_session, "plant-1", org_id="acme", name="Plant Renamed")
    assert updated.id == created.id
    assert updated.name == "Plant Renamed"
    assert updated.org_id == "acme"

    with pytest.raises(ValueError, match="belongs to org 'acme', not 'other'"):
        upsert_plant(db_session, "plant-1", org_id="other")

    # failed re-assignment must not have mutated the row
    assert upsert_plant(db_session, "plant-1", org_id="acme").org_id == "acme"


def test_upsert_asset_expected_org_guard(db_session):
    """upsert_asset raises when expected_org mismatches the existing row."""
    upsert_asset(db_session, "motor-1", org="other", plant="plant-1")
    db_session.commit()

    with pytest.raises(ValueError, match="belongs to org 'other', not 'acme'"):
        upsert_asset(db_session, "motor-1", expected_org="acme", org="acme")

    # guard is a no-op when org matches, and when expected_org is None
    same = upsert_asset(db_session, "motor-1", expected_org="other", org="other")
    assert same.org == "other"
    legacy = upsert_asset(db_session, "motor-1", org="acme")
    assert legacy.org == "acme"

    # creating a brand-new asset never trips the guard
    fresh = upsert_asset(db_session, "new-motor", expected_org="acme", org="acme")
    assert fresh.org == "acme"


def test_get_asset_org_scope(db_session):
    """org-scoped get_asset: in-org found, cross-org None."""
    upsert_asset(db_session, "motor-1", org="acme")
    db_session.commit()

    assert get_asset(db_session, "motor-1", org="acme") is not None
    assert get_asset(db_session, "motor-1", org="other") is None
    # unscoped lookup still finds the asset
    assert get_asset(db_session, "motor-1") is not None


def test_list_scores_org_scope(db_session):
    """org-scoped list_scores returns only scores for assets in the org."""
    upsert_asset(db_session, "acme-motor", org="acme")
    upsert_asset(db_session, "other-motor", org="other")
    save_score(db_session, "acme-motor", _MOCK_RESULT)
    save_score(db_session, "other-motor", _MOCK_RESULT)
    db_session.commit()

    acme_scores = list_scores(db_session, org="acme")
    assert [s.asset_id for s in acme_scores] == ["acme-motor"]

    other_scores = list_scores(db_session, org="other")
    assert [s.asset_id for s in other_scores] == ["other-motor"]

    # asset_id filter is still respected alongside the org scope
    filtered = list_scores(db_session, asset_id="acme-motor", org="acme")
    assert [s.asset_id for s in filtered] == ["acme-motor"]


def test_get_latest_score(db_session):
    """get_latest_score returns the newest score record for an asset."""
    upsert_asset(db_session, "motor-1", org="acme")
    older = dict(_MOCK_RESULT)
    older["health_score"] = 0.5
    newer = dict(_MOCK_RESULT)
    newer["health_score"] = 0.9

    r_older = save_score(db_session, "motor-1", older)
    r_newer = save_score(db_session, "motor-1", newer)
    # Deterministic ordering regardless of wall-clock ties between flushes.
    r_older.created_at = datetime(2020, 1, 1, tzinfo=UTC)
    r_newer.created_at = datetime(2021, 1, 1, tzinfo=UTC)
    db_session.commit()

    latest = get_latest_score(db_session, "motor-1")
    assert latest is not None
    assert latest.health_score == 0.9

    # unknown asset -> None
    assert get_latest_score(db_session, "missing") is None


def test_acknowledge_alert(db_session):
    """acknowledge_alert flips acknowledged=1; unknown id returns None."""
    upsert_asset(db_session, "motor-1", org="acme")
    alert = save_alert(
        db_session,
        "motor-1",
        level="critical",
        reason="vibration",
        health_score=0.2,
    )
    db_session.commit()
    assert alert.acknowledged == 0

    updated = acknowledge_alert(db_session, alert.id)
    assert updated is not None
    assert updated.acknowledged == 1

    assert acknowledge_alert(db_session, 999999) is None


def test_acknowledge_alert_org_scoped(db_session):
    """acknowledge_alert refuses cross-org acks when org is provided."""
    upsert_asset(db_session, "motor-1", org="other")
    alert = save_alert(
        db_session,
        "motor-1",
        level="critical",
        reason="vibration",
        health_score=0.2,
    )
    db_session.commit()
    assert alert.acknowledged == 0

    # cross-org acknowledge must be refused and leave the row untouched
    assert acknowledge_alert(db_session, alert.id, org="acme") is None
    db_session.commit()
    assert db_session.get(Alert, alert.id).acknowledged == 0

    # owning org can acknowledge
    updated = acknowledge_alert(db_session, alert.id, org="other")
    assert updated is not None
    assert updated.acknowledged == 1

    # unknown org for an existing asset is also refused
    other_alert = save_alert(
        db_session,
        "motor-1",
        level="warning",
        reason="noise",
        health_score=0.5,
    )
    db_session.commit()
    assert acknowledge_alert(db_session, other_alert.id, org="ghost") is None
    db_session.commit()
    assert db_session.get(Alert, other_alert.id).acknowledged == 0


def test_list_alerts_level_and_asset_filters(db_session):
    """list_alerts supports level/asset_id filters and org scoping."""
    upsert_asset(db_session, "acme-motor", org="acme")
    upsert_asset(db_session, "other-motor", org="other")
    save_alert(db_session, "acme-motor", level="critical", reason="vibration", health_score=0.2)
    save_alert(db_session, "acme-motor", level="warning", reason="noise", health_score=0.5)
    save_alert(db_session, "other-motor", level="critical", reason="shock", health_score=0.1)
    db_session.commit()

    critical = list_alerts(db_session, level="critical")
    assert {a.asset_id for a in critical} == {"acme-motor", "other-motor"}

    acme_alerts = list_alerts(db_session, asset_id="acme-motor")
    assert len(acme_alerts) == 2

    combined = list_alerts(db_session, asset_id="acme-motor", level="warning")
    assert len(combined) == 1
    assert combined[0].reason == "noise"

    org_scoped = list_alerts(db_session, org="acme")
    assert len(org_scoped) == 2
    assert all(a.asset_id == "acme-motor" for a in org_scoped)

    empty = list_alerts(db_session, asset_id="acme-motor", level="critical", org="other")
    assert empty == []


# ---------------------------------------------------------------------------
# API coverage: cross-org 404 + org list endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_asset_cross_org_404(client, db_session):
    """GET /v1/assets/{id} for an asset in another org -> 404 (not leaked)."""
    upsert_asset(db_session, "secret-motor", org="other", plant="plant-1")
    db_session.commit()

    acme_key = create_key(db_session, name="acme-key", org="acme")
    resp = await client.get(
        "/v1/assets/secret-motor",
        headers={"X-API-Key": acme_key["api_key"]},
    )
    assert resp.status_code == 404

    # the owning org still sees its own asset
    other_key = create_key(db_session, name="other-key", org="other")
    resp_ok = await client.get(
        "/v1/assets/secret-motor",
        headers={"X-API-Key": other_key["api_key"]},
    )
    assert resp_ok.status_code == 200
    assert resp_ok.json()["asset_id"] == "secret-motor"
    assert resp_ok.json()["org"] == "other"


@pytest.mark.asyncio
async def test_get_asset_missing_404(client, db_session):
    """Unknown asset id -> 404."""
    acme_key = create_key(db_session, name="acme-key", org="acme")
    resp = await client.get(
        "/v1/assets/does-not-exist",
        headers={"X-API-Key": acme_key["api_key"]},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_orgs_endpoint(client, db_session):
    """GET /v1/orgs lists all organizations for any tenant."""
    upsert_organization(db_session, "acme", name="Acme Corp")
    upsert_organization(db_session, "other", name="Other Inc")
    db_session.commit()

    acme_key = create_key(db_session, name="acme-key", org="acme")
    resp = await client.get("/v1/orgs", headers={"X-API-Key": acme_key["api_key"]})
    assert resp.status_code == 200
    data = resp.json()
    assert [o["org_id"] for o in data] == ["acme", "other"]


@pytest.mark.asyncio
async def test_list_org_plants_scoping(client, db_session):
    """GET /v1/orgs/{org_id}/plants: own org 200, other org 403."""
    upsert_organization(db_session, "acme", name="Acme Corp")
    upsert_organization(db_session, "other", name="Other Inc")
    upsert_plant(db_session, "plant-1", org_id="acme", name="Plant One")
    db_session.commit()

    acme_key = create_key(db_session, name="acme-key", org="acme")

    resp_own = await client.get(
        "/v1/orgs/acme/plants",
        headers={"X-API-Key": acme_key["api_key"]},
    )
    assert resp_own.status_code == 200
    assert [p["plant_id"] for p in resp_own.json()] == ["plant-1"]

    resp_other = await client.get(
        "/v1/orgs/other/plants",
        headers={"X-API-Key": acme_key["api_key"]},
    )
    assert resp_other.status_code == 403


@pytest.mark.asyncio
async def test_list_org_assets_scoping(client, db_session):
    """GET /v1/orgs/{org_id}/assets: own org 200, other org 403."""
    upsert_asset(db_session, "acme-motor", org="acme")
    upsert_asset(db_session, "other-motor", org="other")
    db_session.commit()

    acme_key = create_key(db_session, name="acme-key", org="acme")

    resp_own = await client.get(
        "/v1/orgs/acme/assets",
        headers={"X-API-Key": acme_key["api_key"]},
    )
    assert resp_own.status_code == 200
    assert [a["asset_id"] for a in resp_own.json()] == ["acme-motor"]

    resp_other = await client.get(
        "/v1/orgs/other/assets",
        headers={"X-API-Key": acme_key["api_key"]},
    )
    assert resp_other.status_code == 403
