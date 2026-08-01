"""Tests for API key authentication."""

import hashlib
import os
import secrets

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aether_pdm.db.database import Base
from aether_pdm.ops.apikeys import create_key, list_keys, revoke_key
from aether_pdm.serve.app import app, get_db
from aether_pdm.serve.auth import (
    _get_db as auth_get_db,
)
from aether_pdm.serve.auth import (
    generate_api_key,
    hash_api_key,
    verify_api_key_hash,
)

# In-memory SQLite for tests
_test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
_test_session_local = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


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
async def test_health_open_without_key(client):
    """Health endpoint should be open even with auth enabled."""
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_metrics_open_without_key(client):
    """Metrics endpoint should be open even with auth enabled."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_assets_requires_key(client):
    """GET /v1/assets without a key -> 401."""
    resp = await client.get("/v1/assets")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_assets_valid_key(client, db_session):
    """A valid API key grants access to /v1/assets."""
    key = create_key(db_session, name="demo")
    resp = await client.get("/v1/assets", headers={"X-API-Key": key["api_key"]})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_score_requires_key(client):
    """POST /v1/assets/{id}/score without a key -> 401."""
    payload = {"waveform": [0.1] * 2048, "sampling_rate": 12000}
    resp = await client.post("/v1/assets/motor-001/score", json=payload)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_invalid_key_rejected(client):
    """A well-formed but unknown key is rejected."""
    resp = await client.get(
        "/v1/assets",
        headers={"X-API-Key": "aether_Ab12cD34_somewrongsecretvalue"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_malformed_key_rejected(client):
    """A key without the <prefix>_<secret> shape is rejected as format-invalid."""
    resp = await client.get(
        "/v1/assets",
        headers={"X-API-Key": "aether_onlyone"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_bare_format_key_accepted(client, db_session):
    """Bare '<prefix><secret>' (no aether_ prefix) is accepted for robustness."""
    key = create_key(db_session, name="bare")
    secret = key["api_key"].split("_", 2)[2]
    bare = key["key_prefix"] + secret
    resp = await client.get("/v1/assets", headers={"X-API-Key": bare})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_wrong_secret_for_prefix_rejected(client, db_session):
    """A known prefix with a wrong secret fails hash verification -> 401."""
    key = create_key(db_session, name="demo")
    wrong_secret = "0" * 48
    resp = await client.get(
        "/v1/assets",
        headers={"X-API-Key": f"aether_{key['key_prefix']}_{wrong_secret}"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_revoked_key_rejected(client, db_session):
    """A revoked key is rejected."""
    key = create_key(db_session, name="demo")
    revoke_key(db_session, key["id"])
    resp = await client.get("/v1/assets", headers={"X-API-Key": key["api_key"]})
    assert resp.status_code == 401


def test_hash_roundtrip():
    """Hashing + constant-time verification roundtrip."""
    _, _, secret = generate_api_key()
    stored = hash_api_key(secret)
    assert verify_api_key_hash(secret, stored) is True
    assert verify_api_key_hash("wrong-secret", stored) is False
    assert verify_api_key_hash(secret, "not-a-valid-stored-hash") is False


def test_hash_v2_roundtrip():
    """New hashes are v2-versioned and verify with the right secret only."""
    _, _, secret = generate_api_key()
    stored = hash_api_key(secret)
    # v2 scheme: v2:<salt_hex>:<dk_hex>
    assert stored.startswith("v2:")
    parts = stored.split(":")
    assert len(parts) == 3
    # salt + derived key are hex
    assert bytes.fromhex(parts[1])
    assert bytes.fromhex(parts[2])
    # same secret verifies; wrong secret is rejected
    assert verify_api_key_hash(secret, stored) is True
    assert verify_api_key_hash("wrong-secret", stored) is False


def test_verify_legacy_v1_hash():
    """Legacy 100k-iteration hashes (salt:dk and v1: salt:dk) still verify."""
    secret = "legacy-secret"
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt, 100_000)

    # pre-versioning format: salt:dk (implicit v1)
    legacy = f"{salt.hex()}:{dk.hex()}"
    assert verify_api_key_hash(secret, legacy) is True
    assert verify_api_key_hash("wrong-secret", legacy) is False

    # explicit v1-prefixed format
    versioned_v1 = f"v1:{salt.hex()}:{dk.hex()}"
    assert verify_api_key_hash(secret, versioned_v1) is True
    assert verify_api_key_hash("wrong-secret", versioned_v1) is False


def test_verify_malformed_hash_returns_false():
    """Garbage, truncated, and unknown-version hashes are rejected."""
    _, _, secret = generate_api_key()
    malformed = [
        "not-a-valid-stored-hash",
        "v2:onlyone",
        "v2:zz:yy",             # invalid hex salt
        "v2:salt:dk:extra",     # too many parts
        "v3:abc:def",           # unknown version
        "v1:onlytwo",
        "v2:",                  # empty parts
        "v0:00:00",             # bogus version with valid hex
    ]
    for stored in malformed:
        assert verify_api_key_hash(secret, stored) is False, stored


def test_generate_key_format():
    """Generated keys use aether_<prefix>_<secret> with 3 parts."""
    full_key, prefix, secret = generate_api_key()
    assert full_key.startswith("aether_")
    parts = full_key.split("_")
    assert len(parts) == 3
    assert parts[1] == prefix
    assert parts[2] == secret
    assert len(prefix) == 8


@pytest.mark.asyncio
async def test_auth_disabled_by_default(client):
    """Without the env var, auth is a no-op and /v1/* is open."""
    os.environ.pop("AETHER_API_KEY_AUTH_ENABLED", None)
    resp = await client.get("/v1/assets")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_multiple_keys_prefix_isolation(client, db_session):
    """Each key's prefix isolates it; cross-prefix forgery is rejected."""
    key1 = create_key(db_session, name="key1")
    key2 = create_key(db_session, name="key2")

    # Each key authenticates with its own full key
    resp1 = await client.get("/v1/assets", headers={"X-API-Key": key1["api_key"]})
    resp2 = await client.get("/v1/assets", headers={"X-API-Key": key2["api_key"]})
    assert resp1.status_code == 200
    assert resp2.status_code == 200

    # Forgery: key1's prefix + key2's secret -> must fail hash verification
    forged = f"aether_{key1['key_prefix']}_{key2['api_key'].split('_', 2)[2]}"
    resp_forged = await client.get("/v1/assets", headers={"X-API-Key": forged})
    assert resp_forged.status_code == 401


def test_list_keys_never_exposes_secret_or_hash(db_session):
    """list_keys must not leak the plaintext key or the stored hash."""
    create_key(db_session, name="demo")
    keys = list_keys(db_session)
    assert len(keys) >= 1
    for k in keys:
        assert "api_key" not in k
        assert "key_hash" not in k
        assert "secret" not in k
