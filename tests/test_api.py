"""Contract tests for the FastAPI application."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aether_pdm.db.database import Base
from aether_pdm.serve.app import app, get_db

# Use in-memory SQLite for tests
TEST_DB_URL = "sqlite:///:memory:"
_test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
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


# Apply dependency override
app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=_test_engine)
    yield
    Base.metadata.drop_all(bind=_test_engine)


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_score_asset_mocked(client):
    """Test score endpoint with mocked inference engine."""
    mock_result = {
        "model_versions": {"anomaly": "1", "fault": "1"},
        "health_score": 0.85,
        "anomaly_score": 0.15,
        "fault": {"class": "normal", "confidence": 0.92},
        "alert": {"level": "healthy", "reason": None},
        "top_features": [{"name": "rms", "contribution": 0.21}],
    }

    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = mock_result

        payload = {
            "waveform": [0.1] * 2048,
            "sampling_rate": 12000,
        }
        resp = await client.post("/v1/assets/motor-001/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["asset_id"] == "motor-001"
        assert data["health_score"] == 0.85
        assert data["fault"]["class"] == "normal"
        assert data["alert"]["level"] == "healthy"
        assert "score_id" in data


@pytest.mark.asyncio
async def test_score_asset_empty_waveform(client):
    payload = {
        "waveform": [],
        "sampling_rate": 12000,
    }
    resp = await client.post("/v1/assets/motor-001/score", json=payload)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_list_alerts(client):
    resp = await client.get("/v1/alerts?limit=5")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_list_assets(client):
    resp = await client.get("/v1/assets")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_score_then_list_alerts(client):
    """Score an asset, then verify an alert was persisted."""
    mock_result = {
        "model_versions": {"anomaly": "1", "fault": "1"},
        "health_score": 0.35,
        "anomaly_score": 0.78,
        "fault": {"class": "inner_race", "confidence": 0.88},
        "alert": {"level": "critical", "reason": "detected_inner_race_fault"},
        "top_features": [{"name": "kurtosis", "contribution": 0.45}],
    }

    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = mock_result

        await client.post("/v1/assets/pump-101/score", json={
            "waveform": [0.5] * 2048,
            "sampling_rate": 12000,
        })

        resp = await client.get("/v1/alerts?limit=5")
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["asset_id"] == "pump-101"
        assert data[0]["level"] == "critical"
        assert data[0]["health_score"] == 0.35
