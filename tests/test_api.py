"""Contract tests for the FastAPI application."""

import pytest
from httpx import ASGITransport, AsyncClient

from aether_pdm.serve.app import app


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
async def test_score_asset(client):
    payload = {
        "waveform": [0.1] * 1024,
        "sampling_rate": 12000,
    }
    resp = await client.post("/v1/assets/motor-001/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset_id"] == "motor-001"
    assert "health_score" in data
    assert "anomaly_score" in data
    assert "fault" in data
    assert "alert" in data


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
