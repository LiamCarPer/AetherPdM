"""Contract tests for the FastAPI application."""

import os
from unittest.mock import patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.ensemble import IsolationForest, RandomForestClassifier

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
async def test_score_asset_mocked(client):
    """Test score endpoint with mocked inference engine."""
    # Patch the engine to return known values
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
