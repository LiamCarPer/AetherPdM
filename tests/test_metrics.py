"""Tests for Prometheus metrics."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aether_pdm.db.database import Base
from aether_pdm.serve.app import app, get_db
from aether_pdm.serve.metrics import MODEL_VERSION

# Use in-memory SQLite for tests (same pattern as test_api.py)
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


PAYLOAD = {
    "waveform": [0.1] * 2048,
    "sampling_rate": 12000,
}

MOCK_RESULT = {
    "model_versions": {"anomaly": "1", "fault": "1"},
    "health_score": 0.85,
    "anomaly_score": 0.15,
    "fault": {"class": "normal", "confidence": 0.92},
    "alert": {"level": "healthy", "reason": None},
    "top_features": [],
}


def _series_lines(body: str, series: str) -> list[str]:
    """Return every non-comment line whose series name is ``series``."""
    return [
        line
        for line in body.splitlines()
        if line.startswith(series) and not line.startswith("#")
    ]


def _series_value(body: str, series: str) -> float:
    """Parse the numeric value of the latest sample for ``series`` (0.0 if absent).

    Prometheus only emits a series after its first ``inc()``, so callers must
    treat "absent" as 0.0 rather than fail.
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
    """Install a local DB dependency override, restoring any previous one."""
    previous = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = _override_get_db
    yield
    if previous is None:
        app.dependency_overrides.pop(get_db, None)
    else:
        app.dependency_overrides[get_db] = previous


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_format(client):
    """GET /metrics should return prometheus text format."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    content_type = resp.headers["content-type"]
    assert "text/plain" in content_type or "openmetrics" in content_type
    body = resp.text
    assert "aetherpdm_http_requests_total" in body
    assert "aetherpdm_http_request_duration_seconds" in body


@pytest.mark.asyncio
async def test_metrics_endpoint_counts_requests(client):
    """After a request, the counter should increment."""
    await client.get("/health")
    await client.get("/health")
    body = (await client.get("/metrics")).text
    assert 'endpoint="/health"' in body


@pytest.mark.asyncio
async def test_score_endpoint_increments_business_counters(client):
    """A successful score increments prediction + alert counters."""
    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = MOCK_RESULT

        resp = await client.post("/v1/assets/motor-001/score", json=PAYLOAD)
        assert resp.status_code == 200

        body = (await client.get("/metrics")).text
        assert 'aetherpdm_predictions_total{class="normal"}' in body
        assert 'aetherpdm_alerts_total{level="healthy"}' in body
        assert 'aetherpdm_health_score{asset_id="motor-001"}' in body


@pytest.mark.asyncio
async def test_score_error_does_not_increment_predictions(client):
    """A 503 (no models) should NOT increment prediction counters."""
    # Create the series for a known class via a successful score first, so the
    # before/after comparison works even when tests run in isolation.
    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = MOCK_RESULT
        resp = await client.post("/v1/assets/motor-001/score", json=PAYLOAD)
        assert resp.status_code == 200

    series = 'aetherpdm_predictions_total{class="normal"}'
    before = _series_value((await client.get("/metrics")).text, series)

    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.side_effect = RuntimeError("Models not loaded")
        resp = await client.post("/v1/assets/motor-001/score", json=PAYLOAD)
        assert resp.status_code == 503

    after = _series_value((await client.get("/metrics")).text, series)
    assert after == before


@pytest.mark.asyncio
async def test_model_version_skips_non_numeric_labels(client):
    """Non-numeric model version strings must NOT export a 0.0 gauge sample.

    Regression: calling ``labels()`` before ``float(version)`` registers the
    child series at default 0.0, so a swallowed conversion error still leaked
    a misleading "model version 0" sample for semver strings like "v1.2.3-beta".
    """
    # The registry is shared across tests; clear it so the "fault" label is
    # NOT pre-registered. The bug only leaks a 0.0 sample on first registration
    # of a label, so a dirty registry would mask the regression.
    MODEL_VERSION.clear()

    mixed_result = {
        "model_versions": {"anomaly": "1", "fault": "latest"},
        "health_score": 0.85,
        "anomaly_score": 0.15,
        "fault": {"class": "normal", "confidence": 0.92},
        "alert": {"level": "healthy", "reason": None},
        "top_features": [],
    }

    with patch("aether_pdm.serve.app.get_engine") as mock_get_engine:
        mock_engine = mock_get_engine.return_value
        mock_engine.score.return_value = mixed_result

        resp = await client.post("/v1/assets/motor-001/score", json=PAYLOAD)
        assert resp.status_code == 200

        body = (await client.get("/metrics")).text
        # 1) Numeric version emits correctly.
        anomaly_lines = _series_lines(
            body, 'aetherpdm_model_version{model_name="anomaly"}'
        )
        assert anomaly_lines, "expected a numeric model_version sample for anomaly"
        assert float(anomaly_lines[-1].rsplit(" ", 1)[-1]) == 1.0
        # 2) Non-numeric version does NOT emit any sample (especially not 0.0).
        assert _series_lines(body, 'aetherpdm_model_version{model_name="fault"}') == []
