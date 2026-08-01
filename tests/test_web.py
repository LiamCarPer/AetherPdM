"""Unit tests for AetherPdM Streamlit web components and API client."""

from unittest.mock import MagicMock, patch

from web.api_client import (
    _headers,
    check_health,
    fetch_alerts,
    fetch_asset_detail,
    fetch_assets,
    post_score,
)
from web.config import get_api_base_url


def test_config_base_url():
    url = get_api_base_url()
    assert isinstance(url, str)
    assert url.startswith("http")


@patch("httpx.get")
def test_check_health(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "ok", "version": "0.1.0"}
    mock_get.return_value = mock_resp

    assert check_health("http://test:8000") is True


@patch("httpx.get")
def test_fetch_assets(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"asset_id": "motor-001"}]
    mock_get.return_value = mock_resp

    res = fetch_assets.__wrapped__("http://test:8000")
    assert len(res) == 1
    assert res[0]["asset_id"] == "motor-001"


@patch("httpx.get")
def test_fetch_asset_detail(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"asset_id": "pump-101", "plant": "plant-A"}
    mock_get.return_value = mock_resp

    res = fetch_asset_detail.__wrapped__("http://test:8000", "pump-101")
    assert res is not None
    assert res["asset_id"] == "pump-101"


@patch("httpx.get")
def test_fetch_alerts(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [{"id": 1, "level": "critical", "asset_id": "motor-001"}]
    mock_get.return_value = mock_resp

    res = fetch_alerts.__wrapped__("http://test:8000", limit=10)
    assert len(res) == 1
    assert res[0]["level"] == "critical"


@patch("httpx.post")
def test_post_score(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "asset_id": "motor-001",
        "health_score": 0.85,
        "score_id": 1,
    }
    mock_post.return_value = mock_resp

    res = post_score("http://test:8000", "motor-001", [0.1] * 100, 12000.0, 1772.0)
    assert res is not None
    assert res["health_score"] == 0.85


def test_api_client_sends_key_header(monkeypatch):
    """When AETHER_API_KEY is set, requests carry X-API-Key."""
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    assert _headers() == {"X-API-Key": "test-key"}


def test_api_client_sends_no_key_without_env(monkeypatch):
    """Without AETHER_API_KEY, no auth header is sent (dev mode)."""
    monkeypatch.delenv("AETHER_API_KEY", raising=False)
    assert _headers() == {}


@patch("httpx.get")
def test_fetch_assets_sends_api_key_header(mock_get, monkeypatch):
    """fetch_assets passes the configured API key through to httpx."""
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = []
    mock_get.return_value = mock_resp

    fetch_assets.__wrapped__("http://test:8000")
    _, kwargs = mock_get.call_args
    assert kwargs["headers"] == {"X-API-Key": "test-key"}


@patch("httpx.post")
def test_post_score_sends_api_key_header(mock_post, monkeypatch):
    """post_score passes the configured API key through to httpx."""
    monkeypatch.setenv("AETHER_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"asset_id": "motor-001", "health_score": 0.85}
    mock_post.return_value = mock_resp

    post_score("http://test:8000", "motor-001", [0.1] * 100, 12000.0)
    _, kwargs = mock_post.call_args
    assert kwargs["headers"] == {"X-API-Key": "test-key"}
