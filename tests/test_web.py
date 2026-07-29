"""Unit tests for AetherPdM Streamlit web components and API client."""

from unittest.mock import MagicMock, patch

from web.api_client import (
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
