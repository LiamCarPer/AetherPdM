"""API client module for fetching data from AetherPdM FastAPI service."""

from typing import Any

import httpx
import streamlit as st


def check_health(base_url: str) -> bool:
    """Check if the AetherPdM API is online and responding."""
    try:
        response = httpx.get(f"{base_url}/health", timeout=3.0)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except Exception:
        return False


@st.cache_data(ttl=10)
def fetch_assets(base_url: str) -> list[dict[str, Any]]:
    """Fetch all registered assets from the backend."""
    try:
        response = httpx.get(f"{base_url}/v1/assets", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch assets: {e}")
        return []


@st.cache_data(ttl=10)
def fetch_asset_detail(base_url: str, asset_id: str) -> dict[str, Any] | None:
    """Fetch details for a specific asset by ID."""
    try:
        response = httpx.get(f"{base_url}/v1/assets/{asset_id}", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch asset '{asset_id}': {e}")
        return None


@st.cache_data(ttl=10)
def fetch_alerts(
    base_url: str,
    asset_id: str | None = None,
    level: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Fetch alerts log with optional filtering."""
    params: dict[str, Any] = {"limit": limit}
    if asset_id:
        params["asset_id"] = asset_id
    if level:
        params["level"] = level

    try:
        response = httpx.get(f"{base_url}/v1/alerts", params=params, timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch alerts: {e}")
        return []


def post_score(
    base_url: str,
    asset_id: str,
    waveform: list[float],
    sampling_rate: float,
    rpm: float | None = None,
) -> dict[str, Any] | None:
    """Score a vibration waveform for a given asset ID."""
    payload: dict[str, Any] = {
        "waveform": waveform,
        "sampling_rate": sampling_rate,
    }
    if rpm is not None:
        payload["rpm"] = rpm

    try:
        response = httpx.post(
            f"{base_url}/v1/assets/{asset_id}/score",
            json=payload,
            timeout=15.0,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Scoring request failed: {e}")
        return None
