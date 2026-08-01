"""Configuration constants for AetherPdM Streamlit dashboard."""

import os


def get_api_base_url() -> str:
    """Return the base URL for the AetherPdM REST API."""
    return os.environ.get("AETHER_API_URL", "http://localhost:8000").rstrip("/")


def get_api_key() -> str | None:
    """Return the AetherPdM API key for dashboard auth, if configured.

    Empty/unset means the dashboard talks to an unauthenticated API.
    """
    key = os.environ.get("AETHER_API_KEY")
    return key if key else None


# Severity color tokens
COLOR_HEALTHY = "#10B981"   # Emerald Green
COLOR_WARNING = "#F59E0B"   # Amber Yellow
COLOR_CRITICAL = "#EF4444"  # Ruby Red
COLOR_NEUTRAL = "#9CA3AF"   # Muted Gray / Unscored
