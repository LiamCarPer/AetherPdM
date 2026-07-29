"""Configuration constants for AetherPdM Streamlit dashboard."""

import os


def get_api_base_url() -> str:
    """Return the base URL for the AetherPdM REST API."""
    return os.environ.get("AETHER_API_URL", "http://localhost:8000").rstrip("/")


# Severity color tokens
COLOR_HEALTHY = "#10B981"   # Emerald Green
COLOR_WARNING = "#F59E0B"   # Amber Yellow
COLOR_CRITICAL = "#EF4444"  # Ruby Red
COLOR_NEUTRAL = "#9CA3AF"   # Muted Gray / Unscored
