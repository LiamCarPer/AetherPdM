"""Page bootstrap module for Streamlit multi-page app."""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st  # noqa: E402

from web.api_client import check_health  # noqa: E402
from web.config import get_api_base_url  # noqa: E402


def init_page() -> None:
    """Initialize page configuration, CSS styles, session state, and health check."""
    # Must be the first Streamlit command executed on every script run
    st.set_page_config(
        page_title="AetherPdM Operations Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject CSS
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.html(f"<style>{css_path.read_text(encoding='utf-8')}</style>")

    # Initialize Session State defaults using setdefault
    ss = st.session_state
    base_url = ss.setdefault("api_base_url", get_api_base_url())
    ss.setdefault("selected_asset_id", None)
    ss.setdefault("last_score_result", None)
    ss.setdefault("auto_refresh", False)

    # Health check
    api_online = check_health(base_url)
    ss["api_online"] = api_online

    if not api_online:
        st.warning(f"API unreachable at {base_url}")
