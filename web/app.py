"""AetherPdM Operations Dashboard - Home / Plant Overview."""

import streamlit as st

from web.api_client import fetch_alerts, fetch_assets
from web.bootstrap import init_page
from web.components.metrics import render_kpi_cards
from web.components.tables import render_alerts_table, render_assets_table

# Must be called first on script execution
init_page()

# Page Header
col_title, col_sync = st.columns([4, 1])
with col_title:
    st.title("AetherPdM — Plant Operations Overview")
    st.caption("Real-time predictive maintenance & condition monitoring for rotating equipment")

with col_sync:
    st.write("")
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

base_url = st.session_state.get("api_base_url", "http://localhost:8000")
api_online = st.session_state.get("api_online", False)

# Fetch Fleet Data
assets = fetch_assets(base_url) if api_online else []
alerts = fetch_alerts(base_url, limit=50) if api_online else []

# Top KPI Summary Cards
render_kpi_cards(assets, alerts, api_online)

st.markdown("---")

# Main Content Grid
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Registered Asset Inventory")
    if not assets:
        st.info("No assets registered. Score a waveform to register one.")
    else:
        render_assets_table(assets)

with col_right:
    st.subheader("Recent Operational Alerts")
    if not alerts:
        st.info("No alerts. All assets are healthy.")
    else:
        render_alerts_table(alerts[:10])
