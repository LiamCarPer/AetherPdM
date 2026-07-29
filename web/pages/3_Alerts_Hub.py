"""Operational Alert Management Hub Page."""

import pandas as pd
import streamlit as st

from web.api_client import fetch_alerts, fetch_assets
from web.bootstrap import init_page
from web.components.tables import render_alerts_table

# Must be called first on script execution
init_page()

# Page Header
col_title, col_sync = st.columns([4, 1])
with col_title:
    st.title("Operational Alerts Hub")
    st.caption("Central alert monitoring feed, severity filtering, and historical logs")

with col_sync:
    st.write("")
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

base_url = st.session_state.get("api_base_url", "http://localhost:8000")
api_online = st.session_state.get("api_online", False)

assets = fetch_assets(base_url) if api_online else []
asset_options = ["All Assets"] + [a.get("asset_id") for a in assets if a.get("asset_id")]

# Filter Bar
f_col1, f_col2, f_col3 = st.columns([2, 2, 2])

with f_col1:
    selected_asset = st.selectbox("Filter by Asset:", options=asset_options)
    asset_filter = None if selected_asset == "All Assets" else selected_asset

with f_col2:
    selected_level = st.selectbox(
        "Filter by Severity Level:",
        options=["All Levels", "critical", "warning", "healthy"],
    )
    level_filter = None if selected_level == "All Levels" else selected_level

with f_col3:
    limit_val = st.slider("Max Records Limit:", min_value=10, max_value=200, value=50, step=10)

st.markdown("---")

# Fetch Filtered Alerts
alerts = fetch_alerts(
    base_url,
    asset_id=asset_filter,
    level=level_filter,
    limit=limit_val,
) if api_online else []

# Overview Counters
c_col1, c_col2, c_col3 = st.columns(3)
crit_cnt = sum(1 for a in alerts if a.get("level") == "critical")
warn_cnt = sum(1 for a in alerts if a.get("level") == "warning")
hlth_cnt = sum(1 for a in alerts if a.get("level") == "healthy")

c_col1.metric(
    "Critical Alerts",
    crit_cnt,
    delta="Requires Immediate Action" if crit_cnt > 0 else "None",
    delta_color="inverse",
)
c_col2.metric(
    "Warning Alerts",
    warn_cnt,
    delta="Schedule Maintenance" if warn_cnt > 0 else "None",
    delta_color="inverse",
)
c_col3.metric("Healthy Events", hlth_cnt, delta="Normal Operation")

st.markdown("---")

st.subheader("Filtered Alert Logs")

if not alerts:
    st.info("No alerts. All assets are healthy.")
else:
    render_alerts_table(alerts)

    # Export CSV Option
    export_df = pd.DataFrame(alerts)
    csv_data = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Export Alerts to CSV",
        data=csv_data,
        file_name="aether_pdm_alerts.csv",
        mime="text/csv",
        use_container_width=True,
    )
