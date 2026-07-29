"""DataFrame tables component for AetherPdM Dashboard."""

from typing import Any

import pandas as pd
import streamlit as st


def render_alerts_table(alerts: list[dict[str, Any]]) -> None:
    """Render formatted alerts log table."""
    if not alerts:
        st.info("No alerts. All assets are healthy.")
        return

    records = []
    for a in alerts:
        level = (a.get("level") or "unknown").lower()
        badge = (
            "CRITICAL"
            if level == "critical"
            else "WARNING"
            if level == "warning"
            else "HEALTHY"
        )
        hs = a.get("health_score")
        hs_str = f"{hs * 100:.1f}%" if hs is not None else "N/A"
        records.append({
            "ID": a.get("id"),
            "Asset ID": a.get("asset_id"),
            "Severity": badge,
            "Health Score": hs_str,
            "Fault Class": a.get("fault_class") or "N/A",
            "Reason": a.get("reason") or "N/A",
            "Timestamp": str(a.get("created_at", ""))[:19].replace("T", " "),
        })

    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_assets_table(assets: list[dict[str, Any]]) -> None:
    """Render formatted asset inventory table."""
    if not assets:
        st.info("No assets registered. Score a waveform to register one.")
        return

    records = []
    for a in assets:
        records.append({
            "Asset ID": a.get("asset_id"),
            "Organization": a.get("org", "default"),
            "Plant": a.get("plant", "default"),
            "Asset Type": a.get("asset_type") or "Bearing",
            "Nominal RPM": a.get("rpm_nominal") or "N/A",
            "Anomaly Threshold": a.get("anomaly_threshold", 0.8),
            "Registered At": str(a.get("created_at", ""))[:10],
        })

    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True, hide_index=True)
