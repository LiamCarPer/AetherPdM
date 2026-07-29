"""KPI Metric Cards Component for AetherPdM Dashboard."""

from typing import Any

import streamlit as st


def render_kpi_cards(
    assets: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    api_online: bool,
) -> None:
    """Render top KPI summary cards."""
    col1, col2, col3, col4 = st.columns(4)

    # Metric 1: Monitored Assets
    total_assets = len(assets)
    col1.metric(
        label="Monitored Assets",
        value=total_assets,
        help="Total registered machines/bearings in fleet",
    )

    # Metric 2: Active Alerts (Critical + Warning)
    critical_count = sum(1 for a in alerts if a.get("level") == "critical")
    warning_count = sum(1 for a in alerts if a.get("level") == "warning")
    active_alerts = critical_count + warning_count
    delta_msg = (
        f"{critical_count} Critical, {warning_count} Warning"
        if active_alerts > 0
        else "All Clear"
    )
    col2.metric(
        label="Active Alerts",
        value=active_alerts,
        delta=delta_msg,
        delta_color="inverse" if active_alerts > 0 else "normal",
        help="Alerts requiring operator inspection",
    )

    # Metric 3: Fleet Avg Health Score
    health_scores: list[float] = [
        float(a["health_score"])
        for a in alerts
        if a.get("health_score") is not None
    ]
    if health_scores:
        avg_health = sum(health_scores) / len(health_scores)
        health_pct = f"{avg_health * 100:.1f}%"
    else:
        health_pct = "N/A"

    col3.metric(
        label="Fleet Avg Health",
        value=health_pct,
        help="Average health score across scored assets",
    )

    # Metric 4: API Readiness Status
    status_label = "ONLINE" if api_online else "OFFLINE"
    col4.metric(
        label="API Status",
        value=status_label,
        delta="Connected" if api_online else "Disconnected",
        delta_color="normal" if api_online else "off",
        help="FastAPI Backend Readiness Status",
    )
