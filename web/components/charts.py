"""Plotly visualization charts for AetherPdM Dashboard."""

from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from web.config import COLOR_CRITICAL, COLOR_HEALTHY, COLOR_NEUTRAL, COLOR_WARNING


def render_feature_importance(top_features: list[dict[str, Any]]) -> None:
    """Render horizontal Plotly bar chart of normalized top feature contributions."""
    if not top_features:
        st.info("No feature contribution data available.")
        return

    # Extract names and contributions
    names = [f.get("name", "unknown") for f in top_features]
    raw_vals = [float(f.get("contribution", 0.0)) for f in top_features]
    total_val = sum(raw_vals)

    # Normalize contributions to percentages
    if total_val > 0:
        pct_vals = [(val / total_val) * 100 for val in raw_vals]
    else:
        pct_vals = [0.0] * len(raw_vals)

    # Sort ascending for horizontal bar chart (top feature at top)
    names_sorted = names[::-1]
    pct_sorted = pct_vals[::-1]

    fig = go.Figure(
        go.Bar(
            x=pct_sorted,
            y=names_sorted,
            orientation="h",
            marker=dict(
                color=pct_sorted,
                colorscale="Viridis",
                showscale=False,
            ),
            text=[f"{p:.1f}%" for p in pct_sorted],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Top Diagnostic Features (% Contribution)",
        xaxis_title="Relative Contribution (%)",
        yaxis_title="Feature Name",
        height=320,
        margin=dict(l=20, r=40, t=40, b=40),
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_waveform_plot(waveform: list[float], sampling_rate: float) -> None:
    """Render interactive Plotly time-series vibration waveform plot."""
    if not waveform or sampling_rate <= 0:
        st.info("No waveform signal to display.")
        return

    arr = np.array(waveform, dtype=float)
    time_axis = np.arange(len(arr)) / sampling_rate

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=arr,
            mode="lines",
            name="Acceleration",
            line=dict(color="#38BDF8", width=1.2),
        )
    )

    fig.update_layout(
        title=f"Raw Vibration Waveform ({len(arr)} samples @ {sampling_rate:.0f} Hz)",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (g / m/s²)",
        height=350,
        margin=dict(l=20, r=20, t=40, b=40),
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_health_gauge(health_score: float | None) -> None:
    """Render semi-circle Plotly gauge indicator for asset health score."""
    if health_score is None:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=0,
                number={
                    "prefix": "",
                    "suffix": "% (Unscored)",
                    "font": {"color": COLOR_NEUTRAL, "size": 24},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": COLOR_NEUTRAL},
                    "bar": {"color": COLOR_NEUTRAL},
                    "bgcolor": "#1E293B",
                },
                title={"text": "Asset Health Score", "font": {"size": 18}},
            )
        )
    else:
        score_pct = float(np.clip(health_score * 100, 0, 100))
        gauge_color = (
            COLOR_HEALTHY
            if score_pct >= 75
            else COLOR_WARNING
            if score_pct >= 40
            else COLOR_CRITICAL
        )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score_pct,
                number={"suffix": "%", "font": {"color": gauge_color, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "bgcolor": "#1E293B",
                    "steps": [
                        {"range": [0, 40], "color": "rgba(239, 68, 68, 0.15)"},
                        {"range": [40, 75], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [75, 100], "color": "rgba(16, 185, 129, 0.15)"},
                    ],
                },
                title={"text": "Asset Health Score", "font": {"size": 18}},
            )
        )

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)
