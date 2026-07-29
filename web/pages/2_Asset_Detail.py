"""Asset Monitor & Diagnostic Waveform Scorer Page."""

import numpy as np
import streamlit as st

from web.api_client import fetch_asset_detail, fetch_assets, post_score
from web.bootstrap import init_page
from web.components.charts import (
    render_feature_importance,
    render_health_gauge,
    render_waveform_plot,
)

# Must be called first on script execution
init_page()

# Page Header
col_title, col_sync = st.columns([4, 1])
with col_title:
    st.title("Asset Monitor & Diagnostic Scorer")
    st.caption(
        "Inspect asset condition, analyze vibration waveforms, and run ML scoring diagnostics"
    )

with col_sync:
    st.write("")
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

base_url = str(st.session_state.get("api_base_url", "http://localhost:8000"))
api_online = bool(st.session_state.get("api_online", False))

assets = fetch_assets(base_url) if api_online else []
asset_ids: list[str] = [
    str(a.get("asset_id")) for a in assets if a.get("asset_id")
] if assets else ["motor-001", "pump-101"]

# Asset Selection Controls
col_select, col_custom = st.columns([2, 1])
with col_select:
    selected_idx = 0
    saved_id = str(st.session_state.get("selected_asset_id", ""))
    if saved_id in asset_ids:
        selected_idx = asset_ids.index(saved_id)

    selected_asset_id: str = str(st.selectbox(
        "Select Asset to Monitor / Score:",
        options=asset_ids,
        index=selected_idx,
    ))
    st.session_state["selected_asset_id"] = selected_asset_id

with col_custom:
    custom_id = st.text_input("Or enter Custom Asset ID:", value="")
    if custom_id.strip():
        selected_asset_id = custom_id.strip()
        st.session_state["selected_asset_id"] = selected_asset_id

# Fetch Asset Metadata
asset_detail = fetch_asset_detail(base_url, selected_asset_id) if api_online else None

st.markdown("---")

# Section 1: Asset Metadata & Diagnostics Overview
col_meta, col_gauge = st.columns([1, 1])

with col_meta:
    st.subheader(f"Asset Metadata: {selected_asset_id}")
    if asset_detail:
        st.json({
            "asset_id": asset_detail.get("asset_id"),
            "organization": asset_detail.get("org", "default"),
            "plant": asset_detail.get("plant", "default"),
            "asset_type": asset_detail.get("asset_type") or "Bearing",
            "nominal_rpm": asset_detail.get("rpm_nominal") or 1772.0,
            "anomaly_threshold": asset_detail.get("anomaly_threshold", 0.8),
            "created_at": str(asset_detail.get("created_at", "")),
        })
    else:
        st.info("No recorded metadata for this asset yet. Scoring a signal will register metadata.")

with col_gauge:
    last_result = st.session_state.get("last_score_result")
    current_health = (
        float(last_result["health_score"])
        if last_result and last_result.get("health_score") is not None
        else None
    )
    render_health_gauge(current_health)

    # Model Version Lineage Display
    if last_result and "model_versions" in last_result:
        m_vers = last_result["model_versions"]
        anom_v = m_vers.get("anomaly", "?")
        fault_v = m_vers.get("fault", "?")
        st.caption(f"Active Model Lineage — Anomaly: v{anom_v} | Fault: v{fault_v}")

st.markdown("---")

# Section 2: Waveform Signal Generator & Scorer Workbench
st.subheader("Signal Scorer Workbench")
st.markdown(
    "Supply a vibration waveform signal to score against anomaly detection and "
    "fault classification models."
)

tab_synthetic, tab_upload = st.tabs([
    "Synthetic Bearing Signal Generator",
    "Upload Waveform File (.csv / .npy)",
])

waveform_data: list[float] = []
sampling_rate_val = 12000.0
rpm_val = 1772.0

with tab_synthetic:
    col_fault, col_params = st.columns([1, 1])
    with col_fault:
        fault_mode = st.selectbox(
            "Simulate Bearing State:",
            options=["Healthy Signal", "Inner Race Fault", "Outer Race Fault", "Ball Defect"],
        )
    with col_params:
        sampling_rate_val = float(st.number_input(
            "Sampling Rate (Hz):",
            min_value=1000.0,
            max_value=50000.0,
            value=12000.0,
            step=1000.0,
        ))
        rpm_val = float(st.number_input(
            "Shaft Speed (RPM):",
            min_value=100.0,
            max_value=10000.0,
            value=1772.0,
            step=50.0,
        ))

    # Generate signal samples
    t = np.linspace(0, 0.2, int(sampling_rate_val * 0.2))  # 0.2s window
    noise = np.random.normal(0, 0.05, len(t))

    if fault_mode == "Healthy Signal":
        sig = 0.1 * np.sin(2 * np.pi * 30 * t) + noise
    elif fault_mode == "Inner Race Fault":
        bpfi = 160.0  # Hz impact freq
        pulses = np.exp(-((t % (1 / bpfi)) / 0.002) ** 2)
        sig = 0.5 * pulses * np.sin(2 * np.pi * 3000 * t) + noise * 1.5
    elif fault_mode == "Outer Race Fault":
        bpfo = 105.0
        pulses = np.exp(-((t % (1 / bpfo)) / 0.003) ** 2)
        sig = 0.4 * pulses * np.sin(2 * np.pi * 2500 * t) + noise * 1.2
    else:  # Ball Defect
        bsf = 70.0
        pulses = np.exp(-((t % (1 / bsf)) / 0.004) ** 2)
        sig = 0.3 * pulses * np.sin(2 * np.pi * 2000 * t) + noise * 1.1

    waveform_data = [float(x) for x in sig]

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload CSV or NPY vibration waveform:",
        type=["csv", "npy"],
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".npy"):
                arr = np.load(uploaded_file)
                waveform_data = [float(x) for x in arr.flatten()]
            else:
                arr = np.loadtxt(uploaded_file, delimiter=",")
                waveform_data = [float(x) for x in arr.flatten()]
            st.success(f"Loaded {len(waveform_data)} waveform samples from file.")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {e}")

# Action Button: Run Scoring
if st.button("Score Asset Waveform", type="primary", use_container_width=True):
    if not api_online:
        st.error("Cannot score signal — API is offline.")
    elif not waveform_data:
        st.error("Waveform data is empty.")
    else:
        with st.spinner(f"Scoring waveform for '{selected_asset_id}'..."):
            result = post_score(
                base_url=base_url,
                asset_id=selected_asset_id,
                waveform=waveform_data,
                sampling_rate=sampling_rate_val,
                rpm=rpm_val,
            )
            if result:
                st.session_state["last_score_result"] = result
                st.success(f"Waveform successfully scored. Score ID: {result.get('score_id')}")
                st.rerun()

# Section 3: Diagnostic Scoring Results Display
last_result = st.session_state.get("last_score_result")

if last_result:
    st.markdown("---")
    st.subheader("Diagnostic Scoring Results")

    res_col1, res_col2, res_col3 = st.columns(3)

    health_val = float(last_result.get("health_score", 0.0))
    anom_val = float(last_result.get("anomaly_score", 0.0))
    fault_info = last_result.get("fault") or {}
    alert_info = last_result.get("alert") or {}

    res_col1.metric("Health Score", f"{health_val * 100:.1f}%")
    res_col2.metric("Anomaly Score", f"{anom_val:.4f}")

    fault_cls = fault_info.get("class", "N/A")
    fault_conf = float(fault_info.get("confidence", 0.0))
    res_col3.metric("Predicted Fault Class", fault_cls, f"Confidence: {fault_conf * 100:.1f}%")

    alert_lvl = str(alert_info.get("level", "healthy")).upper()
    alert_reason = alert_info.get("reason", "N/A")

    if alert_lvl == "CRITICAL":
        st.error(f"ALERT LEVEL: CRITICAL — Reason: {alert_reason}")
    elif alert_lvl == "WARNING":
        st.warning(f"ALERT LEVEL: WARNING — Reason: {alert_reason}")
    else:
        st.success("ALERT LEVEL: HEALTHY — Machine operating normally.")

    # Charts: Feature Importance & Waveform Plot
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        render_feature_importance(last_result.get("top_features", []))

    with chart_col2:
        render_waveform_plot(waveform_data, sampling_rate_val)
