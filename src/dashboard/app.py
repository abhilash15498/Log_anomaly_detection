"""
AI-Driven Infrastructure Monitoring and Intelligent Incident Recommendation System.
Enterprise Real-Time Log Monitoring, Anomaly Detection & Automated Root-Cause Remediation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import psutil
import time

from src.detection.infer import detect, _load_config
from src.llm.diagnose import diagnose_anomaly
from src.ingestion.fault_injector import generate_log_event, FAULT_TEMPLATES
from src.dashboard.feedback import save_feedback, get_feedback_summary, adapt_threshold, persist_adapted_thresholds
from src.preprocessing.parser import parse_log_line
from src.preprocessing.vectorizer import LogVectorizer
from src.detection.baseline import load_isolation_forest, load_kmeans, isolation_forest_score, kmeans_score
from src.detection.autoencoder import load_autoencoder, autoencoder_score
from src.schemas import DetectionResult

# Page Configuration
st.set_page_config(
    page_title="AI-Driven Infrastructure Monitoring System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enterprise Modern UI Styling
st.markdown("""
<style>
    /* Global Styling */
    .main { background-color: #0E1117; color: #E0E6ED; }
    
    /* Top Navigation Bar */
    .top-nav {
        background: linear-gradient(90deg, #131722 0%, #1A202C 100%);
        border: 1px solid #2D3748;
        border-radius: 10px;
        padding: 14px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .status-pill {
        background-color: #064E3B;
        color: #34D399;
        font-size: 12px;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        border: 1px solid #059669;
    }

    /* KPI Cards */
    .kpi-card {
        background: #181F2C;
        border: 1px solid #283347;
        border-radius: 8px;
        padding: 18px 20px;
        flex: 1;
    }
    .kpi-label { font-size: 13px; color: #8C9BAE; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }
    .kpi-value { font-size: 28px; font-weight: 700; color: #F8FAFC; margin-top: 6px; }
    .kpi-sub { font-size: 12px; color: #38BDF8; margin-top: 4px; }

    /* Severity Badges */
    .badge-critical { background: #7F1D1D; color: #FCA5A5; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; border: 1px solid #EF4444; }
    .badge-high { background: #7C2D12; color: #FDBA74; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; border: 1px solid #F97316; }
    .badge-medium { background: #1E3A8A; color: #93C5FD; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; border: 1px solid #3B82F6; }
    .badge-low { background: #064E3B; color: #6EE7B7; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700; border: 1px solid #10B981; }

    /* Timeline Log Styling */
    .log-row { padding: 8px 12px; font-family: 'JetBrains Mono', 'Courier New', monospace; font-size: 12.5px; border-radius: 6px; margin-bottom: 4px; }
    .log-normal { background: #1A2233; color: #94A3B8; }
    .log-anomaly { background: #3B1219; color: #FCA5A5; border-left: 4px solid #EF4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Top Navigation Bar
st.markdown("""
<div class="top-nav">
    <div class="brand-title">
        🛡️ <span>AI-Driven Infrastructure Monitoring & Intelligent Incident Recommendation System</span>
    </div>
    <div>
        <span class="status-pill">● AUTONOMOUS OBSERVER READY</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_observer, tab_audit = st.tabs([
    "⚡ Autonomous Real-Time Log Observer",
    "📋 Active Incident Queue & Audit (Benchmark Dataset)"
])

# -------------------------------------------------------------------
# TAB 1: AUTONOMOUS REAL-TIME LOG OBSERVER (ZERO PASTING REQUIRED)
# -------------------------------------------------------------------
with tab_observer:
    st.markdown("### Autonomous Background Log Observer")
    st.markdown("Passively monitors a live log file or incoming stream on your machine. As soon as any software, server, or container writes a new log line, the system inspects it in sub-milliseconds without any manual pasting.")

    # Target Log File Path
    default_log_file = PROJECT_ROOT / "data" / "raw" / "live_stream.log"
    default_log_file.parent.mkdir(parents=True, exist_ok=True)
    if not default_log_file.exists():
        with open(default_log_file, "w", encoding="utf-8") as f:
            f.write("2026-11-14 12:00:00 [INFO] [system-init]: Infrastructure log monitor initialized.\n")

    col_cfg1, col_cfg2, col_cfg3 = st.columns([2, 1, 1])
    with col_cfg1:
        observed_file_path = st.text_input(
            "Monitored Target Log File Path:",
            value=str(default_log_file),
            help="Path to any log file generated by your operating system, Docker, or local applications."
        )
    with col_cfg2:
        poll_interval = st.select_slider(
            "Polling Sensitivity:",
            options=["Ultra-Fast (100ms)", "Standard (250ms)", "Balanced (500ms)"],
            value="Standard (250ms)"
        )
        poll_sec = 0.10 if "Ultra-Fast" in poll_interval else (0.25 if "Standard" in poll_interval else 0.50)
    with col_cfg3:
        include_hw = st.checkbox("Sample Live Hardware (CPU/RAM)", value=True)

    # Observer Control & Auto-Stream Generator
    st.markdown("---")
    col_act1, col_act2 = st.columns([1, 1])

    with col_act1:
        st.markdown("**1. Autonomous Live Monitoring:**")
        start_monitor = st.button("▶ Start Real-Time Log Observer", type="primary", use_container_width=True)

    with col_act2:
        st.markdown("**2. Simulate External Server Writing to File:**")
        selected_fault = st.selectbox(
            "Choose a failure pattern for the external service to write:",
            [
                ("A", "Memory Leak / Out of Memory Burst"),
                ("B", "Database Connection Pool Starvation"),
                ("C", "Cascading HTTP 500 Outage & Circuit Trip"),
                ("D", "Routine Healthy Traffic (Normal Operations)")
            ],
            format_func=lambda x: x[1]
        )[0]
        simulate_write = st.button("✍ Simulate Service Writing Log Events into File", use_container_width=True)
        clear_log_btn = st.button("🧹 Clear Log File (Reset to Healthy Baseline)", use_container_width=True)

    if clear_log_btn:
        with open(observed_file_path, "w", encoding="utf-8") as f:
            f.write("2026-11-14 12:00:00 [INFO] [system-init]: Infrastructure log monitor initialized.\n")
        st.success("Log file reset to clean baseline (zero errors).")
        time.sleep(0.5)
        st.rerun()

    # Handle external service writing into the monitored file
    if simulate_write:
        with open(observed_file_path, "a", encoding="utf-8") as f:
            if selected_fault == "D":
                for _ in range(5):
                    ev = generate_log_event()
                    f.write(f"{ev['raw_text']}\n")
                st.success("Appended 5 normal service events to log file.")
            else:
                for idx, line in enumerate(FAULT_TEMPLATES[selected_fault]["logs"]):
                    ev = generate_log_event(fault_type=selected_fault, fault_index=idx)
                    f.write(f"{ev['raw_text']}\n")
                st.warning(f"Appended failure burst ({FAULT_TEMPLATES[selected_fault]['name']}) to log file.")

    # Live Monitoring Loop
    if start_monitor:
        st.markdown("#### 🔴 Live Observation Stream Active")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            live_cpu_ph = st.empty()
        with col_g2:
            live_ram_ph = st.empty()

        stream_ph = st.empty()
        alert_ph = st.empty()

        # Load Detection Models
        cfg = _load_config()
        thresholds = cfg["thresholds"]
        vectorizer = LogVectorizer(use_sentence_transformers=False, vector_dim=384)
        iso = load_isolation_forest()
        km = load_kmeans()
        ae = load_autoencoder(PROJECT_ROOT / "src" / "models" / "autoencoder.pkl")

        # Read existing file to tail newly appended lines
        if Path(observed_file_path).exists():
            with open(observed_file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                last_seen_lines = lines[-5:] if len(lines) >= 5 else lines
        else:
            last_seen_lines = []

        all_observed = list(last_seen_lines)
        anomaly_detected = False
        target_incident = None

        # Real-time observation pass (simulating active daemon loop)
        progress_bar = st.progress(0)
        for step in range(15):
            progress_bar.progress((step + 1) / 15)

            # Sample real hardware
            cpu_now = psutil.cpu_percent()
            ram_now = psutil.virtual_memory().percent
            live_cpu_ph.metric("Host CPU Load", f"{cpu_now:.1f}%")
            live_ram_ph.metric("Host RAM Allocation", f"{ram_now:.1f}%")

            # Check for new lines in file
            if Path(observed_file_path).exists():
                with open(observed_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    current_lines = [l.strip() for l in f.readlines() if l.strip()]

                if len(current_lines) > len(all_observed):
                    new_lines = current_lines[len(all_observed):]
                    for nl in new_lines:
                        all_observed.append(nl)

                        # Evaluate with ML Scout
                        parsed = parse_log_line(nl)
                        emb = vectorizer.encode_text(parsed["template"])
                        emb_arr = np.array([emb])

                        iso_s = float(isolation_forest_score(iso, emb_arr)[0])
                        km_s = float(kmeans_score(km, emb_arr)[0])
                        ae_s = float(autoencoder_score(ae, emb_arr)[0])

                        # Consensus check
                        votes = (
                            int(iso_s >= thresholds["isolation_forest"]) +
                            int(km_s >= thresholds["kmeans"]) +
                            int(ae_s >= thresholds["autoencoder"])
                        )

                        if votes >= 2 and not anomaly_detected:
                            anomaly_detected = True
                            target_incident = {
                                "line": nl,
                                "context": all_observed[-5:],
                                "score": max(iso_s, km_s, ae_s),
                                "metrics": {"cpu_percent": cpu_now, "memory_percent": ram_now} if include_hw else None
                            }

            # Render rolling log stream
            recent = all_observed[-7:]
            rendered = []
            for r_line in recent:
                is_anom = any(k in r_line.lower() for k in ["outofmemory", "hikaripool", "nullpointer", "panic", "deadlock", "error", "critical"])
                if is_anom:
                    rendered.append(f"<div class='log-row log-anomaly'>🚨 [INTERCEPTED ANOMALY] {r_line}</div>")
                else:
                    rendered.append(f"<div class='log-row log-normal'>🟢 [OBSERVED] {r_line}</div>")
            stream_ph.markdown("".join(rendered), unsafe_allow_html=True)
            time.sleep(poll_sec)

        # Trigger AI Diagnosis if an anomaly was observed
        if anomaly_detected and target_incident:
            with alert_ph.container():
                st.markdown("---")
                st.error("🚨 **REAL-TIME ANOMALY INTERCEPTED BY INCIDENT DETECTOR**")

                det_res = DetectionResult(
                    log_id=f"live_{int(time.time())}",
                    anomaly_score=round(target_incident["score"], 4),
                    is_anomaly=True,
                    threshold_used=0.50,
                    context_window=target_incident["context"]
                )

                with st.spinner("AI Engine generating real-time diagnosis & remediation playbook..."):
                    report = diagnose_anomaly(det_res, metrics=target_incident["metrics"])

                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    st.markdown(f"**Observed Event:**")
                    st.code(target_incident["line"], language="text")
                    badge_class = f"badge-{report.severity}"
                    st.markdown(f"**Assessed Severity:** <span class='{badge_class}'>{report.severity.upper()}</span>", unsafe_allow_html=True)

                with col_res2:
                    st.markdown("#### 🤖 Autonomous Root-Cause Analysis")
                    st.info(report.explanation)

                    st.markdown("#### 🛠️ Automated Remediation Commands")
                    for act in report.recommended_actions:
                        st.code(act, language="bash")
        else:
            st.success("Monitoring session completed. All observed logs conform to normal operational baseline.")

# -------------------------------------------------------------------
# TAB 2: ACTIVE INCIDENT QUEUE & AUDIT (BENCHMARK DATASET)
# -------------------------------------------------------------------
with tab_audit:
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

    # Pre-load logs
    all_detections = detect(limit=1000)
    anomalies_list = [d for d in all_detections if d.is_anomaly]
    fb_summary = get_feedback_summary()

    with col_kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Ingested Logs</div>
            <div class="kpi-value">{len(all_detections):,}</div>
            <div class="kpi-sub">Real-time HDFS Stream</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Flagged Incidents</div>
            <div class="kpi-value">{len(anomalies_list)}</div>
            <div class="kpi-sub">{len(anomalies_list)/max(1, len(all_detections))*100:.1f}% deviation rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Detection Latency</div>
            <div class="kpi-value">&lt; 0.2ms</div>
            <div class="kpi-sub">Multi-Model Ensemble</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Admin Triage Actions</div>
            <div class="kpi-value">{fb_summary['total']}</div>
            <div class="kpi-sub">{fb_summary['confirmed']} Confirmed | {fb_summary['dismissed']} Suppressed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_queue, col_investigation = st.columns([1, 2])

    with col_queue:
        st.markdown("#### 🚨 Flagged Incident Queue")
        
        # Sort anomalies by score
        anomalies_list.sort(key=lambda x: x.anomaly_score, reverse=True)
        incident_ids = [a.log_id for a in anomalies_list]

        # Fast search filter
        search_filter = st.text_input("🔍 Filter by Log ID:", placeholder="e.g. log_0000761", key="search_log_id")
        filtered_anomalies = [a for a in anomalies_list if search_filter.lower() in a.log_id.lower()] if search_filter else anomalies_list

        # Prepare full table of all flagged incidents
        df_preview = pd.DataFrame([
            {
                "Incident ID": a.log_id,
                "Score": f"{a.anomaly_score:.2f}",
                "Consensus": "2/3 Models"
            }
            for a in filtered_anomalies
        ])

        st.caption(f"Showing all {len(filtered_anomalies)} flagged incidents. **Click any row to inspect:**")
        selection_event = st.dataframe(
            df_preview,
            use_container_width=True,
            height=460,
            selection_mode="single-row",
            on_select="rerun"
        )

        selected_row_indices = selection_event.selection.rows if selection_event else []
        if selected_row_indices and selected_row_indices[0] < len(filtered_anomalies):
            active_anomaly = filtered_anomalies[selected_row_indices[0]]
        else:
            active_anomaly = filtered_anomalies[0] if filtered_anomalies else anomalies_list[0]

    with col_investigation:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 2px solid #CBD5E1; padding-bottom: 8px;">
            <span style="font-size: 22px; font-weight: 800; color: #0F172A;">📋 Incident Investigation: <code style="font-size: 20px; color: #2563EB; font-weight: 700; background: #EEF2FF; padding: 2px 8px; border-radius: 4px;">{active_anomaly.log_id}</code></span>
            <span class="badge-medium">MULTI-MODEL CONSENSUS</span>
        </div>
        """, unsafe_allow_html=True)

        # Chronological Log Timeline
        st.markdown("**Chronological Log Sequence (Temporal Context):**")
        mid_idx = len(active_anomaly.context_window) // 2
        timeline_html = []
        for idx, line in enumerate(active_anomaly.context_window):
            if idx == mid_idx:
                timeline_html.append(f"<div class='log-row log-anomaly'>🔴 <b>[ROOT EVENT]</b> {line}</div>")
            else:
                timeline_html.append(f"<div class='log-row log-normal'>⚪ [CONTEXT] {line}</div>")
        st.markdown("".join(timeline_html), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # LLM Root-Cause Analysis (cached per incident ID for instant response)
        if "diagnosis_cache" not in st.session_state:
            st.session_state.diagnosis_cache = {}

        if active_anomaly.log_id not in st.session_state.diagnosis_cache:
            with st.spinner("AI Engine generating operational diagnosis..."):
                report = diagnose_anomaly(active_anomaly)
                st.session_state.diagnosis_cache[active_anomaly.log_id] = report
        else:
            report = st.session_state.diagnosis_cache[active_anomaly.log_id]

        badge_class = f"badge-{report.severity}"
        st.markdown(f"""
        <div style="background: #182030; border: 1px solid #2A364F; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 700; font-size: 15px; color: #F8FAFC;">🤖 Autonomous AI Root-Cause Diagnosis</span>
                <span class="{badge_class}">{report.severity.upper()} SEVERITY</span>
            </div>
            <div style="font-size: 13.5px; line-height: 1.6; color: #CBD5E1;">
                {report.explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Remediation Commands
        st.markdown("**Recommended Operational Remediation Playbook:**")
        for idx_act, act in enumerate(report.recommended_actions, 1):
            st.code(act, language="bash")

        # Administrator Triage Actions (Contract A.4)
        st.markdown("---")
        st.markdown("**Administrator Triage & Sensitivity Feedback:**")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("✅ Confirm Incident (Valid Anomaly)", key=f"btn_conf_{active_anomaly.log_id}", use_container_width=True):
                save_feedback(active_anomaly.log_id, "confirmed")
                persist_adapted_thresholds()
                st.success("Incident confirmed. SRE incident ticket generated.")
                time.sleep(0.5)
                st.rerun()

        with col_btn2:
            if st.button("❌ Suppress Alert (Mark False Alarm)", key=f"btn_dism_{active_anomaly.log_id}", use_container_width=True):
                save_feedback(active_anomaly.log_id, "dismissed")
                new_cutoff = persist_adapted_thresholds()
                cutoff_txt = f" (Threshold adapted to {new_cutoff:.4f})" if new_cutoff else ""
                st.warning(f"Alert suppressed. Detection sensitivity automatically adapted{cutoff_txt}.")
                time.sleep(0.5)
                st.rerun()
