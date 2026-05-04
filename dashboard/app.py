"""
Early Warning Financial Intelligence System — Streamlit Dashboard

Run with:
    streamlit run dashboard/app.py

The dashboard is purely a read layer — it reads from the SQLite audit log
and does not re-run any inference. This means it works even when the
ingestion pipeline is not running.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running via streamlit run
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from config import DEFAULT_WATCHLIST
from dashboard.components.audit_panel import render_audit_panel
from dashboard.components.briefing_panel import render_briefing_panel
from dashboard.components.risk_chart import render_risk_chart
from dashboard.components.watchlist_table import render_watchlist_table
from output.audit_log import AuditLog
from risk.models import RiskLevel

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Early Warning Financial Intelligence",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚠️ EWFI System")
    st.caption("Early Warning Financial Intelligence")
    st.divider()

    st.markdown("**Watchlist**")
    for entry in DEFAULT_WATCHLIST:
        st.caption(f"• {entry.ticker} — {entry.company_name}")

    st.divider()
    page = st.radio(
        "View",
        options=["Watchlist", "Company Detail", "Audit Log"],
        index=0,
    )

    if st.button("🔄 Refresh"):
        st.rerun()

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def load_latest_outputs() -> list:
    log = AuditLog()
    return log.get_latest_per_company()


@st.cache_resource
def get_audit_log() -> AuditLog:
    return AuditLog()


latest_outputs = load_latest_outputs()
audit_log = get_audit_log()

# ── Risk summary metrics ───────────────────────────────────────────────────────
if latest_outputs:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Companies Monitored", len(DEFAULT_WATCHLIST))
    with col2:
        high_count = sum(1 for o in latest_outputs if o.risk_level == RiskLevel.HIGH)
        st.metric("🔴 HIGH Risk", high_count, delta=None)
    with col3:
        elevated_count = sum(1 for o in latest_outputs if o.risk_level == RiskLevel.ELEVATED)
        st.metric("🟡 ELEVATED Risk", elevated_count)
    with col4:
        total_records = audit_log.count()
        st.metric("Total Assessments", total_records)

    st.divider()

# ── Page routing ───────────────────────────────────────────────────────────────
if page == "Watchlist":
    st.header("Company Watchlist")
    render_watchlist_table(latest_outputs)

elif page == "Company Detail":
    st.header("Company Risk Detail")

    tickers = [o.ticker for o in latest_outputs]
    if not tickers:
        st.warning("No assessments available yet. Run the ingestion pipeline first.")
    else:
        selected_ticker = st.selectbox("Select Company", options=tickers)

        # Latest assessment
        current = next((o for o in latest_outputs if o.ticker == selected_ticker), None)
        if current:
            render_briefing_panel(current)
            st.divider()

        # Historical chart
        history = audit_log.get_history(ticker=selected_ticker, limit=50)
        if history:
            render_risk_chart(history, selected_ticker)

elif page == "Audit Log":
    render_audit_panel(audit_log)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Early Warning Financial Intelligence System · "
    "Data sources: SEC EDGAR, NewsAPI, Yahoo Finance · "
    "All assessments require human review before action."
)
