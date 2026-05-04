"""
Audit log panel and disparity ratio table.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from output.audit_log import AuditLog
from output.disparity_monitor import calculate_disparity_ratios, disparity_summary_table


def render_audit_panel(audit_log: AuditLog) -> None:
    """Render recent audit log entries and the disparity ratio table."""
    st.subheader("Audit Log")

    records = audit_log.get_all_records(limit=200)
    if not records:
        st.info("No audit records yet.")
        return

    # ── Recent records ────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    display_cols = [
        "timestamp", "ticker", "company", "risk_level",
        "confidence_score", "human_review_status",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[existing_cols], use_container_width=True, hide_index=True)

    # ── Disparity monitoring ──────────────────────────────────────────────────
    st.subheader("Disparity Monitoring")
    st.caption(
        "Four-Fifths Rule adapted for algorithmic flagging. "
        "Ratio > 1.25 triggers an alert. Requires ≥3 assessments per group for reliability."
    )

    results = calculate_disparity_ratios(audit_log)
    if not results:
        st.info("Not enough data for disparity analysis yet.")
        return

    table = disparity_summary_table(results)
    disp_df = pd.DataFrame(table)

    def _highlight_alert(row: pd.Series) -> list[str]:
        if "FLAGGED" in str(row.get("Alert", "")):
            return ["background-color: #f8d7da"] * len(row)
        if "Low sample" in str(row.get("Alert", "")):
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        disp_df.style.apply(_highlight_alert, axis=1),
        use_container_width=True,
        hide_index=True,
    )
