"""
Watchlist table component — shows each company with color-coded risk badge.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from risk.models import RiskLevel, RiskOutput

RISK_COLORS = {
    RiskLevel.LOW: "🟢",
    RiskLevel.ELEVATED: "🟡",
    RiskLevel.HIGH: "🔴",
}

RISK_CSS = {
    "LOW": "background-color: #d4edda; color: #155724;",
    "ELEVATED": "background-color: #fff3cd; color: #856404;",
    "HIGH": "background-color: #f8d7da; color: #721c24;",
}


def render_watchlist_table(outputs: list[RiskOutput]) -> None:
    """Render the watchlist as a styled DataFrame."""
    if not outputs:
        st.info("No risk assessments available yet. Run the pipeline to populate the watchlist.")
        return

    rows = []
    for o in outputs:
        icon = RISK_COLORS.get(o.risk_level, "⚪")
        rows.append({
            "Ticker": o.ticker,
            "Company": o.company,
            "Risk Level": f"{icon} {o.risk_level.value}",
            "Confidence": f"{o.confidence_score:.0%}",
            "Signals": len(o.signals),
            "Rule Flags": len(o.rule_engine_flags),
            "Last Assessed": o.assessment_date[:10] if o.assessment_date else "—",
            "Review": o.human_review_status,
        })

    df = pd.DataFrame(rows)

    def _style_row(row: pd.Series) -> list[str]:
        risk_word = row["Risk Level"].split()[-1]
        css = RISK_CSS.get(risk_word, "")
        return [css] * len(row)

    st.dataframe(
        df.style.apply(_style_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )
