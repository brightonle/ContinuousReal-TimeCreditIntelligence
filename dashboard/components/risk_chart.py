"""
Risk level history chart — Plotly time-series per company.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from risk.models import RiskLevel, RiskOutput

RISK_NUMERIC = {RiskLevel.LOW: 0, RiskLevel.ELEVATED: 1, RiskLevel.HIGH: 2}
RISK_LABELS = {0: "LOW", 1: "ELEVATED", 2: "HIGH"}
RISK_COLORS_PLOTLY = {0: "#28a745", 1: "#ffc107", 2: "#dc3545"}


def render_risk_chart(history: list[RiskOutput], ticker: str) -> None:
    """Render a time-series chart of risk level escalation for a company."""
    if not history:
        st.info(f"No historical records found for {ticker}.")
        return

    sorted_history = sorted(history, key=lambda o: o.assessment_date)
    dates = [o.assessment_date[:10] for o in sorted_history]
    numeric_risk = [RISK_NUMERIC[o.risk_level] for o in sorted_history]
    confidence = [o.confidence_score for o in sorted_history]

    fig = go.Figure()

    # Color each point by risk level
    colors = [RISK_COLORS_PLOTLY[r] for r in numeric_risk]

    fig.add_trace(go.Scatter(
        x=dates,
        y=numeric_risk,
        mode="lines+markers",
        name="Risk Level",
        line=dict(color="#6c757d", width=2),
        marker=dict(size=10, color=colors),
        customdata=list(zip(
            [RISK_LABELS[r] for r in numeric_risk],
            [f"{c:.0%}" for c in confidence],
        )),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Risk: %{customdata[0]}<br>"
            "Confidence: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=f"Risk Level History — {ticker}",
        xaxis_title="Assessment Date",
        yaxis=dict(
            tickvals=[0, 1, 2],
            ticktext=["LOW", "ELEVATED", "HIGH"],
            range=[-0.5, 2.5],
        ),
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)
