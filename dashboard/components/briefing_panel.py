"""
Briefing panel — shows the latest narrative briefing and source signals.
"""

from __future__ import annotations

import streamlit as st

from risk.models import RiskLevel, RiskOutput

RISK_BADGE_CSS = {
    "LOW":      "padding: 2px 8px; border-radius: 4px; background:#d4edda; color:#155724; font-weight:bold;",
    "ELEVATED": "padding: 2px 8px; border-radius: 4px; background:#fff3cd; color:#856404; font-weight:bold;",
    "HIGH":     "padding: 2px 8px; border-radius: 4px; background:#f8d7da; color:#721c24; font-weight:bold;",
}


def render_briefing_panel(risk_output: RiskOutput) -> None:
    """Render the briefing panel for a single company's risk output."""
    risk_css = RISK_BADGE_CSS.get(risk_output.risk_level.value, "")
    st.markdown(
        f"**{risk_output.company}** ({risk_output.ticker}) &nbsp;"
        f'<span style="{risk_css}">{risk_output.risk_level.value}</span> &nbsp;'
        f"Confidence: {risk_output.confidence_score:.0%} &nbsp;|&nbsp; "
        f"Assessed: {risk_output.assessment_date[:10]}",
        unsafe_allow_html=True,
    )

    # ── Narrative briefing ────────────────────────────────────────────────────
    if risk_output.narrative_summary:
        st.markdown("**Early Warning Briefing**")
        st.markdown(
            f'<div style="background:#f8f9fa; padding:16px; border-radius:8px; '
            f'border-left:4px solid #6c757d; font-size:0.95em;">'
            f"{risk_output.narrative_summary}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No narrative briefing generated yet.")

    # ── Signals ───────────────────────────────────────────────────────────────
    if risk_output.signals:
        st.markdown("**Detected Signals**")
        for sig in sorted(
            risk_output.signals,
            key=lambda s: ["LOW", "ELEVATED", "HIGH"].index(s.severity.value),
            reverse=True,
        ):
            icon = {"LOW": "🟢", "ELEVATED": "🟡", "HIGH": "🔴"}.get(sig.severity.value, "⚪")
            rule_tag = " *(rule trigger)*" if sig.triggered_by_rule else ""
            with st.expander(
                f"{icon} {sig.signal_type.value} — {sig.description[:80]}...{rule_tag}",
                expanded=sig.severity.value == "HIGH",
            ):
                st.write(sig.description)
                if sig.metric_name and sig.metric_value is not None:
                    st.caption(f"Metric: `{sig.metric_name}` = `{sig.metric_value:.4f}`")
                for ref in sig.source_refs:
                    st.caption(
                        f"Source: {ref.document_type}"
                        + (f" ({ref.document_date[:10]})" if ref.document_date else "")
                        + (f" — {ref.section}" if ref.section else "")
                    )
                    if ref.excerpt:
                        st.markdown(f"> {ref.excerpt[:300]}")

    # ── Rule engine flags ─────────────────────────────────────────────────────
    if risk_output.rule_engine_flags:
        st.markdown(
            "**Rule Engine Flags:** " + " · ".join(
                f"`{f}`" for f in risk_output.rule_engine_flags
            )
        )
