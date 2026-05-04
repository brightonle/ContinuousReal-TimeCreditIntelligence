"""
Claude API risk assessment layer.

Uses Claude with tool use to enforce structured RiskOutput output.
The system prompt is marked for prompt caching — it is large and static,
so caching it significantly reduces latency and cost across multiple runs.

Risk level escalation rule:
    final_risk = MAX(rule_engine_floor, claude_risk_level)

Claude can raise the risk level above what the rule engine detected but
can never lower it below the rule engine's hard-threshold determination.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_RISK_MODEL, WatchlistEntry
from risk.models import (
    RiskLevel,
    RiskOutput,
    Signal,
    SignalType,
    SourceRef,
)
from risk.prompts import (
    RISK_ASSESSMENT_SYSTEM_PROMPT,
    RISK_ASSESSMENT_USER_TEMPLATE,
    format_market_metrics,
    format_retrieved_context,
    format_rule_engine_summary,
)

logger = logging.getLogger(__name__)

# Tool schema that forces Claude to produce a structured RiskOutput
ASSESS_RISK_TOOL = {
    "name": "assess_risk",
    "description": (
        "Record your structured risk assessment for the company under review. "
        "You MUST call this tool — do not produce a free-text response."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "risk_level": {
                "type": "string",
                "enum": ["LOW", "ELEVATED", "HIGH"],
                "description": "Overall risk level based on all available evidence.",
            },
            "confidence_score": {
                "type": "number",
                "description": "Your confidence in this assessment from 0.0 (very uncertain) to 1.0 (very certain).",
            },
            "signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "signal_type": {
                            "type": "string",
                            "enum": [
                                "BALANCE_SHEET", "FUNDING", "EARNINGS", "MARKET",
                                "CREDIT_RATING", "REGULATORY", "MANAGEMENT", "MACRO", "OTHER",
                            ],
                        },
                        "description": {"type": "string"},
                        "severity": {"type": "string", "enum": ["LOW", "ELEVATED", "HIGH"]},
                        "metric_name": {"type": "string"},
                        "metric_value": {"type": "number"},
                        "source_document": {"type": "string"},
                        "source_date": {"type": "string"},
                        "source_section": {"type": "string"},
                        "excerpt": {"type": "string"},
                    },
                    "required": ["signal_type", "description", "severity"],
                },
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of your overall assessment rationale.",
            },
        },
        "required": ["risk_level", "confidence_score", "signals", "reasoning"],
        "additionalProperties": False,
    },
}


def assess_company_risk(
    company: WatchlistEntry,
    rule_signals: list[Signal],
    rule_floor: RiskLevel,
    rule_flags: list[str],
    dimension_docs: dict[str, list],
    market_metrics: dict,
    assessment_date: str | None = None,
) -> RiskOutput:
    """
    Call Claude to perform a structured risk assessment for a company.

    Args:
        company:         watchlist entry being assessed
        rule_signals:    signals already detected by the rule engine
        rule_floor:      minimum risk level set by the rule engine
        rule_flags:      list of threshold names that were triggered
        dimension_docs:  RAG-retrieved chunks keyed by risk dimension
        market_metrics:  structured market metrics dict from yfinance
        assessment_date: ISO-8601 datetime string (defaults to now)

    Returns:
        RiskOutput with risk_level = MAX(rule_floor, claude_level)
    """
    if assessment_date is None:
        assessment_date = datetime.utcnow().isoformat()

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # ── Build prompt context ──────────────────────────────────────────────────
    rule_summary = format_rule_engine_summary(rule_flags, rule_signals)
    metrics_summary = format_market_metrics(market_metrics)
    retrieved_context = format_retrieved_context(dimension_docs)

    user_message = RISK_ASSESSMENT_USER_TEMPLATE.format(
        company_name=company.company_name,
        ticker=company.ticker,
        industry=company.industry,
        sector=company.sector,
        assessment_date=assessment_date[:10],
        rule_engine_summary=rule_summary,
        market_metrics_summary=metrics_summary,
        retrieved_context=retrieved_context,
    )

    logger.info(
        "Calling Claude (%s) for risk assessment of %s",
        CLAUDE_RISK_MODEL,
        company.ticker,
    )

    # ── Claude API call with prompt caching + tool use ────────────────────────
    response = client.messages.create(
        model=CLAUDE_RISK_MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": RISK_ASSESSMENT_SYSTEM_PROMPT,
                # Cache the large, static system prompt — saves ~$0.002 per call
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[ASSESS_RISK_TOOL],
        tool_choice={"type": "tool", "name": "assess_risk"},
        messages=[{"role": "user", "content": user_message}],
    )

    logger.info(
        "Claude usage — input: %d, output: %d, cache_read: %d, cache_write: %d",
        response.usage.input_tokens,
        response.usage.output_tokens,
        getattr(response.usage, "cache_read_input_tokens", 0),
        getattr(response.usage, "cache_creation_input_tokens", 0),
    )

    # ── Parse tool use response ───────────────────────────────────────────────
    tool_use_block = next(
        (b for b in response.content if b.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        logger.warning(
            "Claude did not call assess_risk tool for %s — defaulting to LOW risk.",
            company.ticker,
        )
        return _fallback_output(company, rule_signals, rule_flags, rule_floor, assessment_date)

    raw: dict = tool_use_block.input  # already a dict from the SDK

    # ── Convert raw tool input to Pydantic models ─────────────────────────────
    claude_risk_level = RiskLevel(raw["risk_level"])

    # Escalation rule: take the MAX of rule engine floor and Claude's assessment
    final_risk_level = RiskLevel.max(rule_floor, claude_risk_level)

    if final_risk_level != claude_risk_level:
        logger.info(
            "Risk escalated for %s: Claude said %s but rule engine floor is %s → using %s",
            company.ticker,
            claude_risk_level.value,
            rule_floor.value,
            final_risk_level.value,
        )

    # Merge rule engine signals + Claude signals (avoid duplicates)
    all_signals: list[Signal] = list(rule_signals)  # start with hard-rule signals
    for raw_sig in raw.get("signals", []):
        # Skip if Claude is just restating a rule-triggered signal
        if raw_sig.get("metric_name") and any(
            s.metric_name == raw_sig.get("metric_name") and s.triggered_by_rule
            for s in rule_signals
        ):
            continue

        source_refs: list[SourceRef] = []
        if raw_sig.get("source_document") or raw_sig.get("source_date"):
            source_refs.append(
                SourceRef(
                    document_type=raw_sig.get("source_document", "document"),
                    document_date=raw_sig.get("source_date", ""),
                    section=raw_sig.get("source_section"),
                    excerpt=raw_sig.get("excerpt"),
                )
            )

        all_signals.append(
            Signal(
                signal_type=SignalType(raw_sig.get("signal_type", "OTHER")),
                description=raw_sig.get("description", ""),
                severity=RiskLevel(raw_sig.get("severity", "LOW")),
                metric_name=raw_sig.get("metric_name"),
                metric_value=raw_sig.get("metric_value"),
                source_refs=source_refs,
                triggered_by_rule=False,
            )
        )

    return RiskOutput(
        company=company.company_name,
        ticker=company.ticker,
        assessment_date=assessment_date,
        risk_level=final_risk_level,
        signals=all_signals,
        confidence_score=float(raw.get("confidence_score", 0.5)),
        rule_engine_flags=rule_flags,
        industry=company.industry,
        sector=company.sector,
        market_cap_bucket=company.market_cap_bucket,
        human_review_status="PENDING",
    )


def _fallback_output(
    company: WatchlistEntry,
    rule_signals: list[Signal],
    rule_flags: list[str],
    rule_floor: RiskLevel,
    assessment_date: str,
) -> RiskOutput:
    """Fallback when Claude fails to return a tool call."""
    return RiskOutput(
        company=company.company_name,
        ticker=company.ticker,
        assessment_date=assessment_date,
        risk_level=rule_floor,
        signals=rule_signals,
        confidence_score=0.3,
        rule_engine_flags=rule_flags,
        industry=company.industry,
        sector=company.sector,
        market_cap_bucket=company.market_cap_bucket,
        human_review_status="PENDING",
    )
