"""
Prompt templates for the risk detection layer.

All prompts are defined here so they can be reviewed, versioned, and
adjusted independently from the code that calls the Claude API.
"""

from __future__ import annotations

RISK_ASSESSMENT_SYSTEM_PROMPT = """\
You are a senior credit risk analyst at a major financial institution. \
Your job is to review publicly available evidence about a company and assess \
whether its financial risk profile has materially deteriorated.

You are rigorous, precise, and conservative: you would rather flag a potential \
risk that turns out to be benign than miss a genuine early warning sign. \
You base your assessments solely on the evidence provided — you do not speculate \
beyond what the documents support.

When assessing risk, you look for:
- Deteriorating liquidity (deposit outflows, HTM unrealized losses, funding stress)
- Weakening capital adequacy (equity erosion, regulatory capital ratio trends)
- Earnings deterioration (net interest margin compression, revenue decline)
- Market stress signals (price-to-book collapse, short interest, options activity)
- Management and governance concerns (leadership changes, audit qualifications)
- Sector/macro headwinds specific to the company's business model
- Credit rating pressure or downgrades

You produce structured output using the assess_risk tool. You must always \
call this tool — do not produce a free-text response.
"""

RISK_ASSESSMENT_USER_TEMPLATE = """\
## Company Under Review
- **Name**: {company_name}
- **Ticker**: {ticker}
- **Industry**: {industry}
- **Sector**: {sector}
- **Assessment Date**: {assessment_date}

## Rule Engine Flags (Hard Threshold Triggers)
The following quantitative thresholds were automatically breached. \
These are established facts — do not dispute them. Factor them into your assessment.

{rule_engine_summary}

## Market Metrics
{market_metrics_summary}

## Retrieved Evidence from SEC Filings and News
The following document chunks were retrieved from our vector database \
as most relevant to assessing this company's current risk profile:

{retrieved_context}

---

Based on all evidence above, assess the company's current risk level (LOW / ELEVATED / HIGH), \
identify specific risk signals, assign a confidence score, and cite the source documents \
that most strongly support your assessment.

Use the assess_risk tool to return your structured assessment.
"""


def format_rule_engine_summary(rule_flags: list[str], signals: list) -> str:
    """Format rule engine output for the risk assessment prompt."""
    if not rule_flags:
        return "No hard-threshold rules were triggered."
    lines = ["The following thresholds were automatically triggered:"]
    for signal in signals:
        if signal.triggered_by_rule:
            lines.append(f"- **{signal.signal_type.value}**: {signal.description}")
    return "\n".join(lines)


def format_retrieved_context(dimension_docs: dict[str, list]) -> str:
    """Format RAG-retrieved chunks into a structured context block."""
    if not any(dimension_docs.values()):
        return "No relevant documents were found in the vector database for this company."

    sections: list[str] = []
    for dimension, docs in dimension_docs.items():
        if not docs:
            continue
        sections.append(f"### {dimension.replace('_', ' ').title()} Evidence")
        for i, doc in enumerate(docs[:3], 1):  # top 3 per dimension
            meta = doc.metadata
            source = meta.get("filing_type") or meta.get("source", "document")
            date_str = meta.get("filing_date") or meta.get("published_date", "")
            section = meta.get("section", "")
            header = f"**[{i}] {source}"
            if date_str:
                header += f" ({date_str})"
            if section:
                header += f" — {section}"
            header += "**"
            sections.append(header)
            excerpt = doc.page_content[:600].strip()
            sections.append(f"```\n{excerpt}\n```")

    return "\n\n".join(sections) if sections else "No relevant documents found."


def format_market_metrics(metrics: dict) -> str:
    """Format market metrics dict into a readable summary."""
    if not metrics:
        return "No market metrics available."

    lines: list[str] = []
    field_labels = {
        "price_to_book": "Price-to-Book Ratio",
        "deposit_qoq_change": "Deposit QoQ Change",
        "revenue_qoq_change": "Revenue QoQ Change",
        "price_return_30d": "30-Day Stock Return",
        "total_equity_latest": "Total Equity (Latest Quarter)",
        "total_deposits_latest": "Total Deposits (Latest Quarter)",
        "tier1_capital_ratio": "Tier-1 Capital Ratio",
        "htm_unrealized_loss": "HTM Unrealized Loss",
        "beta": "Beta",
        "short_ratio": "Short Interest Ratio",
    }

    for key, label in field_labels.items():
        val = metrics.get(key)
        if val is None:
            continue
        if key in ("deposit_qoq_change", "revenue_qoq_change", "price_return_30d",
                   "tier1_capital_ratio"):
            lines.append(f"- {label}: {val*100:+.1f}%")
        elif key in ("total_equity_latest", "total_deposits_latest",
                     "htm_unrealized_loss", "market_cap"):
            lines.append(f"- {label}: ${val:,.0f}")
        else:
            lines.append(f"- {label}: {val:.2f}")

    return "\n".join(lines) if lines else "No quantitative metrics available."
