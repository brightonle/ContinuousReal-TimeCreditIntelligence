"""
SVB Retrospective Demo — End-to-end run

This script demonstrates that the Early Warning Financial Intelligence System
would have detected warning signs in Silicon Valley Bank's 2022 public data,
months before the bank's collapse in March 2023.

Usage:
    python demo/run_svb_demo.py

Steps:
    1. Ingest SVB SEC filings (2022-01-01 to 2023-03-10)
    2. Load mock news articles (NewsAPI historical data unavailable on free tier)
    3. Inject known HTM unrealized loss metrics into market data
    4. Run the rule engine — should fire on deposit decline and HTM ratio
    5. Run RAG retrieval against the indexed filings
    6. Call Claude for structured risk assessment
    7. Generate plain-English briefing
    8. Print the full output

Expected result at Q3 2022 signals:
    risk_level: ELEVATED or HIGH
    key signals: HTM loss ratio, deposit QoQ decline
    confidence_score: > 0.70
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("svb_demo")


def run_demo() -> None:
    from demo.svb_config import (
        SVB_HTM_METRICS_BY_QUARTER,
        SVB_WATCHLIST,
        SVB_SEC_START,
        SVB_SEC_END,
    )
    from ingestion.chunker import chunk_documents
    from ingestion.embedder import embed_and_store
    from ingestion.market_fetcher import MarketFetcher
    from ingestion.news_fetcher import NewsFetcher
    from ingestion.sec_fetcher import SECFetcher
    from output.audit_log import AuditLog
    from output.briefing import generate_and_attach_briefing
    from risk.agent import run_risk_detection
    from risk.claude_risk import assess_company_risk
    from risk.models import RiskLevel
    from risk.retriever import FinancialRiskRetriever
    from risk.rule_engine import RuleEngine

    company = SVB_WATCHLIST[0]
    start_date = date.fromisoformat(SVB_SEC_START)
    end_date = date.fromisoformat(SVB_SEC_END)

    print("\n" + "=" * 70)
    print(" EARLY WARNING FINANCIAL INTELLIGENCE SYSTEM")
    print(" SVB Retrospective Demo — Silicon Valley Bank 2022")
    print("=" * 70 + "\n")

    # ── Step 1: Ingest SEC filings ────────────────────────────────────────────
    print("Step 1: Ingesting SEC EDGAR filings for SIVB...")
    sec = SECFetcher()
    try:
        filing_docs = sec.fetch(company, start_date, end_date)
        print(f"  → Downloaded {len(filing_docs)} filing documents")
        if filing_docs:
            chunks = chunk_documents(filing_docs)
            inserted, skipped = embed_and_store(chunks, "filings")
            print(f"  → Embedded {inserted} new chunks ({skipped} duplicates skipped)")
        else:
            print("  → No filings downloaded (check internet connection / SEC rate limits)")
    except Exception as e:
        print(f"  ⚠ SEC fetch error: {e} — continuing with mock data only")

    # ── Step 2: Load mock news ────────────────────────────────────────────────
    print("\nStep 2: Loading mock SVB news articles...")
    news = NewsFetcher(use_mock=True)
    news_docs = news.fetch(company, start_date, end_date)
    print(f"  → Loaded {len(news_docs)} news articles")
    if news_docs:
        chunks = chunk_documents(news_docs)
        inserted, skipped = embed_and_store(chunks, "news")
        print(f"  → Embedded {inserted} new chunks ({skipped} duplicates skipped)")

    # ── Step 3: Build Q3 2022 market metrics with known HTM losses ────────────
    print("\nStep 3: Building Q3 2022 market metrics...")
    market = MarketFetcher()
    try:
        market_docs = market.fetch(company, start_date, end_date)
        market_metrics: dict = {}
        for doc in market_docs:
            if doc.metadata.get("source_type") == "market_metrics":
                market_metrics = json.loads(doc.page_content)
                break
    except Exception as e:
        print(f"  ⚠ yfinance error: {e} — using synthetic metrics")
        market_metrics = {}

    # Inject known Q3 2022 SVB HTM metrics (from public 10-Q filing)
    q3_metrics = SVB_HTM_METRICS_BY_QUARTER["Q3 2022"]
    market_metrics.update({
        "ticker": "SIVB",
        "htm_unrealized_loss": q3_metrics["htm_unrealized_loss"],
        "total_equity_latest": q3_metrics["total_equity"],
        "deposit_qoq_change": -0.088,   # -8.8% from 10-Q Note 10
        "price_to_book": 0.85,           # approximate late-2022 value
        "price_return_30d": -0.22,       # approximate Nov 2022
        "data_as_of": "2022-11-04",
    })
    print(f"  → Market metrics: HTM loss = ${q3_metrics['htm_unrealized_loss']/1e9:.1f}B, "
          f"Equity = ${q3_metrics['total_equity']/1e9:.1f}B, "
          f"HTM/Equity = {q3_metrics['htm_unrealized_loss']/q3_metrics['total_equity']*100:.0f}%")

    # ── Step 4: Run the rule engine ───────────────────────────────────────────
    print("\nStep 4: Running rule engine against Q3 2022 metrics...")
    rule_engine = RuleEngine()
    rule_signals, rule_floor = rule_engine.check(market_metrics)
    rule_flags = rule_engine.flag_names(rule_signals)
    print(f"  → {len(rule_signals)} rule signal(s) detected — floor risk: {rule_floor.value}")
    for sig in rule_signals:
        print(f"     • [{sig.severity.value}] {sig.description[:80]}")

    # ── Step 5: RAG retrieval ─────────────────────────────────────────────────
    print("\nStep 5: Running RAG retrieval from ChromaDB...")
    retriever = FinancialRiskRetriever(ticker="SIVB", collection="both")
    dimension_docs = retriever.query_all_dimensions()
    total_chunks = sum(len(docs) for docs in dimension_docs.values())
    print(f"  → Retrieved {total_chunks} chunks across {len(dimension_docs)} risk dimensions")

    # ── Step 6: Claude risk assessment ───────────────────────────────────────
    print("\nStep 6: Calling Claude for structured risk assessment...")
    try:
        risk_output = assess_company_risk(
            company=company,
            rule_signals=rule_signals,
            rule_floor=rule_floor,
            rule_flags=rule_flags,
            dimension_docs=dimension_docs,
            market_metrics=market_metrics,
            assessment_date=datetime(2022, 11, 5).isoformat(),
        )
        print(f"  → Risk level: {risk_output.risk_level.value}")
        print(f"  → Confidence: {risk_output.confidence_score:.0%}")
        print(f"  → Total signals: {len(risk_output.signals)}")
    except Exception as e:
        print(f"  ⚠ Claude API error: {e}")
        print("  → Falling back to rule engine output only")
        from risk.models import RiskOutput
        risk_output = RiskOutput(
            company=company.company_name,
            ticker=company.ticker,
            assessment_date=datetime(2022, 11, 5).isoformat(),
            risk_level=rule_floor,
            signals=rule_signals,
            confidence_score=0.85,
            rule_engine_flags=rule_flags,
            industry=company.industry,
            sector=company.sector,
            market_cap_bucket=company.market_cap_bucket,
        )

    # ── Step 7: Generate briefing ─────────────────────────────────────────────
    print("\nStep 7: Generating plain-English briefing...")
    try:
        risk_output = generate_and_attach_briefing(risk_output)
        print("  → Briefing generated successfully")
    except Exception as e:
        print(f"  ⚠ Briefing generation error: {e}")

    # ── Step 8: Write to audit log ────────────────────────────────────────────
    print("\nStep 8: Writing to audit log...")
    log = AuditLog()
    rowid = log.write(risk_output)
    print(f"  → Audit record #{rowid} written")

    # ── Final output ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" RISK ASSESSMENT OUTPUT")
    print("=" * 70)

    print(f"\nCompany:         {risk_output.company} ({risk_output.ticker})")
    print(f"Assessment Date: {risk_output.assessment_date[:10]}")
    print(f"Risk Level:      {risk_output.risk_level.value}")
    print(f"Confidence:      {risk_output.confidence_score:.0%}")
    print(f"Rule Flags:      {', '.join(risk_output.rule_engine_flags) or 'None'}")

    print("\n── Signals ──────────────────────────────────────────────────────────")
    for sig in sorted(risk_output.signals, key=lambda s: s.severity.value):
        bullet = "🔴" if sig.severity.value == "HIGH" else ("🟡" if sig.severity.value == "ELEVATED" else "🟢")
        rule_tag = " [RULE]" if sig.triggered_by_rule else " [LLM]"
        print(f"\n{bullet} {sig.signal_type.value}{rule_tag}")
        print(f"   {sig.description}")

    if risk_output.narrative_summary:
        print("\n── Early Warning Briefing ───────────────────────────────────────────")
        print(f"\n{risk_output.narrative_summary}")

    print("\n── Structured JSON Output ───────────────────────────────────────────")
    print(json.dumps(risk_output.model_dump_audit(), indent=2))

    print("\n" + "=" * 70)
    print(" DEMO COMPLETE")
    print("=" * 70)
    print(f"\nResult: {risk_output.risk_level.value} risk detected at Q3 2022 assessment date.")
    if risk_output.risk_level.value in ("ELEVATED", "HIGH"):
        print("✓ System would have flagged SVB BEFORE its March 2023 collapse.")
    else:
        print("✗ System did not flag SVB at the expected ELEVATED/HIGH level.")
    print()


if __name__ == "__main__":
    run_demo()
