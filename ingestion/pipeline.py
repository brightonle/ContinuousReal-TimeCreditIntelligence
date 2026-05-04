"""
Ingestion pipeline orchestrator.

Runs all data fetchers concurrently (using asyncio.gather), then passes
the resulting Documents through the chunker and into ChromaDB via the
embedder.

Market metric Documents (source_type="market_metrics" / "price_history")
are NOT embedded into ChromaDB — they are returned separately so the
risk agent can pass them directly as structured context to Claude and
the rule engine.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta

from langchain_core.documents import Document

from config import DEFAULT_LOOKBACK_DAYS, DEFAULT_WATCHLIST, WatchlistEntry
from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_and_store
from ingestion.market_fetcher import MarketFetcher
from ingestion.news_fetcher import NewsFetcher
from ingestion.sec_fetcher import SECFetcher

logger = logging.getLogger(__name__)


def run_pipeline_for_company(
    company: WatchlistEntry,
    start_date: date,
    end_date: date,
    use_mock_news: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline for a single company synchronously.

    Returns:
        dict with keys: ticker, filings_inserted, news_inserted,
                        market_docs (raw Documents for rule engine), errors
    """
    sec = SECFetcher()
    news = NewsFetcher(use_mock=use_mock_news)
    market = MarketFetcher()

    result: dict = {
        "ticker": company.ticker,
        "filings_inserted": 0,
        "news_inserted": 0,
        "market_docs": [],
        "errors": [],
    }

    # ── SEC filings ───────────────────────────────────────────────────────────
    try:
        filing_docs = sec.fetch(company, start_date, end_date)
        if filing_docs:
            chunks = chunk_documents(filing_docs)
            inserted, _ = embed_and_store(chunks, "filings")
            result["filings_inserted"] = inserted
    except Exception as exc:
        logger.error("SEC fetch failed for %s: %s", company.ticker, exc)
        result["errors"].append(f"SEC: {exc}")

    # ── News ──────────────────────────────────────────────────────────────────
    try:
        news_docs = news.fetch(company, start_date, end_date)
        if news_docs:
            chunks = chunk_documents(news_docs)
            inserted, _ = embed_and_store(chunks, "news")
            result["news_inserted"] = inserted
    except Exception as exc:
        logger.error("News fetch failed for %s: %s", company.ticker, exc)
        result["errors"].append(f"News: {exc}")

    # ── Market metrics (not embedded — returned raw for rule engine) ──────────
    try:
        market_docs = market.fetch(company, start_date, end_date)
        result["market_docs"] = market_docs
    except Exception as exc:
        logger.error("Market fetch failed for %s: %s", company.ticker, exc)
        result["errors"].append(f"Market: {exc}")

    logger.info(
        "Pipeline complete for %s: %d filing chunks, %d news chunks inserted.",
        company.ticker,
        result["filings_inserted"],
        result["news_inserted"],
    )
    return result


async def _fetch_one_async(
    company: WatchlistEntry,
    start_date: date,
    end_date: date,
    use_mock_news: bool,
) -> dict:
    """Async wrapper so multiple companies can be processed concurrently."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        run_pipeline_for_company,
        company,
        start_date,
        end_date,
        use_mock_news,
    )


async def run_pipeline_async(
    watchlist: list[WatchlistEntry] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    use_mock_news: bool = False,
) -> list[dict]:
    """
    Run the ingestion pipeline for all companies in the watchlist concurrently.

    Args:
        watchlist:     companies to process; defaults to DEFAULT_WATCHLIST
        lookback_days: how many days back to fetch data
        use_mock_news: pass True for the SVB retrospective demo

    Returns:
        List of per-company result dicts.
    """
    companies = watchlist or DEFAULT_WATCHLIST
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    logger.info(
        "Starting ingestion pipeline for %d companies [%s → %s]",
        len(companies),
        start_date,
        end_date,
    )

    tasks = [
        _fetch_one_async(company, start_date, end_date, use_mock_news)
        for company in companies
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)


def run_pipeline(
    watchlist: list[WatchlistEntry] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    use_mock_news: bool = False,
) -> list[dict]:
    """Synchronous entry point for the scheduler and demo scripts."""
    return asyncio.run(
        run_pipeline_async(watchlist, lookback_days, use_mock_news)
    )


def get_latest_market_docs(
    company: WatchlistEntry,
    lookback_days: int = 35,
) -> list[Document]:
    """
    Fetch fresh market metrics for a single company without touching ChromaDB.
    Used by the risk agent just before running the rule engine.
    """
    market = MarketFetcher()
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    return market.fetch(company, start_date, end_date)
