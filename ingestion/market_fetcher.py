"""
Market data fetcher using yfinance.

Pulls price history and quarterly financials (balance sheet, income
statement, cash flow) for each watchlist company. yfinance requires
no API key and is free.

Unlike SEC and news data, market metrics are NOT stored in ChromaDB.
They are returned as structured Documents with JSON content so the
pipeline can pass them directly to the rule engine and as structured
context to Claude.

Implements DataSourceProtocol.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd
import yfinance as yf
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from config import WatchlistEntry

logger = logging.getLogger(__name__)


def _safe_float(value: object) -> float | None:
    """Convert a potentially NaN or None value to float or None."""
    try:
        f = float(value)  # type: ignore[arg-type]
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _qoq_change(series: pd.Series) -> float | None:
    """
    Calculate the most recent quarter-over-quarter change as a fraction.

    Args:
        series: pandas Series indexed by date, most recent first.

    Returns:
        QoQ change as a decimal (e.g. -0.088 for -8.8%), or None if not enough data.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return None
    latest = float(clean.iloc[0])
    prior = float(clean.iloc[1])
    if prior == 0:
        return None
    return (latest - prior) / abs(prior)


class MarketFetcher:
    """
    Pulls price history and quarterly financial metrics from yfinance.

    Returns two Documents per company:
      1. "market_metrics" — structured JSON of key ratios for the rule engine
      2. "price_history"  — last 90 days of daily price data as a text table
    """

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch(
        self,
        company: WatchlistEntry,
        start_date: date,
        end_date: date,
    ) -> list[Document]:
        ticker_obj = yf.Ticker(company.ticker)
        documents: list[Document] = []

        # ── Structured financial metrics ──────────────────────────────────────
        metrics = self._extract_metrics(ticker_obj, company.ticker)
        if metrics:
            docs_str = json.dumps(metrics, indent=2, default=str)
            documents.append(
                Document(
                    page_content=docs_str,
                    metadata={
                        "company": company.company_name,
                        "ticker": company.ticker,
                        "source_type": "market_metrics",
                        "doc_id": f"{company.ticker}_market_metrics_{end_date.strftime('%Y%m%d')}",
                        "data_as_of": end_date.isoformat(),
                    },
                )
            )

        # ── Price history text table ──────────────────────────────────────────
        try:
            hist = ticker_obj.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
            )
            if not hist.empty:
                price_text = self._price_history_to_text(hist, company.ticker)
                documents.append(
                    Document(
                        page_content=price_text,
                        metadata={
                            "company": company.company_name,
                            "ticker": company.ticker,
                            "source_type": "price_history",
                            "doc_id": f"{company.ticker}_price_{start_date}_{end_date}",
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                        },
                    )
                )
        except Exception as exc:
            logger.warning("Could not fetch price history for %s: %s", company.ticker, exc)

        logger.info(
            "Market fetcher: %d documents for %s [%s → %s]",
            len(documents),
            company.ticker,
            start_date,
            end_date,
        )
        return documents

    def _extract_metrics(self, ticker_obj: yf.Ticker, ticker: str) -> dict:
        """Extract key financial metrics as a flat dict for the rule engine."""
        metrics: dict = {"ticker": ticker}

        try:
            info = ticker_obj.info or {}
            metrics["price_to_book"] = _safe_float(info.get("priceToBook"))
            metrics["market_cap"] = _safe_float(info.get("marketCap"))
            metrics["beta"] = _safe_float(info.get("beta"))
            metrics["short_ratio"] = _safe_float(info.get("shortRatio"))
            metrics["current_ratio"] = _safe_float(info.get("currentRatio"))
        except Exception as exc:
            logger.warning("Could not fetch info for %s: %s", ticker, exc)

        # Quarterly balance sheet
        try:
            bs = ticker_obj.quarterly_balance_sheet
            if bs is not None and not bs.empty:
                total_equity_row = next(
                    (r for r in bs.index if "stockholder" in r.lower() or "equity" in r.lower()),
                    None,
                )
                total_deposits_row = next(
                    (r for r in bs.index if "deposit" in r.lower()), None
                )
                if total_equity_row is not None:
                    metrics["total_equity_series"] = bs.loc[total_equity_row].to_dict()
                    metrics["total_equity_latest"] = _safe_float(
                        bs.loc[total_equity_row].iloc[0]
                    )
                if total_deposits_row is not None:
                    deposit_series = bs.loc[total_deposits_row]
                    metrics["total_deposits_latest"] = _safe_float(deposit_series.iloc[0])
                    metrics["deposit_qoq_change"] = _qoq_change(deposit_series)
        except Exception as exc:
            logger.warning("Could not fetch balance sheet for %s: %s", ticker, exc)

        # Quarterly income statement
        try:
            inc = ticker_obj.quarterly_income_stmt
            if inc is not None and not inc.empty:
                revenue_row = next(
                    (r for r in inc.index if "revenue" in r.lower() or "total revenue" in r.lower()),
                    None,
                )
                if revenue_row is not None:
                    rev_series = inc.loc[revenue_row]
                    metrics["revenue_latest"] = _safe_float(rev_series.iloc[0])
                    metrics["revenue_qoq_change"] = _qoq_change(rev_series)
        except Exception as exc:
            logger.warning("Could not fetch income statement for %s: %s", ticker, exc)

        # 30-day price return
        try:
            hist_30 = ticker_obj.history(period="35d", interval="1d")
            if len(hist_30) >= 2:
                price_latest = float(hist_30["Close"].iloc[-1])
                price_30d_ago = float(hist_30["Close"].iloc[0])
                if price_30d_ago > 0:
                    metrics["price_return_30d"] = (price_latest - price_30d_ago) / price_30d_ago
        except Exception as exc:
            logger.warning("Could not compute 30-day return for %s: %s", ticker, exc)

        return metrics

    def _price_history_to_text(self, hist: pd.DataFrame, ticker: str) -> str:
        """Convert a yfinance price history DataFrame to a readable text summary."""
        lines = [f"Price history for {ticker}:", "Date | Open | High | Low | Close | Volume"]
        for dt, row in hist.tail(90).iterrows():
            lines.append(
                f"{str(dt)[:10]} | {row['Open']:.2f} | {row['High']:.2f} | "
                f"{row['Low']:.2f} | {row['Close']:.2f} | {int(row['Volume']):,}"
            )
        return "\n".join(lines)
