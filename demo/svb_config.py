"""
SVB retrospective demo configuration.

Overrides config.py defaults to focus exclusively on Silicon Valley Bank (SIVB)
using 2022 public filings and news coverage.

The demo fetches SEC filings filed between 2022-01-01 and 2023-03-10,
news articles from the same period (falling back to mock data if NewsAPI
historical access is unavailable), and yfinance market data.

Expected outcome: the system detects ELEVATED/HIGH risk signals by Q3 2022
(filing date: Nov 4 2022) based on:
  - HTM unrealized losses ~97% of total equity
  - Deposit QoQ decline of -8.8%
  - Price-to-book falling below 1.0
"""

from __future__ import annotations

from config import WatchlistEntry

# SVB-specific watchlist (single company for the demo)
SVB_WATCHLIST = [
    WatchlistEntry(
        ticker="SIVB",
        company_name="SVB Financial Group",
        cik="0000719739",
        industry="Banks",
        sector="Financials",
        market_cap_bucket="Large",
        news_queries=[
            '"SVB Financial" OR "Silicon Valley Bank" AND (liquidity OR deposits OR "interest rate")',
            '"SIVB" AND (downgrade OR "credit rating" OR risk)',
            '"venture capital" AND (deposits OR "bank run" OR withdrawal)',
            '"SVB" AND ("held-to-maturity" OR "HTM" OR "unrealized loss")',
        ],
    )
]

# Date ranges for the SVB retrospective analysis
SVB_SEC_START = "2022-01-01"
SVB_SEC_END = "2023-03-10"

# News phases (for reference — the pipeline fetches the full range at once)
SVB_NEWS_PHASES = [
    ("2022-01-01", "2022-06-30", "Baseline — no stress signals expected"),
    ("2022-07-01", "2022-12-31", "Rate hike anxiety — HTM losses growing"),
    ("2023-01-01", "2023-03-07", "VC/tech sector stress — deposit flight signals"),
    ("2023-03-08", "2023-03-15", "Collapse event"),
]

# HTM unrealized loss values from SVB's quarterly filings (hard-coded for demo)
# These are extracted from public SEC filings and injected into the rule engine
SVB_HTM_METRICS_BY_QUARTER = {
    "Q1 2022": {"htm_unrealized_loss": 1_800_000_000, "total_equity": 16_300_000_000},
    "Q2 2022": {"htm_unrealized_loss": 9_500_000_000, "total_equity": 16_200_000_000},
    "Q3 2022": {"htm_unrealized_loss": 15_900_000_000, "total_equity": 16_300_000_000},
    "Q4 2022": {"htm_unrealized_loss": 17_100_000_000, "total_equity": 15_800_000_000},
}

# Expected signals for evaluation (ground truth)
SVB_EXPECTED_SIGNALS = [
    "HTM unrealized losses exceeding 80% of total equity (Q3 2022: ~97%)",
    "Deposit QoQ decline of -8.8% in Q3 2022",
    "Price-to-book falling below 1.0 in late 2022",
    "30-day stock return < -20% in week of March 6-10 2023",
    "Emergency capital raise announcement (8-K, March 8 2023)",
]
