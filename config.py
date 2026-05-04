"""
Central configuration for the Early Warning Financial Intelligence System.

All modules import from here — never hardcode paths, collection names,
thresholds, or model names in individual modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
FILINGS_DIR = Path(os.getenv("FILINGS_DIR", DATA_DIR / "filings"))
AUDIT_DB_PATH = Path(os.getenv("AUDIT_DB_PATH", DATA_DIR / "audit.db"))

# Ensure runtime directories exist
for _dir in (DATA_DIR, CHROMA_DIR, FILINGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
NEWSAPI_KEY: str = os.getenv("NEWSAPI_KEY", "")
SEC_USER_AGENT_ORG: str = os.getenv("SEC_USER_AGENT_ORG", "EarlyWarningSystem")
SEC_USER_AGENT_EMAIL: str = os.getenv("SEC_USER_AGENT_EMAIL", "admin@example.com")

# ── ChromaDB Collections ───────────────────────────────────────────────────────

CHROMA_COLLECTION_FILINGS = "filings"
CHROMA_COLLECTION_NEWS = "news"

# ── Embedding Model ────────────────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 22MB, fast, good for financial text

# ── Chunking ───────────────────────────────────────────────────────────────────

CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 150      # overlap between consecutive chunks
RETRIEVAL_TOP_K = 8      # number of chunks returned per RAG query

# ── Claude Models ──────────────────────────────────────────────────────────────

CLAUDE_RISK_MODEL = "claude-sonnet-4-5"     # structured risk assessment (tool use)
CLAUDE_BRIEFING_MODEL = "claude-haiku-4-5-20251001"  # narrative generation (cheaper/faster)

# ── Risk Thresholds (Rule Engine) ──────────────────────────────────────────────

@dataclass
class RuleThresholds:
    # Deposit growth: flag if QoQ change falls below this fraction (e.g. -0.05 = -5%)
    deposit_qoq_decline: float = -0.05

    # HTM unrealized loss as a fraction of total equity: flag if ratio exceeds this
    htm_loss_equity_ratio: float = 0.80

    # Revenue QoQ decline threshold
    revenue_qoq_decline: float = -0.10

    # Net interest margin decline (percentage points) in a single quarter
    nim_decline_pp: float = 0.20

    # Stock price 30-day return: flag if worse than this
    price_return_30d: float = -0.20

    # Price-to-book ratio: flag if below this
    price_to_book_min: float = 0.70

    # Equity tier-1 capital ratio: flag if below regulatory minimum threshold
    tier1_capital_ratio_min: float = 0.08


THRESHOLDS = RuleThresholds()

# ── Disparity Monitoring ───────────────────────────────────────────────────────

DISPARITY_RATIO_THRESHOLD = 1.25  # Four-Fifths Rule: flag if group A/B > this value

# ── Scheduler ─────────────────────────────────────────────────────────────────

PIPELINE_SCHEDULE_HOURS = 6  # run ingestion pipeline every N hours

# ── Watchlist ──────────────────────────────────────────────────────────────────

@dataclass
class WatchlistEntry:
    ticker: str
    company_name: str
    cik: str                  # SEC EDGAR Central Index Key
    industry: str
    sector: str
    market_cap_bucket: str    # "Small", "Mid", or "Large"
    news_queries: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        valid_buckets = {"Small", "Mid", "Large"}
        if self.market_cap_bucket not in valid_buckets:
            raise ValueError(f"market_cap_bucket must be one of {valid_buckets}")


# Default watchlist — focused on regional/community banks for the demo
DEFAULT_WATCHLIST: list[WatchlistEntry] = [
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
        ],
    ),
]

# ── Ingestion Date Range Defaults ──────────────────────────────────────────────

# For the SVB retrospective demo, override these in demo/svb_config.py
DEFAULT_LOOKBACK_DAYS = 90
FILINGS_LOOKBACK_YEARS = 3
