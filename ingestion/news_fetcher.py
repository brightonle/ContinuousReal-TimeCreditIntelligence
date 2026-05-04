"""
Financial news fetcher.

Primary source: NewsAPI (requires NEWSAPI_KEY).
Fallback: feedparser RSS (Reuters, Yahoo Finance, Google News RSS — all free).

Note: NewsAPI free tier only returns articles from the past 30 days.
For the SVB retrospective demo, set USE_MOCK_NEWS=True and point to
demo/mock_news_svb.json to bypass this limitation.

Implements DataSourceProtocol.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import feedparser
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from config import NEWSAPI_KEY, ROOT_DIR, WatchlistEntry

logger = logging.getLogger(__name__)

# RSS feeds used as a fallback (all public, no API key needed)
RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "https://news.google.com/rss/search?q={ticker}+stock+financial&hl=en-US&gl=US&ceid=US:en",
]

MOCK_NEWS_PATH = ROOT_DIR / "demo" / "mock_news_svb.json"


def _make_news_doc_id(ticker: str, published: str, headline: str) -> str:
    payload = f"{ticker}_news_{published}_{headline[:40]}"
    return f"{ticker}_news_{published[:10]}_{hashlib.md5(payload.encode()).hexdigest()[:8]}"


def _article_to_document(
    company: WatchlistEntry,
    headline: str,
    content: str,
    published_date: str,
    source_name: str,
) -> Document:
    doc_id = _make_news_doc_id(company.ticker, published_date, headline)
    return Document(
        page_content=f"{headline}\n\n{content}".strip(),
        metadata={
            "company": company.company_name,
            "ticker": company.ticker,
            "source": source_name,
            "published_date": published_date,
            "headline": headline[:200],
            "source_type": "news",
            "doc_id": doc_id,
        },
    )


class NewsFetcher:
    """
    Fetches financial news articles for a company.

    Tries NewsAPI first; falls back to RSS feeds; falls back to mock data
    (for the SVB retrospective demo when historical data is unavailable).
    """

    def __init__(self, use_mock: bool = False) -> None:
        self._use_mock = use_mock

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch(
        self,
        company: WatchlistEntry,
        start_date: date,
        end_date: date,
    ) -> list[Document]:
        if self._use_mock or not NEWSAPI_KEY:
            return self._fetch_mock(company)

        docs = self._fetch_newsapi(company, start_date, end_date)
        if not docs:
            logger.info("NewsAPI returned no results for %s — trying RSS.", company.ticker)
            docs = self._fetch_rss(company)
        return docs

    def _fetch_newsapi(
        self,
        company: WatchlistEntry,
        start_date: date,
        end_date: date,
    ) -> list[Document]:
        try:
            from newsapi import NewsApiClient  # type: ignore[import]
        except ImportError:
            logger.warning("newsapi-python not installed — skipping NewsAPI fetch.")
            return []

        client = NewsApiClient(api_key=NEWSAPI_KEY)
        documents: list[Document] = []

        for query in company.news_queries or [company.company_name]:
            try:
                response = client.get_everything(
                    q=query,
                    from_param=start_date.isoformat(),
                    to=end_date.isoformat(),
                    language="en",
                    sort_by="relevancy",
                    page_size=100,
                )
            except Exception as exc:
                logger.warning("NewsAPI error for query '%s': %s", query, exc)
                continue

            for article in response.get("articles", []):
                headline = article.get("title", "")
                content = article.get("content") or article.get("description") or ""
                published = (article.get("publishedAt") or "")[:10]
                source = article.get("source", {}).get("name", "NewsAPI")
                if not headline or not content:
                    continue
                documents.append(
                    _article_to_document(company, headline, content, published, source)
                )

        logger.info(
            "NewsAPI fetcher: %d articles for %s [%s → %s]",
            len(documents),
            company.ticker,
            start_date,
            end_date,
        )
        return documents

    def _fetch_rss(self, company: WatchlistEntry) -> list[Document]:
        documents: list[Document] = []
        for feed_template in RSS_FEEDS:
            url = feed_template.format(ticker=company.ticker)
            try:
                feed = feedparser.parse(url)
            except Exception as exc:
                logger.warning("RSS feed error for %s: %s", url, exc)
                continue

            for entry in feed.entries[:50]:
                headline = entry.get("title", "")
                content = entry.get("summary", "")
                published = entry.get("published", "")[:10]
                if not headline:
                    continue
                documents.append(
                    _article_to_document(company, headline, content, published, "RSS")
                )

        logger.info("RSS fetcher: %d articles for %s", len(documents), company.ticker)
        return documents

    def _fetch_mock(self, company: WatchlistEntry) -> list[Document]:
        """Load curated historical articles from mock_news_svb.json."""
        mock_path = Path(MOCK_NEWS_PATH)
        if not mock_path.exists():
            logger.warning("Mock news file not found at %s", mock_path)
            return []

        with open(mock_path, encoding="utf-8") as f:
            articles: list[dict] = json.load(f)

        # Filter to articles relevant to this ticker
        documents: list[Document] = []
        for article in articles:
            if article.get("ticker", company.ticker) != company.ticker:
                continue
            documents.append(
                _article_to_document(
                    company,
                    article.get("headline", ""),
                    article.get("content", ""),
                    article.get("published_date", ""),
                    article.get("source", "Mock"),
                )
            )

        logger.info("Mock news fetcher: %d articles for %s", len(documents), company.ticker)
        return documents
