"""
SEC EDGAR data fetcher.

Downloads 10-K, 10-Q, and 8-K filings for a company using the
sec-edgar-downloader library, then extracts text from the HTML/XML files
using BeautifulSoup. Implements DataSourceProtocol.

Rate limiting: SEC EDGAR asks for ≤10 requests/second. sec-edgar-downloader
handles this automatically.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from sec_edgar_downloader import Downloader
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    FILINGS_DIR,
    SEC_USER_AGENT_EMAIL,
    SEC_USER_AGENT_ORG,
    WatchlistEntry,
)

logger = logging.getLogger(__name__)

# Filing types to retrieve — ordered by information density
FILING_TYPES = ["10-K", "10-Q", "8-K"]

# High-signal section headers in SEC filings (used to tag chunk metadata)
SECTION_PATTERNS: list[tuple[str, str]] = [
    (r"item\s+1a", "Item 1A - Risk Factors"),
    (r"item\s+7a", "Item 7A - Market Risk"),
    (r"item\s+7[^a]", "Item 7 - MD&A"),
    (r"note\s+[45][^\d]", "Note 4/5 - Investment Securities"),
    (r"note\s+10[^\d]", "Note 10 - Deposits"),
    (r"management.{0,20}discussion", "MD&A"),
    (r"liquidity", "Liquidity Section"),
    (r"capital\s+resources", "Capital Resources"),
]


def _detect_section(text_lower: str) -> str:
    """Heuristically identify the section from a block of text."""
    for pattern, label in SECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return label
    return "General"


def _extract_text_from_file(filepath: Path) -> str:
    """Strip HTML/XML tags and return clean text from a filing file."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(content, "html.parser")
    # Remove script/style noise
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_doc_id(ticker: str, filing_type: str, filing_date: str) -> str:
    payload = f"{ticker}_{filing_type}_{filing_date}"
    return payload + "_" + hashlib.md5(payload.encode()).hexdigest()[:8]


class SECFetcher:
    """
    Fetches SEC 10-K, 10-Q, and 8-K filings for a single company.

    Returns LangChain Documents with metadata fields:
        company, ticker, filing_type, filing_date, section, doc_id, chunk_index
    """

    def __init__(self) -> None:
        self._downloader = Downloader(
            company_name=SEC_USER_AGENT_ORG,
            email_address=SEC_USER_AGENT_EMAIL,
            save_path=str(FILINGS_DIR),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def fetch(
        self,
        company: WatchlistEntry,
        start_date: date,
        end_date: date,
    ) -> list[Document]:
        """
        Download filings for *company* filed between start_date and end_date,
        extract their text, and return as Documents.
        """
        documents: list[Document] = []

        for filing_type in FILING_TYPES:
            try:
                self._downloader.get(
                    filing_type,
                    company.ticker,
                    after=start_date.strftime("%Y-%m-%d"),
                    before=end_date.strftime("%Y-%m-%d"),
                    download_details=True,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to download %s for %s: %s",
                    filing_type,
                    company.ticker,
                    exc,
                )
                continue

            # Walk the downloaded directory tree for this ticker + filing type
            ticker_dir = FILINGS_DIR / "sec-edgar-filings" / company.ticker / filing_type
            if not ticker_dir.exists():
                continue

            for filing_dir in sorted(ticker_dir.iterdir()):
                if not filing_dir.is_dir():
                    continue

                # Prefer the primary document (.htm / .html) over .xml
                candidates = (
                    list(filing_dir.glob("*.htm"))
                    + list(filing_dir.glob("*.html"))
                    + list(filing_dir.glob("*.txt"))
                )
                if not candidates:
                    continue

                primary_file = candidates[0]
                try:
                    text = _extract_text_from_file(primary_file)
                except Exception as exc:
                    logger.warning("Could not extract text from %s: %s", primary_file, exc)
                    continue

                if len(text) < 200:  # skip near-empty files
                    continue

                # Use folder name (accession number) to derive a stable date
                # The folder name format is typically: 0000000000-YY-NNNNNN
                filing_date_str = start_date.strftime("%Y-%m-%d")  # fallback
                folder_name = filing_dir.name.replace("-", "")
                doc_id = _make_doc_id(company.ticker, filing_type, folder_name)

                section = _detect_section(text[:2000].lower())

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "company": company.company_name,
                            "ticker": company.ticker,
                            "filing_type": filing_type,
                            "filing_date": filing_date_str,
                            "section": section,
                            "source_type": "filings",
                            "doc_id": doc_id,
                        },
                    )
                )
                logger.info(
                    "Loaded %s %s filing (%d chars) from %s",
                    company.ticker,
                    filing_type,
                    len(text),
                    primary_file.name,
                )

        logger.info(
            "SEC fetcher: %d documents for %s [%s → %s]",
            len(documents),
            company.ticker,
            start_date,
            end_date,
        )
        return documents
