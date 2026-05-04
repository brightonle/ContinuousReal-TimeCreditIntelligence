"""
Singleton ChromaDB client factory and collection management.

Two collections are maintained:
  - "filings"  — chunks from SEC 10-K / 10-Q / 8-K documents
  - "news"     — chunks from news articles

Both collections store metadata conforming to a defined schema so that
retriever.py can filter by ticker and source_type reliably.
"""

from __future__ import annotations

import threading
from typing import Optional

import chromadb
from chromadb import Collection
from chromadb.config import Settings

from config import CHROMA_COLLECTION_FILINGS, CHROMA_COLLECTION_NEWS, CHROMA_DIR

# ── Metadata schemas (documentation only — ChromaDB does not enforce these) ────
#
# filings collection metadata fields:
#   company       str   full company name
#   ticker        str   stock ticker symbol
#   filing_type   str   "10-K", "10-Q", "8-K", etc.
#   filing_date   str   ISO-8601 date string (YYYY-MM-DD)
#   section       str   filing section label (e.g. "Item 7A", "Note 4")
#   doc_id        str   {ticker}_{filing_type}_{date}_{hash8}
#   chunk_index   int   position of this chunk within the source document
#
# news collection metadata fields:
#   company        str   full company name
#   ticker         str   stock ticker symbol
#   source         str   news source name (e.g. "Reuters", "NewsAPI")
#   published_date str   ISO-8601 date string
#   headline       str   article headline (truncated to 200 chars)
#   doc_id         str   {ticker}_news_{date}_{hash8}
#   chunk_index    int   position of this chunk within the article


_lock = threading.Lock()
_client: Optional[chromadb.PersistentClient] = None


def get_client() -> chromadb.PersistentClient:
    """Return the singleton ChromaDB persistent client."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = chromadb.PersistentClient(
                    path=str(CHROMA_DIR),
                    settings=Settings(anonymized_telemetry=False),
                )
    return _client


def get_collection(name: str) -> Collection:
    """
    Return the named collection, creating it if it does not yet exist.

    Args:
        name: one of CHROMA_COLLECTION_FILINGS or CHROMA_COLLECTION_NEWS
    """
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for semantic search
    )


def get_filings_collection() -> Collection:
    return get_collection(CHROMA_COLLECTION_FILINGS)


def get_news_collection() -> Collection:
    return get_collection(CHROMA_COLLECTION_NEWS)


def collection_doc_ids(collection_name: str) -> set[str]:
    """Return the set of all doc_ids currently stored in a collection."""
    collection = get_collection(collection_name)
    results = collection.get(include=["metadatas"])
    return {
        meta["doc_id"]
        for meta in (results["metadatas"] or [])
        if meta and "doc_id" in meta
    }


def reset_collection(name: str) -> None:
    """Delete and recreate a collection. Use only in tests."""
    client = get_client()
    client.delete_collection(name)
    get_collection(name)
