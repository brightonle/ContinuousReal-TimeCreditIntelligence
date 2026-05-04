"""
Ingestion package — data sources, chunking, embedding, and vector store.

Defines the DataSourceProtocol that all fetchers must implement so the
pipeline can treat SEC, news, and market fetchers interchangeably.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document

from config import WatchlistEntry


@runtime_checkable
class DataSourceProtocol(Protocol):
    """
    Interface every data fetcher must implement.

    Returns a list of LangChain Document objects where:
      - doc.page_content  — the raw text to be chunked and embedded
      - doc.metadata      — must include at minimum:
            company (str), ticker (str), source_type (str), doc_id (str)
    """

    def fetch(
        self,
        company: WatchlistEntry,
        start_date: date,
        end_date: date,
    ) -> list[Document]:
        ...
