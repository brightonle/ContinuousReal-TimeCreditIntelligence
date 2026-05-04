"""
RAG retriever for financial risk signals.

Wraps ChromaDB as a LangChain BaseRetriever. Supports filtering by:
  - ticker symbol (always applied)
  - collection ("filings", "news", or "both")
  - date range

The agent calls this retriever with targeted queries per risk dimension
(e.g., "liquidity risk indicators", "credit rating changes") rather than
one generic query, to maximise retrieval precision.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, RETRIEVAL_TOP_K
from ingestion.chroma_client import get_filings_collection, get_news_collection

logger = logging.getLogger(__name__)

CollectionTarget = Literal["filings", "news", "both"]

# Targeted queries used by the agent for each risk dimension
RISK_DIMENSION_QUERIES: dict[str, list[str]] = {
    "liquidity": [
        "liquidity risk cash reserves deposit outflows funding",
        "held-to-maturity securities unrealized losses interest rate",
    ],
    "credit_rating": [
        "credit rating downgrade Moody's S&P Fitch outlook negative",
    ],
    "earnings": [
        "revenue decline net interest margin earnings net income quarterly results",
    ],
    "capital": [
        "capital ratios tier 1 regulatory capital adequacy",
        "equity total stockholders equity balance sheet",
    ],
    "management": [
        "management changes CEO CFO leadership departure resignation",
        "going concern audit opinion material weakness internal controls",
    ],
    "macro": [
        "interest rate risk federal reserve monetary policy",
        "venture capital startup tech sector deposit concentration",
    ],
}


class FinancialRiskRetriever(BaseRetriever):
    """
    Retrieves the most relevant document chunks for a company and query.

    Supports querying both the filings and news collections simultaneously
    and deduplicating results by chunk_id.
    """

    ticker: str
    collection: CollectionTarget = "both"
    top_k: int = RETRIEVAL_TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        return self.query(query)

    def query(
        self,
        query: str,
        collection: Optional[CollectionTarget] = None,
        top_k: Optional[int] = None,
    ) -> list[Document]:
        """
        Query ChromaDB for chunks relevant to *query* filtered by ticker.

        Args:
            query:      free-text query string
            collection: override which collection(s) to search
            top_k:      number of results to return

        Returns:
            Deduplicated list of Document objects ranked by relevance.
        """
        target = collection or self.collection
        k = top_k or self.top_k

        model = _get_embedding_model()
        embedding = model.encode([query], show_progress_bar=False)[0].tolist()

        results: list[Document] = []
        seen_ids: set[str] = set()

        collections_to_query = []
        if target in ("filings", "both"):
            collections_to_query.append(("filings", get_filings_collection()))
        if target in ("news", "both"):
            collections_to_query.append(("news", get_news_collection()))

        for coll_name, coll in collections_to_query:
            try:
                response = coll.query(
                    query_embeddings=[embedding],
                    n_results=min(k, coll.count()),
                    where={"ticker": self.ticker},
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as exc:
                logger.warning("ChromaDB query failed on '%s': %s", coll_name, exc)
                continue

            docs = response.get("documents", [[]])[0]
            metas = response.get("metadatas", [[]])[0]

            for text, meta in zip(docs, metas):
                chunk_id = (meta or {}).get("chunk_id", "")
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                results.append(Document(page_content=text, metadata=meta or {}))

        logger.debug(
            "Retriever returned %d chunks for ticker=%s query='%s'",
            len(results),
            self.ticker,
            query[:60],
        )
        return results[:k]

    def query_all_dimensions(self) -> dict[str, list[Document]]:
        """
        Run all risk-dimension queries and return results keyed by dimension name.
        Used by the agent to build a comprehensive context before calling Claude.
        """
        all_results: dict[str, list[Document]] = {}
        for dimension, queries in RISK_DIMENSION_QUERIES.items():
            dimension_docs: list[Document] = []
            seen: set[str] = set()
            for q in queries:
                for doc in self.query(q, top_k=4):
                    cid = doc.metadata.get("chunk_id", "")
                    if cid not in seen:
                        seen.add(cid)
                        dimension_docs.append(doc)
            all_results[dimension] = dimension_docs
        return all_results


# Module-level singleton to avoid reloading the model on every retrieval
_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model
