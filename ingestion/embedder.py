"""
Embedding and ChromaDB upsert with doc_id deduplication.

Uses sentence-transformers locally (no API cost) to produce embeddings,
then upserts chunks into the appropriate ChromaDB collection.

Deduplication contract:
  - Every chunk carries a ``chunk_id`` metadata field of the form
    ``{doc_id}_c{index}``
  - Before inserting, we check for existing chunk_ids to avoid duplicates
    across pipeline runs.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from config import CHROMA_COLLECTION_FILINGS, CHROMA_COLLECTION_NEWS, EMBEDDING_MODEL_NAME
from ingestion.chroma_client import collection_doc_ids, get_collection

logger = logging.getLogger(__name__)

# Module-level singleton — model is loaded once and reused across all calls
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


CollectionName = Literal["filings", "news"]


def upsert_chunks(
    chunks: list[Document],
    collection_name: CollectionName,
) -> tuple[int, int]:
    """
    Embed chunks and upsert them into ChromaDB.

    Skips chunks whose ``chunk_id`` is already present in the collection
    to make pipeline runs idempotent.

    Args:
        chunks:          chunked Documents with metadata (must include chunk_id)
        collection_name: "filings" or "news"

    Returns:
        Tuple of (inserted_count, skipped_count).
    """
    if not chunks:
        return 0, 0

    collection = get_collection(collection_name)

    # Fetch all existing chunk_ids from ChromaDB in one call
    existing = collection_doc_ids(collection_name)

    new_chunks = [c for c in chunks if c.metadata.get("chunk_id") not in existing]
    skipped = len(chunks) - len(new_chunks)

    if not new_chunks:
        logger.info(
            "All %d chunks already in '%s' — skipping upsert.",
            len(chunks),
            collection_name,
        )
        return 0, skipped

    model = _get_model()
    texts = [c.page_content for c in new_chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    ids = [c.metadata["chunk_id"] for c in new_chunks]
    metadatas = [c.metadata for c in new_chunks]

    # ChromaDB upsert is idempotent for existing IDs but we've already filtered
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info(
        "Upserted %d chunks into '%s' (skipped %d duplicates).",
        len(new_chunks),
        collection_name,
        skipped,
    )
    return len(new_chunks), skipped


def embed_and_store(
    chunks: list[Document],
    source_type: Literal["filings", "news"],
) -> tuple[int, int]:
    """
    Convenience wrapper: determines collection from source_type and upserts.
    """
    collection_map = {
        "filings": CHROMA_COLLECTION_FILINGS,
        "news": CHROMA_COLLECTION_NEWS,
    }
    return upsert_chunks(chunks, collection_map[source_type])
