"""
Document chunking for SEC filings and news articles.

Uses LangChain's RecursiveCharacterTextSplitter so chunks respect natural
text boundaries (paragraphs → sentences → words). All source metadata is
preserved and a chunk_index field is added so chunks can be reassembled
or cited by position within the original document.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of Documents into fixed-size overlapping chunks.

    Each output Document:
      - Inherits all metadata from its parent document
      - Adds a ``chunk_index`` metadata field (0-based position)
      - Preserves the parent ``doc_id`` so dedup logic can reference it

    Args:
        documents: list of LangChain Document objects with page_content and metadata

    Returns:
        Flat list of chunk Documents ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []

    for doc in documents:
        raw_chunks = splitter.split_text(doc.page_content)
        for idx, text in enumerate(raw_chunks):
            chunk_meta = {**doc.metadata, "chunk_index": idx}
            # Generate a unique ID for this specific chunk
            parent_doc_id = doc.metadata.get("doc_id", "unknown")
            chunk_meta["chunk_id"] = f"{parent_doc_id}_c{idx}"
            all_chunks.append(Document(page_content=text, metadata=chunk_meta))

    return all_chunks
