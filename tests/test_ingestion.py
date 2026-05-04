"""Tests for the ingestion layer — chunker and embedder logic."""

import pytest
from langchain_core.documents import Document

from ingestion.chunker import chunk_documents


def make_doc(text: str, **meta) -> Document:
    default_meta = {
        "company": "Test Corp",
        "ticker": "TEST",
        "source_type": "filings",
        "doc_id": "TEST_10-K_20220101_abc12345",
    }
    default_meta.update(meta)
    return Document(page_content=text, metadata=default_meta)


def test_chunk_single_short_document():
    """A document shorter than CHUNK_SIZE should produce exactly one chunk."""
    doc = make_doc("Short text that fits in one chunk.")
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text that fits in one chunk."


def test_chunk_long_document_produces_multiple_chunks():
    """A long document should be split into multiple chunks."""
    long_text = "Financial statement analysis. " * 200  # ~6000 chars
    doc = make_doc(long_text)
    chunks = chunk_documents([doc])
    assert len(chunks) > 1


def test_chunks_inherit_metadata():
    """Each chunk should inherit all metadata from the parent document."""
    doc = make_doc("Some text.", filing_type="10-K", filing_date="2022-01-01")
    chunks = chunk_documents([doc])
    for chunk in chunks:
        assert chunk.metadata["ticker"] == "TEST"
        assert chunk.metadata["filing_type"] == "10-K"
        assert chunk.metadata["filing_date"] == "2022-01-01"


def test_chunks_have_chunk_index():
    """Each chunk should have a sequential chunk_index."""
    long_text = "Word " * 500
    doc = make_doc(long_text)
    chunks = chunk_documents([doc])
    assert all("chunk_index" in c.metadata for c in chunks)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunks_have_unique_chunk_ids():
    """Every chunk should have a unique chunk_id."""
    long_text = "Word " * 500
    doc = make_doc(long_text)
    chunks = chunk_documents([doc])
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"


def test_chunk_id_format():
    """chunk_id should follow {doc_id}_c{index} pattern."""
    doc = make_doc("Some text.")
    chunks = chunk_documents([doc])
    assert chunks[0].metadata["chunk_id"].startswith("TEST_10-K_20220101_abc12345_c0")


def test_multiple_documents_chunked_independently():
    """Multiple documents should be chunked independently, not merged."""
    doc1 = make_doc("Document one content.", doc_id="DOC1_abc")
    doc2 = make_doc("Document two content.", doc_id="DOC2_def")
    chunks = chunk_documents([doc1, doc2])
    doc1_chunks = [c for c in chunks if "DOC1" in c.metadata["doc_id"]]
    doc2_chunks = [c for c in chunks if "DOC2" in c.metadata["doc_id"]]
    assert len(doc1_chunks) >= 1
    assert len(doc2_chunks) >= 1


def test_empty_document_list():
    """Chunking an empty list should return an empty list."""
    assert chunk_documents([]) == []
