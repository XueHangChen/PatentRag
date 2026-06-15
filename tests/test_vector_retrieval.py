from pathlib import Path

from patent_rag.ingestion.jsonl import write_jsonl
from patent_rag.ingestion.pipeline import parse_patent_directory
from patent_rag.retrieval import (
    ChromaVectorStore,
    HashingEmbeddingModel,
    build_chunks,
    build_vector_index,
    search_hybrid_chunks,
    search_vector_chunks,
)

LIQUID_NITROGEN_QUERY = "\u6db2\u6c2e\u7f50\u8fd0\u8f93\u56fa\u5b9a"


def test_chroma_vector_store_indexes_and_searches_patent_chunks(tmp_path: Path) -> None:
    documents, errors = parse_patent_directory(Path("patant"))

    assert errors == {}

    chunks = build_chunks(documents)
    store = ChromaVectorStore(
        tmp_path / "chroma",
        collection_name="test_patent_chunks",
        embedding_model=HashingEmbeddingModel(),
    )

    indexed_count = store.index_chunks(chunks)
    hits = store.search(LIQUID_NITROGEN_QUERY, top_k=5)

    assert indexed_count == len(chunks)
    assert store.count() == len(chunks)
    assert hits
    assert any(hit.patent_id == "CN206539886U" for hit in hits)
    assert all(hit.chunk_id for hit in hits)


def test_vector_and_hybrid_service_search(tmp_path: Path) -> None:
    documents, errors = parse_patent_directory(Path("patant"))

    assert errors == {}

    chunks = build_chunks(documents)
    chunks_path = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "chroma"
    collection_name = "test_service_patent_chunks"
    write_jsonl(chunks, chunks_path)

    indexed_count = build_vector_index(
        chunks_path,
        index_path,
        collection_name=collection_name,
        embedding_provider="hashing",
    )
    vector_hits = search_vector_chunks(
        index_path,
        LIQUID_NITROGEN_QUERY,
        top_k=5,
        collection_name=collection_name,
        embedding_provider="hashing",
    )
    hybrid_hits = search_hybrid_chunks(
        chunks_path,
        index_path,
        LIQUID_NITROGEN_QUERY,
        top_k=5,
        collection_name=collection_name,
        embedding_provider="hashing",
    )

    assert indexed_count == len(chunks)
    assert any(hit.patent_id == "CN206539886U" for hit in vector_hits)
    assert hybrid_hits[0].patent_id == "CN206539886U"
