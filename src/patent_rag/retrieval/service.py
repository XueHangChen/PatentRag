"""Retrieval service loading processed chunks from disk."""

from pathlib import Path

from patent_rag.domain import PatentChunk, PatentDocument
from patent_rag.ingestion.jsonl import read_jsonl, write_jsonl
from patent_rag.retrieval.chroma_store import (
    DEFAULT_CHROMA_PATH,
    DEFAULT_COLLECTION_NAME,
    ChromaVectorStore,
)
from patent_rag.retrieval.chunking import build_chunks
from patent_rag.retrieval.embedding import create_embedding_model
from patent_rag.retrieval.keyword import KeywordSearchIndex, SearchHit


def build_chunks_from_patents_file(input_path: Path, output_path: Path) -> int:
    """Build retrieval chunks from processed patent JSONL."""

    documents = read_jsonl(input_path, PatentDocument)
    chunks = build_chunks(documents)
    return write_jsonl(chunks, output_path)


def load_chunks(input_path: Path) -> list[PatentChunk]:
    """Load retrieval chunks from JSONL."""

    return read_jsonl(input_path, PatentChunk)


def search_chunks(chunks_path: Path, query: str, top_k: int = 8) -> list[SearchHit]:
    """Run keyword search over persisted chunks."""

    chunks = load_chunks(chunks_path)
    index = KeywordSearchIndex(chunks)
    return index.search(query, top_k=top_k)


def build_vector_index(
    chunks_path: Path,
    index_path: Path = DEFAULT_CHROMA_PATH,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_provider: str | None = None,
    embedding_model_name: str | None = None,
    embedding_dimension: int | None = None,
    reset: bool = True,
    batch_size: int = 64,
) -> int:
    """Build a Chroma vector index from persisted chunks."""

    chunks = load_chunks(chunks_path)
    embedding_model = create_embedding_model(
        provider=embedding_provider,
        model_name=embedding_model_name,
        dimension=embedding_dimension,
    )
    store = ChromaVectorStore(
        index_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    return store.index_chunks(chunks, reset=reset, batch_size=batch_size)


def search_vector_chunks(
    index_path: Path,
    query: str,
    *,
    top_k: int = 8,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_provider: str | None = None,
    embedding_model_name: str | None = None,
    embedding_dimension: int | None = None,
) -> list[SearchHit]:
    """Run vector search over a persisted Chroma index."""

    embedding_model = create_embedding_model(
        provider=embedding_provider,
        model_name=embedding_model_name,
        dimension=embedding_dimension,
    )
    store = ChromaVectorStore(
        index_path,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    return store.search(query, top_k=top_k)


def search_hybrid_chunks(
    chunks_path: Path,
    index_path: Path,
    query: str,
    *,
    top_k: int = 8,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    keyword_weight: float = 0.45,
    vector_weight: float = 0.55,
    embedding_provider: str | None = None,
    embedding_model_name: str | None = None,
    embedding_dimension: int | None = None,
) -> list[SearchHit]:
    """Combine BM25 keyword search and Chroma vector search."""

    candidate_count = max(top_k * 3, top_k)
    keyword_hits = search_chunks(chunks_path, query, top_k=candidate_count)
    vector_hits = search_vector_chunks(
        index_path,
        query,
        top_k=candidate_count,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        embedding_dimension=embedding_dimension,
    )
    return _merge_ranked_hits(
        keyword_hits,
        vector_hits,
        top_k=top_k,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
    )


def _merge_ranked_hits(
    keyword_hits: list[SearchHit],
    vector_hits: list[SearchHit],
    *,
    top_k: int,
    keyword_weight: float,
    vector_weight: float,
) -> list[SearchHit]:
    keyword_scores = _normalize_scores(keyword_hits)
    vector_scores = _normalize_scores(vector_hits)
    hits_by_id = {hit.chunk_id: hit for hit in vector_hits}
    hits_by_id.update({hit.chunk_id: hit for hit in keyword_hits})

    scored_hits: list[tuple[float, SearchHit]] = []
    for chunk_id, hit in hits_by_id.items():
        score = keyword_weight * keyword_scores.get(chunk_id, 0.0)
        score += vector_weight * vector_scores.get(chunk_id, 0.0)
        scored_hits.append((score, hit.model_copy(update={"score": round(score, 4)})))

    scored_hits.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in scored_hits[:top_k]]


def _normalize_scores(hits: list[SearchHit]) -> dict[str, float]:
    if not hits:
        return {}
    max_score = max(hit.score for hit in hits)
    if max_score <= 0:
        return {hit.chunk_id: 0.0 for hit in hits}
    return {hit.chunk_id: hit.score / max_score for hit in hits}
