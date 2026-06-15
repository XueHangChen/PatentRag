"""Chroma-backed vector retrieval for patent chunks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from patent_rag.domain import PatentChunk
from patent_rag.retrieval.embedding import EmbeddingModel, create_embedding_model
from patent_rag.retrieval.keyword import SearchHit

DEFAULT_CHROMA_PATH = Path("data") / "indexes" / "chroma"
DEFAULT_COLLECTION_NAME = "patent_chunks"


class ChromaDependencyError(RuntimeError):
    """Raised when chromadb is not installed."""


class ChromaVectorStore:
    """Persistent Chroma vector store for patent chunks."""

    def __init__(
        self,
        persist_path: Path = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model or create_embedding_model()
        self._client = self._create_client()
        self._collection = self._get_or_create_collection()

    def index_chunks(
        self,
        chunks: list[PatentChunk],
        *,
        reset: bool = True,
        batch_size: int = 64,
    ) -> int:
        """Index chunks into Chroma and return the number of indexed chunks."""

        if reset:
            self.reset()
        if not chunks:
            return 0

        for batch in _batched(chunks, batch_size):
            embeddings = self.embedding_model.embed_documents([chunk.text for chunk in batch])
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=embeddings,
                documents=[chunk.text for chunk in batch],
                metadatas=[_chunk_metadata(chunk) for chunk in batch],
            )
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        where: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Search Chroma by query embedding."""

        if top_k <= 0:
            return []

        result = self._collection.query(
            query_embeddings=[self.embedding_model.embed_query(query)],
            n_results=top_k,
            where=dict(where) if where else None,
            include=["documents", "metadatas", "distances"],
        )
        return _query_result_to_hits(result)

    def count(self) -> int:
        """Return the number of indexed chunks."""

        return int(self._collection.count())

    def reset(self) -> None:
        """Delete and recreate the collection."""

        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._get_or_create_collection()

    def _create_client(self):
        try:
            import chromadb
        except ImportError as exc:
            raise ChromaDependencyError(
                "chromadb is required for vector retrieval. "
                'Install it with `pip install -e ".[vector]"`.'
            ) from exc

        self.persist_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self.persist_path))

    def _get_or_create_collection(self):
        metadata: dict[str, str | int] = {
            "embedding_model": self.embedding_model.model_name,
        }
        if self.embedding_model.dimension is not None:
            metadata["embedding_dimension"] = self.embedding_model.dimension

        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata,
            embedding_function=None,
        )


def _chunk_metadata(chunk: PatentChunk) -> dict[str, str | int | bool]:
    metadata: dict[str, str | int | bool] = {
        "chunk_id": chunk.chunk_id,
        "patent_id": chunk.patent_id,
        "section": chunk.section,
        "claim_number": chunk.claim_number if chunk.claim_number is not None else -1,
        "has_claim_number": chunk.claim_number is not None,
        "source_file": str(chunk.source_file),
    }
    for key, value in chunk.metadata.items():
        if value not in ("", None):
            metadata[key] = value
    return metadata


def _query_result_to_hits(result: Mapping[str, Any]) -> list[SearchHit]:
    documents = _first_result_list(result.get("documents"))
    metadatas = _first_result_list(result.get("metadatas"))
    distances = _first_result_list(result.get("distances"))

    hits: list[SearchHit] = []
    for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
        if not isinstance(metadata, dict):
            continue
        claim_number = int(metadata.get("claim_number", -1))
        hits.append(
            SearchHit(
                chunk_id=str(metadata.get("chunk_id", "")),
                patent_id=str(metadata.get("patent_id", "")),
                title=str(metadata.get("title", "")),
                section=str(metadata.get("section", "")),
                claim_number=claim_number if claim_number >= 0 else None,
                score=_distance_to_score(float(distance)),
                snippet=str(document)[:180],
                source_file=str(metadata.get("source_file", "")),
            )
        )
    return hits


def _first_result_list(value: Any) -> list[Any]:
    if not value:
        return []
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def _distance_to_score(distance: float) -> float:
    return round(1.0 / (1.0 + max(distance, 0.0)), 4)


def _batched(records: list[PatentChunk], batch_size: int) -> Iterable[list[PatentChunk]]:
    safe_batch_size = max(batch_size, 1)
    for start in range(0, len(records), safe_batch_size):
        yield records[start : start + safe_batch_size]
