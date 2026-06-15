"""Keyword, vector, hybrid, and graph-augmented retrieval."""

from patent_rag.retrieval.chroma_store import ChromaVectorStore
from patent_rag.retrieval.chunking import build_chunks, build_chunks_for_document
from patent_rag.retrieval.embedding import HashingEmbeddingModel, create_embedding_model
from patent_rag.retrieval.keyword import KeywordSearchIndex, SearchHit
from patent_rag.retrieval.postprocess import RetrievalPostprocessConfig, postprocess_hits
from patent_rag.retrieval.service import (
    build_chunks_from_patents_file,
    build_vector_index,
    load_chunks,
    search_chunks,
    search_hybrid_chunks,
    search_vector_chunks,
)

__all__ = [
    "ChromaVectorStore",
    "HashingEmbeddingModel",
    "KeywordSearchIndex",
    "RetrievalPostprocessConfig",
    "SearchHit",
    "build_chunks",
    "build_chunks_for_document",
    "build_chunks_from_patents_file",
    "build_vector_index",
    "create_embedding_model",
    "load_chunks",
    "postprocess_hits",
    "search_chunks",
    "search_hybrid_chunks",
    "search_vector_chunks",
]
