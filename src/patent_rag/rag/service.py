"""Patent RAG orchestration service."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from patent_rag.llm import ChatClient, create_chat_client
from patent_rag.rag.graph_context import GraphEvidence, retrieve_graph_evidence
from patent_rag.rag.prompting import build_rag_messages
from patent_rag.retrieval import (
    RetrievalPostprocessConfig,
    SearchHit,
    postprocess_hits,
    search_chunks,
    search_hybrid_chunks,
    search_vector_chunks,
)

RetrievalMode = Literal["keyword", "vector", "hybrid"]


class RagSource(BaseModel):
    """A source chunk cited by a RAG answer."""

    source_id: str
    chunk_id: str
    patent_id: str
    title: str
    section: str
    claim_number: int | None = None
    score: float
    snippet: str
    source_file: str


class RagGraphSource(BaseModel):
    """A graph neighborhood cited by a RAG answer."""

    source_id: str
    patent_id: str
    title: str
    score: float
    matched_terms: list[str] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    applicants: list[str] = Field(default_factory=list)
    inventors: list[str] = Field(default_factory=list)
    ipc_classes: list[str] = Field(default_factory=list)
    section_names: list[str] = Field(default_factory=list)
    claim_numbers: list[int] = Field(default_factory=list)
    relation_summary: str


class RagAnswer(BaseModel):
    """A grounded answer and its retrieved sources."""

    question: str
    answer: str
    retrieval_mode: RetrievalMode
    top_k: int
    use_graph: bool = False
    sources: list[RagSource] = Field(default_factory=list)
    graph_sources: list[RagGraphSource] = Field(default_factory=list)


class RagService:
    """Retrieve patent evidence and ask an LLM to answer with citations."""

    def __init__(
        self,
        *,
        chunks_path: Path = Path("data") / "processed" / "chunks.jsonl",
        index_path: Path = Path("data") / "indexes" / "chroma",
        graph_path: Path = Path("data") / "graph" / "patent_graph.json",
        collection_name: str = "patent_chunks",
        chat_client: ChatClient | None = None,
        embedding_provider: str | None = None,
        embedding_model_name: str | None = None,
        embedding_dimension: int | None = None,
        postprocess_config: RetrievalPostprocessConfig | None = None,
    ) -> None:
        self.chunks_path = chunks_path
        self.index_path = index_path
        self.graph_path = graph_path
        self.collection_name = collection_name
        self.chat_client = chat_client or create_chat_client()
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        self.embedding_dimension = embedding_dimension
        self.postprocess_config = postprocess_config or RetrievalPostprocessConfig()

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = "hybrid",
        use_graph: bool = False,
        graph_top_k: int = 3,
    ) -> RagAnswer:
        """Answer a question using retrieved patent evidence."""

        hits = self.retrieve(question, top_k=top_k, retrieval_mode=retrieval_mode)
        graph_evidence = self.retrieve_graph_context(
            question,
            hits,
            top_k=graph_top_k,
        ) if use_graph else []
        answer = generate_answer_from_hits(
            question,
            hits,
            self.chat_client,
            graph_evidence=graph_evidence,
        )
        return RagAnswer(
            question=question,
            answer=answer,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            use_graph=use_graph,
            sources=_to_sources(hits),
            graph_sources=_to_graph_sources(graph_evidence),
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        retrieval_mode: RetrievalMode,
    ) -> list[SearchHit]:
        """Retrieve patent chunks for a question."""

        candidate_k = max(top_k * 3, top_k)
        if retrieval_mode == "keyword":
            hits = search_chunks(self.chunks_path, question, top_k=candidate_k)
            return postprocess_hits(hits, top_k=top_k, config=self.postprocess_config)

        if retrieval_mode == "vector":
            hits = search_vector_chunks(
                self.index_path,
                question,
                top_k=candidate_k,
                collection_name=self.collection_name,
                embedding_provider=self.embedding_provider,
                embedding_model_name=self.embedding_model_name,
                embedding_dimension=self.embedding_dimension,
            )
            return postprocess_hits(hits, top_k=top_k, config=self.postprocess_config)

        hits = search_hybrid_chunks(
            self.chunks_path,
            self.index_path,
            question,
            top_k=candidate_k,
            collection_name=self.collection_name,
            embedding_provider=self.embedding_provider,
            embedding_model_name=self.embedding_model_name,
            embedding_dimension=self.embedding_dimension,
        )
        return postprocess_hits(hits, top_k=top_k, config=self.postprocess_config)

    def retrieve_graph_context(
        self,
        question: str,
        hits: list[SearchHit],
        *,
        top_k: int = 3,
    ) -> list[GraphEvidence]:
        """Retrieve graph evidence related to a question and retrieved chunks."""

        return retrieve_graph_evidence(
            question,
            hits,
            graph_path=self.graph_path,
            top_k=top_k,
        )


def answer_patent_question(
    question: str,
    *,
    chat_client: ChatClient | None = None,
    chunks_path: Path = Path("data") / "processed" / "chunks.jsonl",
    index_path: Path = Path("data") / "indexes" / "chroma",
    graph_path: Path = Path("data") / "graph" / "patent_graph.json",
    collection_name: str = "patent_chunks",
    top_k: int = 5,
    retrieval_mode: RetrievalMode = "hybrid",
    use_graph: bool = False,
    graph_top_k: int = 3,
    embedding_provider: str | None = None,
    embedding_model_name: str | None = None,
    embedding_dimension: int | None = None,
    postprocess_config: RetrievalPostprocessConfig | None = None,
) -> RagAnswer:
    """Convenience helper for one-shot patent RAG answers."""

    service = RagService(
        chunks_path=chunks_path,
        index_path=index_path,
        graph_path=graph_path,
        collection_name=collection_name,
        chat_client=chat_client,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        embedding_dimension=embedding_dimension,
        postprocess_config=postprocess_config,
    )
    return service.answer(
        question,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        use_graph=use_graph,
        graph_top_k=graph_top_k,
    )


def generate_answer_from_hits(
    question: str,
    hits: list[SearchHit],
    chat_client: ChatClient,
    *,
    graph_evidence: list[GraphEvidence] | None = None,
) -> str:
    """Generate a grounded answer from already retrieved hits."""

    messages = build_rag_messages(question, hits, graph_evidence=graph_evidence)
    return chat_client.generate(messages)


def _to_sources(hits: list[SearchHit]) -> list[RagSource]:
    return [
        RagSource(
            source_id=f"S{index}",
            chunk_id=hit.chunk_id,
            patent_id=hit.patent_id,
            title=hit.title,
            section=hit.section,
            claim_number=hit.claim_number,
            score=hit.score,
            snippet=hit.snippet,
            source_file=hit.source_file,
        )
        for index, hit in enumerate(hits, start=1)
    ]


def _to_graph_sources(graph_evidence: list[GraphEvidence]) -> list[RagGraphSource]:
    return [
        RagGraphSource(
            source_id=evidence.source_id,
            patent_id=evidence.patent_id,
            title=evidence.title,
            score=evidence.score,
            matched_terms=evidence.matched_terms,
            supporting_chunk_ids=evidence.supporting_chunk_ids,
            keywords=evidence.keywords,
            applicants=evidence.applicants,
            inventors=evidence.inventors,
            ipc_classes=evidence.ipc_classes,
            section_names=evidence.section_names,
            claim_numbers=evidence.claim_numbers,
            relation_summary=evidence.relation_summary,
        )
        for evidence in graph_evidence
    ]
