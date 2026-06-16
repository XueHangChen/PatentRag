"""Tool layer wrapping retrieval, graph, and GraphRAG capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from patent_rag.agent.schemas import (
    GraphKeywordMatch,
    GraphSearchToolResult,
    PatentSearchToolResult,
    PatentSummaryToolResult,
)
from patent_rag.config import get_settings
from patent_rag.domain import PatentDocument
from patent_rag.graph import find_patents_by_keyword, read_graph_json
from patent_rag.ingestion.jsonl import read_jsonl
from patent_rag.llm import ChatClient
from patent_rag.rag import RagAnswer, RagService
from patent_rag.rag.service import RetrievalMode
from patent_rag.retrieval import (
    RetrievalPostprocessConfig,
    postprocess_hits,
    search_chunks,
    search_hybrid_chunks,
    search_vector_chunks,
)

PatentToolRetrievalMode = Literal["keyword", "vector", "hybrid"]


class PatentAgentTools:
    """A typed tool facade used by the Agent service."""

    def __init__(
        self,
        *,
        patents_path: Path = Path("data") / "processed" / "patents.jsonl",
        chunks_path: Path = Path("data") / "processed" / "chunks.jsonl",
        index_path: Path = Path("data") / "indexes" / "chroma",
        graph_path: Path | None = None,
        collection_name: str = "patent_chunks",
        chat_client: ChatClient | None = None,
        embedding_provider: str | None = None,
        embedding_model_name: str | None = None,
        embedding_dimension: int | None = None,
        postprocess_config: RetrievalPostprocessConfig | None = None,
    ) -> None:
        self.patents_path = patents_path
        self.chunks_path = chunks_path
        self.index_path = index_path
        self.graph_path = graph_path or get_settings().graph_path
        self.collection_name = collection_name
        self.chat_client = chat_client
        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        self.embedding_dimension = embedding_dimension
        self.postprocess_config = postprocess_config or RetrievalPostprocessConfig()

    def search_patents(
        self,
        query: str,
        *,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = "hybrid",
    ) -> PatentSearchToolResult:
        """Search patent chunks without calling an LLM."""

        candidate_k = max(top_k * 3, top_k)
        if retrieval_mode == "keyword":
            hits = search_chunks(self.chunks_path, query, top_k=candidate_k)
        elif retrieval_mode == "vector":
            hits = search_vector_chunks(
                self.index_path,
                query,
                top_k=candidate_k,
                collection_name=self.collection_name,
                embedding_provider=self.embedding_provider,
                embedding_model_name=self.embedding_model_name,
                embedding_dimension=self.embedding_dimension,
            )
        else:
            hits = search_hybrid_chunks(
                self.chunks_path,
                self.index_path,
                query,
                top_k=candidate_k,
                collection_name=self.collection_name,
                embedding_provider=self.embedding_provider,
                embedding_model_name=self.embedding_model_name,
                embedding_dimension=self.embedding_dimension,
            )
        return PatentSearchToolResult(
            query=query,
            retrieval_mode=retrieval_mode,
            hits=postprocess_hits(hits, top_k=top_k, config=self.postprocess_config),
        )

    def graph_search(self, keyword: str, *, limit: int = 5) -> GraphSearchToolResult:
        """Search patent matches through the persisted knowledge graph."""

        if not self.graph_path.exists():
            return GraphSearchToolResult(keyword=keyword, matches=[])
        graph = read_graph_json(self.graph_path)
        raw_matches = find_patents_by_keyword(graph, keyword, limit=limit)
        return GraphSearchToolResult(
            keyword=keyword,
            matches=[GraphKeywordMatch(**match) for match in raw_matches],
        )

    def summarize_patent(self, identifier: str) -> PatentSummaryToolResult:
        """Create a deterministic summary from structured patent JSONL."""

        document = self._find_patent(identifier)
        if document is None:
            return PatentSummaryToolResult(found=False)

        metadata = document.metadata
        return PatentSummaryToolResult(
            found=True,
            patent_id=metadata.patent_id,
            title=metadata.title,
            patent_type=metadata.patent_type.value,
            applicants=metadata.applicants,
            inventors=metadata.inventors,
            ipc_classes=metadata.ipc_classes,
            abstract=_truncate(document.abstract, 600),
            first_claim=_truncate(document.claims[0].text, 600) if document.claims else None,
            claim_count=len(document.claims),
            section_names=[section.name for section in document.sections],
        )

    def graph_rag_answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        retrieval_mode: RetrievalMode = "hybrid",
        use_graph: bool = True,
        graph_top_k: int = 3,
    ) -> RagAnswer:
        """Run the existing GraphRAG answer tool."""

        service = RagService(
            chunks_path=self.chunks_path,
            index_path=self.index_path,
            graph_path=self.graph_path,
            collection_name=self.collection_name,
            chat_client=self.chat_client,
            embedding_provider=self.embedding_provider,
            embedding_model_name=self.embedding_model_name,
            embedding_dimension=self.embedding_dimension,
            postprocess_config=self.postprocess_config,
        )
        return service.answer(
            question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            use_graph=use_graph,
            graph_top_k=graph_top_k,
        )

    def _find_patent(self, identifier: str) -> PatentDocument | None:
        if not self.patents_path.exists():
            return None
        normalized_identifier = identifier.strip().upper()
        documents = read_jsonl(self.patents_path, PatentDocument)
        for document in documents:
            metadata = document.metadata
            if metadata.patent_id.upper() == normalized_identifier:
                return document
            if metadata.title and normalized_identifier in metadata.title.upper():
                return document
        return None


def _truncate(value: str | None, max_chars: int) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."
