"""Retrieval-augmented generation services."""

from patent_rag.rag.graph_context import GraphEvidence, retrieve_graph_evidence
from patent_rag.rag.prompting import (
    build_rag_messages,
    render_evidence_context,
    render_graph_evidence_context,
)
from patent_rag.rag.service import (
    RagAnswer,
    RagGraphSource,
    RagService,
    RagSource,
    answer_patent_question,
)

__all__ = [
    "GraphEvidence",
    "RagAnswer",
    "RagGraphSource",
    "RagService",
    "RagSource",
    "answer_patent_question",
    "build_rag_messages",
    "render_evidence_context",
    "render_graph_evidence_context",
    "retrieve_graph_evidence",
]
