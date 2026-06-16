"""Graph evidence retrieval for GraphRAG answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from patent_rag.graph import read_graph_json
from patent_rag.retrieval import SearchHit

GRAPH_QUERY_TERM_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}")
TECHNICAL_LABELS = {"TechnicalField", "Problem", "Component", "Solution", "Effect"}
TECHNICAL_RELATIONS = {
    "HAS_TECHNICAL_FIELD",
    "SOLVES_PROBLEM",
    "USES_COMPONENT",
    "PROPOSES_SOLUTION",
    "HAS_EFFECT",
}


class GraphEvidence(BaseModel):
    """A compact graph neighborhood used as RAG evidence."""

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
    technical_fields: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    solutions: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    relation_summary: str


@dataclass
class _GraphCandidate:
    """Intermediate patent candidate scored from retrieval hits and graph matches."""

    patent_node_id: str
    score: float = 0.0
    matched_terms: set[str] = field(default_factory=set)
    supporting_chunk_ids: list[str] = field(default_factory=list)


def retrieve_graph_evidence(
    question: str,
    hits: list[SearchHit],
    *,
    graph_path: Path,
    top_k: int = 3,
) -> list[GraphEvidence]:
    """Retrieve patent graph neighborhoods related to the question and retrieved chunks."""

    if not graph_path.exists() or top_k <= 0:
        return []

    graph = read_graph_json(graph_path)
    patent_nodes = _patent_nodes_by_id(graph)
    candidates: dict[str, _GraphCandidate] = {}

    for rank, hit in enumerate(hits, start=1):
        patent_node_id = patent_nodes.get(hit.patent_id)
        if patent_node_id is None:
            continue
        candidate = candidates.setdefault(
            hit.patent_id,
            _GraphCandidate(patent_node_id=patent_node_id),
        )
        candidate.score += max(hit.score, 0.0) + 1 / rank
        if hit.chunk_id not in candidate.supporting_chunk_ids:
            candidate.supporting_chunk_ids.append(hit.chunk_id)

    query_terms = _extract_query_terms(question)
    for node_id, data in graph.nodes(data=True):
        label = data.get("label")
        if label != "Keyword" and label not in TECHNICAL_LABELS:
            continue
        node_name = str(data.get("name", ""))
        matched_terms = _matched_terms(node_name, query_terms, question)
        if not matched_terms:
            continue
        for source_id, _, edge_data in graph.in_edges(node_id, data=True):
            relation = edge_data.get("relation")
            if relation != "MENTIONS_KEYWORD" and relation not in TECHNICAL_RELATIONS:
                continue
            source_data = graph.nodes[source_id]
            if source_data.get("label") != "Patent":
                continue
            patent_id = str(source_data.get("properties", {}).get("patent_id", source_id))
            candidate = candidates.setdefault(
                patent_id,
                _GraphCandidate(patent_node_id=source_id),
            )
            candidate.score += 0.5 + 0.1 * len(matched_terms)
            candidate.matched_terms.update(matched_terms)

    ranked_candidates = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
    return [
        _build_graph_evidence(graph, candidate, source_index=index)
        for index, candidate in enumerate(ranked_candidates[:top_k], start=1)
    ]


def _patent_nodes_by_id(graph: Any) -> dict[str, str]:
    patent_nodes: dict[str, str] = {}
    for node_id, data in graph.nodes(data=True):
        if data.get("label") != "Patent":
            continue
        patent_id = data.get("properties", {}).get("patent_id", node_id)
        patent_nodes[str(patent_id)] = str(node_id)
    return patent_nodes


def _build_graph_evidence(
    graph: Any,
    candidate: _GraphCandidate,
    *,
    source_index: int,
) -> GraphEvidence:
    patent_data = graph.nodes[candidate.patent_node_id]
    properties = patent_data.get("properties", {})
    patent_id = str(properties.get("patent_id", candidate.patent_node_id))
    title = str(patent_data.get("name", patent_id))

    keywords: list[str] = []
    applicants: list[str] = []
    inventors: list[str] = []
    ipc_classes: list[str] = []
    section_names: list[str] = []
    claim_numbers: list[int] = []
    technical_fields: list[str] = []
    problems: list[str] = []
    components: list[str] = []
    solutions: list[str] = []
    effects: list[str] = []

    for _, target_id, edge_data in graph.out_edges(candidate.patent_node_id, data=True):
        relation = edge_data.get("relation")
        target_data = graph.nodes[target_id]
        target_name = str(target_data.get("name", ""))
        target_properties = target_data.get("properties", {})

        if relation == "MENTIONS_KEYWORD":
            _append_unique(keywords, target_name)
        elif relation == "HAS_APPLICANT":
            _append_unique(applicants, target_name)
        elif relation == "HAS_INVENTOR":
            _append_unique(inventors, target_name)
        elif relation == "HAS_IPC":
            _append_unique(ipc_classes, target_name)
        elif relation == "HAS_SECTION":
            _append_unique(section_names, target_name)
        elif relation == "HAS_CLAIM":
            claim_number = _safe_int(target_properties.get("claim_number"))
            if claim_number is not None and claim_number not in claim_numbers:
                claim_numbers.append(claim_number)
        elif relation == "HAS_TECHNICAL_FIELD":
            _append_unique(technical_fields, target_name)
        elif relation == "SOLVES_PROBLEM":
            _append_unique(problems, target_name)
        elif relation == "USES_COMPONENT":
            _append_unique(components, target_name)
        elif relation == "PROPOSES_SOLUTION":
            _append_unique(solutions, target_name)
        elif relation == "HAS_EFFECT":
            _append_unique(effects, target_name)

    claim_numbers.sort()
    relation_summary = _build_relation_summary(
        keywords=keywords,
        applicants=applicants,
        inventors=inventors,
        ipc_classes=ipc_classes,
        section_names=section_names,
        claim_numbers=claim_numbers,
        technical_fields=technical_fields,
        problems=problems,
        components=components,
        solutions=solutions,
        effects=effects,
    )
    return GraphEvidence(
        source_id=f"G{source_index}",
        patent_id=patent_id,
        title=title,
        score=round(candidate.score, 4),
        matched_terms=sorted(candidate.matched_terms),
        supporting_chunk_ids=candidate.supporting_chunk_ids,
        keywords=keywords,
        applicants=applicants,
        inventors=inventors,
        ipc_classes=ipc_classes,
        section_names=section_names,
        claim_numbers=claim_numbers,
        technical_fields=technical_fields,
        problems=problems,
        components=components,
        solutions=solutions,
        effects=effects,
        relation_summary=relation_summary,
    )


def _build_relation_summary(
    *,
    keywords: list[str],
    applicants: list[str],
    inventors: list[str],
    ipc_classes: list[str],
    section_names: list[str],
    claim_numbers: list[int],
    technical_fields: list[str],
    problems: list[str],
    components: list[str],
    solutions: list[str],
    effects: list[str],
) -> str:
    parts: list[str] = []
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords[:8])}")
    if applicants:
        parts.append(f"Applicants: {', '.join(applicants[:4])}")
    if inventors:
        parts.append(f"Inventors: {', '.join(inventors[:4])}")
    if ipc_classes:
        parts.append(f"IPC: {', '.join(ipc_classes[:4])}")
    if section_names:
        parts.append(f"Sections: {', '.join(section_names[:6])}")
    if claim_numbers:
        parts.append(f"Claim count: {len(claim_numbers)}")
    if technical_fields:
        parts.append(f"Technical fields: {', '.join(technical_fields[:5])}")
    if problems:
        parts.append(f"Technical problems: {', '.join(problems[:5])}")
    if components:
        parts.append(f"Key components: {', '.join(components[:8])}")
    if solutions:
        parts.append(f"Technical solutions: {', '.join(solutions[:5])}")
    if effects:
        parts.append(f"Technical effects: {', '.join(effects[:5])}")
    return "; ".join(parts) if parts else "Graph contains this patent node only."


def _extract_query_terms(question: str) -> set[str]:
    terms: set[str] = set()
    for match in GRAPH_QUERY_TERM_PATTERN.findall(question):
        terms.add(match)
        if _is_chinese(match):
            terms.update(_sliding_terms(match, min_size=2, max_size=4))
    return {term for term in terms if len(term) >= 2}


def _matched_terms(node_name: str, query_terms: set[str], question: str) -> set[str]:
    matches: set[str] = set()
    node_name_cf = node_name.casefold()
    question_cf = question.casefold()
    if node_name and node_name_cf in question_cf:
        matches.add(node_name)
    for term in query_terms:
        term_cf = term.casefold()
        if term_cf in node_name_cf or node_name_cf in term_cf:
            matches.add(term)
    return matches


def _sliding_terms(text: str, *, min_size: int, max_size: int) -> set[str]:
    terms: set[str] = set()
    upper = min(max_size, len(text))
    for size in range(min_size, upper + 1):
        for start in range(0, len(text) - size + 1):
            terms.add(text[start : start + size])
    return terms


def _is_chinese(text: str) -> bool:
    return all("\u4e00" <= char <= "\u9fff" for char in text)


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
