"""Rule-based knowledge graph extraction from structured patent documents."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from patent_rag.domain import EvidenceSpan, GraphEdge, GraphNode, PatentDocument

TECHNICAL_SUFFIXES = (
    "固定架",
    "支架",
    "材料",
    "装置",
    "设备",
    "仪器",
    "雾化器",
    "清洗框",
    "护理床",
    "病床",
    "给药器",
    "辅助架",
    "辅助装置",
    "旋转台",
    "抢救车",
)
STOP_KEYWORDS = {
    "一种",
    "用于",
    "包括",
    "所述",
    "进行",
    "以及",
    "本发明",
    "本实用新型",
    "技术领域",
    "背景技术",
    "具体实施方式",
}
LEADING_KEYWORD_NOISE = ("一种", "用于", "的", "种")
CHINESE_PHRASE_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,12}")


@dataclass
class GraphExtractionResult:
    """Graph nodes and edges extracted from patent documents."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


class TechnicalGraphExtractor(Protocol):
    """Optional technical graph extractor contract."""

    def extract(self, document: PatentDocument) -> GraphExtractionResult:
        """Extract technical graph records for one patent document."""


def extract_graph_from_documents(
    documents: list[PatentDocument],
    *,
    keyword_limit_per_patent: int = 8,
    technical_extractor: TechnicalGraphExtractor | None = None,
    technical_max_patents: int | None = None,
) -> GraphExtractionResult:
    """Extract a first-pass knowledge graph from structured patents."""

    nodes_by_id: dict[str, GraphNode] = {}
    edges_by_id: dict[str, GraphEdge] = {}
    technical_patent_count = 0

    for document in documents:
        metadata = document.metadata
        patent_node = _patent_node(document)
        _add_node(nodes_by_id, patent_node)

        for applicant in metadata.applicants:
            applicant_node = _entity_node("Applicant", applicant)
            _add_node(nodes_by_id, applicant_node)
            _add_edge(edges_by_id, patent_node.node_id, "HAS_APPLICANT", applicant_node.node_id)

        for inventor in metadata.inventors:
            inventor_node = _entity_node("Inventor", inventor)
            _add_node(nodes_by_id, inventor_node)
            _add_edge(edges_by_id, patent_node.node_id, "HAS_INVENTOR", inventor_node.node_id)

        for ipc_class in metadata.ipc_classes:
            ipc_node = _entity_node("IPC", ipc_class)
            _add_node(nodes_by_id, ipc_node)
            _add_edge(edges_by_id, patent_node.node_id, "HAS_IPC", ipc_node.node_id)

        for section in document.sections:
            section_node = GraphNode(
                node_id=_node_id("Section", f"{metadata.patent_id}:{section.name}"),
                label="Section",
                name=section.name,
                properties={
                    "patent_id": metadata.patent_id,
                    "text_length": str(len(section.text)),
                },
                evidence=[
                    EvidenceSpan(
                        source_file=metadata.source_file,
                        section=section.name,
                        text=section.text[:300],
                    )
                ],
            )
            _add_node(nodes_by_id, section_node)
            _add_edge(edges_by_id, patent_node.node_id, "HAS_SECTION", section_node.node_id)

        for claim in document.claims:
            claim_node = GraphNode(
                node_id=_node_id("Claim", f"{metadata.patent_id}:{claim.claim_number}"),
                label="Claim",
                name=f"{metadata.patent_id} 权利要求{claim.claim_number}",
                properties={
                    "patent_id": metadata.patent_id,
                    "claim_number": str(claim.claim_number),
                    "text_length": str(len(claim.text)),
                },
                evidence=[
                    EvidenceSpan(
                        source_file=metadata.source_file,
                        section="权利要求",
                        text=claim.text[:300],
                    )
                ],
            )
            _add_node(nodes_by_id, claim_node)
            _add_edge(edges_by_id, patent_node.node_id, "HAS_CLAIM", claim_node.node_id)

        for keyword in extract_keywords_for_document(
            document,
            limit=keyword_limit_per_patent,
        ):
            keyword_node = _entity_node("Keyword", keyword)
            _add_node(nodes_by_id, keyword_node)
            _add_edge(edges_by_id, patent_node.node_id, "MENTIONS_KEYWORD", keyword_node.node_id)

        if technical_extractor is not None and (
            technical_max_patents is None
            or technical_max_patents <= 0
            or technical_patent_count < technical_max_patents
        ):
            technical_result = technical_extractor.extract(document)
            _merge_graph_result(nodes_by_id, edges_by_id, technical_result)
            technical_patent_count += 1

    return GraphExtractionResult(
        nodes=sorted(nodes_by_id.values(), key=lambda node: node.node_id),
        edges=sorted(edges_by_id.values(), key=lambda edge: edge.edge_id),
    )


def extract_keywords_for_document(document: PatentDocument, *, limit: int = 8) -> list[str]:
    """Extract lightweight technical keywords without calling an LLM."""

    texts = [
        document.metadata.title or "",
        document.abstract or "",
        " ".join(claim.text for claim in document.claims[:3]),
        " ".join(section.text for section in document.sections[:2]),
    ]
    candidates: Counter[str] = Counter()

    title = document.metadata.title or ""
    for keyword in _title_keywords(title):
        candidates[keyword] += 6

    for text in texts:
        for phrase in CHINESE_PHRASE_PATTERN.findall(text):
            normalized = _normalize_keyword(phrase)
            if _is_keyword_candidate(normalized):
                candidates[normalized] += 1
            for suffix in TECHNICAL_SUFFIXES:
                match = _suffix_keyword(normalized, suffix)
                if match:
                    candidates[match] += 3

    return [
        keyword
        for keyword, _ in candidates.most_common(limit)
    ]


def _patent_node(document: PatentDocument) -> GraphNode:
    metadata = document.metadata
    return GraphNode(
        node_id=f"patent:{metadata.patent_id}",
        label="Patent",
        name=metadata.title or metadata.patent_id,
        properties={
            "patent_id": metadata.patent_id,
            "patent_type": metadata.patent_type.value,
            "publication_number": metadata.publication_number or "",
            "application_number": metadata.application_number or "",
            "application_date": metadata.application_date or "",
            "publication_date": metadata.publication_date or "",
            "source_file": str(metadata.source_file),
        },
    )


def _entity_node(label: str, name: str) -> GraphNode:
    normalized_name = name.strip()
    return GraphNode(
        node_id=_node_id(label, normalized_name),
        label=label,
        name=normalized_name,
    )


def _add_node(nodes_by_id: dict[str, GraphNode], node: GraphNode) -> None:
    if node.node_id not in nodes_by_id:
        nodes_by_id[node.node_id] = node


def _add_edge(
    edges_by_id: dict[str, GraphEdge],
    source_id: str,
    relation: str,
    target_id: str,
) -> None:
    edge_id = _edge_id(source_id, relation, target_id)
    if edge_id not in edges_by_id:
        edges_by_id[edge_id] = GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
        )


def _merge_graph_result(
    nodes_by_id: dict[str, GraphNode],
    edges_by_id: dict[str, GraphEdge],
    result: GraphExtractionResult,
) -> None:
    for node in result.nodes:
        _add_node(nodes_by_id, node)
    for edge in result.edges:
        if edge.edge_id not in edges_by_id:
            edges_by_id[edge.edge_id] = edge


def _title_keywords(title: str) -> list[str]:
    cleaned = _normalize_keyword(title)
    if cleaned.startswith("一种"):
        cleaned = cleaned[2:]
    if not cleaned:
        return []
    keywords = [cleaned]
    for suffix in TECHNICAL_SUFFIXES:
        match = _suffix_keyword(cleaned, suffix)
        if match and match not in keywords:
            keywords.append(match)
    return keywords


def _suffix_keyword(text: str, suffix: str) -> str | None:
    index = text.find(suffix)
    if index == -1:
        return None
    start = max(index - 6, 0)
    return _trim_keyword_noise(text[start : index + len(suffix)])


def _normalize_keyword(value: str) -> str:
    return _trim_keyword_noise(re.sub(r"\s+", "", value.strip("，。；;：:、,.()（）[]【】")))


def _trim_keyword_noise(value: str) -> str:
    keyword = value
    for noise in LEADING_KEYWORD_NOISE:
        if keyword.startswith(noise):
            keyword = keyword[len(noise) :]
    if "的" in keyword[:4]:
        keyword = keyword.split("的", 1)[1]
    return keyword


def _is_keyword_candidate(keyword: str) -> bool:
    if len(keyword) < 2 or len(keyword) > 12:
        return False
    if keyword in STOP_KEYWORDS:
        return False
    if any(stop_word in keyword for stop_word in STOP_KEYWORDS):
        return False
    return any(suffix in keyword for suffix in TECHNICAL_SUFFIXES)


def _node_id(label: str, name: str) -> str:
    digest = hashlib.sha1(f"{label}|{name}".encode()).hexdigest()[:12]
    return f"{label.lower()}:{digest}"


def _edge_id(source_id: str, relation: str, target_id: str) -> str:
    digest = hashlib.sha1(f"{source_id}|{relation}|{target_id}".encode()).hexdigest()[:12]
    return f"edge:{digest}"
