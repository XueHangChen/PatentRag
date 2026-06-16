"""LLM-backed technical semantic graph extraction."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from patent_rag.domain import EvidenceSpan, GraphEdge, GraphNode, PatentDocument
from patent_rag.graph.extractor import GraphExtractionResult
from patent_rag.llm import ChatClient, ChatMessage

TechnicalEntityLabel = Literal["TechnicalField", "Problem", "Component", "Solution", "Effect"]
TechnicalRelationType = Literal[
    "HAS_TECHNICAL_FIELD",
    "SOLVES_PROBLEM",
    "USES_COMPONENT",
    "PROPOSES_SOLUTION",
    "HAS_EFFECT",
    "SUPPORTED_BY_CLAIM",
]

_DEFAULT_RELATION_BY_LABEL: dict[TechnicalEntityLabel, TechnicalRelationType] = {
    "TechnicalField": "HAS_TECHNICAL_FIELD",
    "Problem": "SOLVES_PROBLEM",
    "Component": "USES_COMPONENT",
    "Solution": "PROPOSES_SOLUTION",
    "Effect": "HAS_EFFECT",
}

_RELATION_ALIASES = {
    "SUPPORTS_CLAIM": "SUPPORTED_BY_CLAIM",
}

_LEGAL_CONCLUSION_TERMS = (
    "infringement",
    "infringe",
    "non-infringement",
    "invalidity conclusion",
    "legal conclusion",
    "patentability conclusion",
    "freedom to operate conclusion",
    "侵权",
    "不侵权",
    "法律结论",
    "专利性结论",
)


class LlmTechnicalEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: TechnicalEntityLabel
    name: str = Field(min_length=1)
    normalized_name: str | None = None
    evidence_text: str = Field(min_length=1)
    section: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class LlmTechnicalRelation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_name: str = Field(min_length=1)
    relation: TechnicalRelationType
    target_name: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)
    section: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class LlmTechnicalGraphDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: list[LlmTechnicalEntity] = Field(default_factory=list)
    relations: list[LlmTechnicalRelation] = Field(default_factory=list)


class TechnicalGraphExtractionSummary(BaseModel):
    attempted_patents: int = 0
    successful_patents: int = 0
    added_entities: int = 0
    added_relations: int = 0
    dropped_entities: int = 0
    dropped_relations: int = 0
    errors: int = 0
    error_messages: list[str] = Field(default_factory=list)


class LlmTechnicalGraphExtractor:
    """Extract evidence-backed technical graph records from patent text."""

    def __init__(self, chat_client: ChatClient, *, min_confidence: float = 0.65) -> None:
        self.chat_client = chat_client
        self.min_confidence = min_confidence
        self.summary = TechnicalGraphExtractionSummary()

    def extract(self, document: PatentDocument) -> GraphExtractionResult:
        self.summary.attempted_patents += 1
        context = _build_patent_context(document)

        try:
            raw_response = self.chat_client.generate(_build_messages(document, context))
            decision = LlmTechnicalGraphDecision.model_validate(
                _normalize_decision_payload(json.loads(raw_response))
            )
        except Exception as exc:
            self.summary.errors += 1
            self.summary.error_messages.append(str(exc))
            return GraphExtractionResult()

        result = _decision_to_graph(
            document=document,
            decision=decision,
            context=context,
            min_confidence=self.min_confidence,
            summary=self.summary,
        )
        if result.nodes or result.edges:
            self.summary.successful_patents += 1
        return result


def _decision_to_graph(
    document: PatentDocument,
    decision: LlmTechnicalGraphDecision,
    context: str,
    min_confidence: float,
    summary: TechnicalGraphExtractionSummary,
) -> GraphExtractionResult:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    entity_node_by_name: dict[str, GraphNode] = {}
    patent_id = document.metadata.patent_id
    patent_node_id = f"patent:{patent_id}"

    for entity in decision.entities:
        if not _valid_evidence(entity.evidence_text, context):
            summary.dropped_entities += 1
            continue
        if entity.confidence < min_confidence or _entity_contains_legal_conclusion(entity):
            summary.dropped_entities += 1
            continue

        node = _entity_to_node(document, entity)
        nodes.append(node)
        entity_node_by_name[_name_key(entity.name)] = node
        if entity.normalized_name:
            entity_node_by_name[_name_key(entity.normalized_name)] = node

        edges.append(
            _edge(
                source_id=patent_node_id,
                relation=_DEFAULT_RELATION_BY_LABEL[entity.label],
                target_id=node.node_id,
                evidence_text=entity.evidence_text,
                section=entity.section,
                document=document,
                confidence=entity.confidence,
            )
        )

    for relation in decision.relations:
        if relation.confidence < min_confidence:
            summary.dropped_relations += 1
            continue
        if not _valid_evidence(relation.evidence_text, context):
            summary.dropped_relations += 1
            continue
        if _contains_legal_conclusion(relation.evidence_text):
            summary.dropped_relations += 1
            continue

        source_id = _resolve_endpoint(
            relation.source_name,
            entity_node_by_name,
            patent_node_id,
            document,
            relation.relation,
        )
        target_id = _resolve_endpoint(
            relation.target_name,
            entity_node_by_name,
            patent_node_id,
            document,
            relation.relation,
        )
        if source_id is None or target_id is None:
            summary.dropped_relations += 1
            continue

        edges.append(
            _edge(
                source_id=source_id,
                relation=relation.relation,
                target_id=target_id,
                evidence_text=relation.evidence_text,
                section=relation.section,
                document=document,
                confidence=relation.confidence,
            )
        )

    nodes = _dedupe_nodes_by_id(nodes)
    edges = _dedupe_edges_by_id(edges)

    summary.added_entities += len(nodes)
    summary.added_relations += len(edges)
    return GraphExtractionResult(nodes=nodes, edges=edges)


def _dedupe_nodes_by_id(nodes: list[GraphNode]) -> list[GraphNode]:
    unique_nodes: dict[str, GraphNode] = {}
    for node in nodes:
        unique_nodes.setdefault(node.node_id, node)
    return list(unique_nodes.values())


def _dedupe_edges_by_id(edges: list[GraphEdge]) -> list[GraphEdge]:
    unique_edges: dict[str, GraphEdge] = {}
    for edge in edges:
        unique_edges.setdefault(edge.edge_id, edge)
    return list(unique_edges.values())


def _normalize_decision_payload(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)
    normalized["entities"] = [
        _normalize_entity_payload(entity) for entity in _as_list(payload.get("entities"))
    ]
    normalized["relations"] = [
        _normalize_relation_payload(relation) for relation in _as_list(payload.get("relations"))
    ]
    return normalized


def _normalize_entity_payload(entity: object) -> object:
    if not isinstance(entity, dict):
        return entity

    normalized = dict(entity)
    if "name" not in normalized:
        normalized["name"] = normalized.get("text") or normalized.get("entity")
    if "evidence_text" not in normalized:
        normalized["evidence_text"] = normalized.get("evidence") or normalized.get("text")
    if "confidence" not in normalized:
        normalized["confidence"] = 0.8
    return normalized


def _normalize_relation_payload(relation: object) -> object:
    if not isinstance(relation, dict):
        return relation

    normalized = dict(relation)
    if "source_name" not in normalized:
        normalized["source_name"] = normalized.get("source") or normalized.get("from")
    if "target_name" not in normalized:
        normalized["target_name"] = normalized.get("target") or normalized.get("to")
    if "relation" not in normalized:
        normalized["relation"] = normalized.get("type") or normalized.get("label")
    if isinstance(normalized.get("relation"), str):
        normalized["relation"] = _RELATION_ALIASES.get(
            normalized["relation"], normalized["relation"]
        )
    if "evidence_text" not in normalized:
        normalized["evidence_text"] = normalized.get("evidence") or normalized.get("text")
    if "confidence" not in normalized:
        normalized["confidence"] = 0.8
    return normalized


def _as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _build_messages(document: PatentDocument, context: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You extract technical semantic graph facts from patent text. "
                "Return strict JSON only. Do not wrap it in markdown. "
                "Use exactly this schema: "
                '{"entities":[{"label":"Component","name":"component name",'
                '"normalized_name":"component name","evidence_text":"exact text from input",'
                '"section":"section name","confidence":0.9}],'
                '"relations":[{"source_name":"source entity or claim 1",'
                '"relation":"USES_COMPONENT","target_name":"target entity",'
                '"evidence_text":"exact text from input","section":"section name",'
                '"confidence":0.9}]}. '
                "Extract at most 8 entities and at most 8 relations per patent. "
                "Allowed entity labels: TechnicalField, Problem, Component, Solution, Effect. "
                "Allowed relations: HAS_TECHNICAL_FIELD, SOLVES_PROBLEM, USES_COMPONENT, "
                "PROPOSES_SOLUTION, HAS_EFFECT, SUPPORTED_BY_CLAIM. "
                "Use confidence as a number from 0 to 1. "
                "Every entity and relation must include exact evidence_text from the input. "
                "Do not infer unsupported facts or legal conclusions."
            ),
        ),
        ChatMessage(
            role="user",
            content=json.dumps(
                {
                    "patent_id": document.metadata.patent_id,
                    "title": document.metadata.title,
                    "context": context,
                },
                ensure_ascii=False,
            ),
        ),
    ]


def _build_patent_context(document: PatentDocument, *, max_chars: int = 6000) -> str:
    parts = [document.metadata.title or "", document.abstract or ""]
    parts.extend(claim.text for claim in document.claims[:3])
    parts.extend(section.text for section in document.sections[:3])
    return "\n".join(part.strip() for part in parts if part and part.strip())[:max_chars]


def _entity_to_node(document: PatentDocument, entity: LlmTechnicalEntity) -> GraphNode:
    normalized_name = _normalized_name(entity)
    return GraphNode(
        node_id=_technical_node_id(document.metadata.patent_id, entity.label, normalized_name),
        label=entity.label,
        name=entity.name.strip(),
        properties={
            "patent_id": document.metadata.patent_id,
            "source": "llm",
            "confidence": str(entity.confidence),
            "normalized_name": normalized_name,
        },
        evidence=[_evidence(document, entity.evidence_text, entity.section)],
    )


def _edge(
    source_id: str,
    relation: str,
    target_id: str,
    evidence_text: str,
    section: str | None,
    document: PatentDocument,
    confidence: float,
) -> GraphEdge:
    return GraphEdge(
        edge_id=_edge_id(source_id, relation, target_id),
        source_id=source_id,
        target_id=target_id,
        relation=relation,
        properties={
            "patent_id": document.metadata.patent_id,
            "source": "llm",
            "confidence": str(confidence),
        },
        evidence=[_evidence(document, evidence_text, section)],
    )


def _evidence(document: PatentDocument, text: str, section: str | None) -> EvidenceSpan:
    return EvidenceSpan(
        source_file=document.metadata.source_file,
        section=section,
        text=text,
    )


def _technical_node_id(patent_id: str, label: str, normalized_name: str) -> str:
    digest = hashlib.sha1(f"{patent_id}|{label}|{normalized_name}".encode()).hexdigest()[:12]
    return f"technical:{label.lower()}:{digest}"


def _claim_node_id(patent_id: str, claim_number: int) -> str:
    digest = hashlib.sha1(f"Claim|{patent_id}:{claim_number}".encode()).hexdigest()[:12]
    return f"claim:{digest}"


def _edge_id(source_id: str, relation: str, target_id: str) -> str:
    digest = hashlib.sha1(f"{source_id}|{relation}|{target_id}".encode()).hexdigest()[:12]
    return f"edge:{digest}"


def _normalized_name(entity: LlmTechnicalEntity) -> str:
    return (entity.normalized_name or entity.name).strip().casefold()


def _name_key(name: str) -> str:
    return re.sub(r"\s+", "", name).casefold()


def _resolve_endpoint(
    name: str,
    entity_node_by_name: dict[str, GraphNode],
    patent_node_id: str,
    document: PatentDocument,
    relation: TechnicalRelationType,
) -> str | None:
    key = _name_key(name)
    if key in {"patent", "thispatent", patent_node_id.casefold()}:
        return patent_node_id
    node = entity_node_by_name.get(key)
    if node:
        return node.node_id
    if relation == "SUPPORTED_BY_CLAIM":
        return _resolve_claim_endpoint(name, document)
    return None


def _resolve_claim_endpoint(name: str, document: PatentDocument) -> str | None:
    claim_number = _claim_number_from_endpoint(name)
    if claim_number is None:
        return None
    if claim_number not in {claim.claim_number for claim in document.claims}:
        return None
    return _claim_node_id(document.metadata.patent_id, claim_number)


def _claim_number_from_endpoint(name: str) -> int | None:
    value = name.strip()
    match = re.fullmatch(r"claims?\s*[:#]?\s*(\d+)", value, flags=re.IGNORECASE)
    if not match:
        match = re.fullmatch(r"权利要求\s*[:：]?\s*(\d+)", value)
    if not match:
        match = re.fullmatch(r"\d+", value)
    return int(match.group(1) if match.lastindex else match.group(0)) if match else None


def _valid_evidence(evidence_text: str, context: str) -> bool:
    return _compact(evidence_text) in _compact(context)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _contains_legal_conclusion(value: str) -> bool:
    normalized = value.casefold()
    return any(term.casefold() in normalized for term in _LEGAL_CONCLUSION_TERMS)


def _entity_contains_legal_conclusion(entity: LlmTechnicalEntity) -> bool:
    return any(
        _contains_legal_conclusion(value)
        for value in (entity.evidence_text, entity.name, entity.normalized_name)
        if value
    )
