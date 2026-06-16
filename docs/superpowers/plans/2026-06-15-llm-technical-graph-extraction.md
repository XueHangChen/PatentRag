# LLM Technical Graph Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional LLM-backed technical semantic graph extraction while keeping the current rule graph as the deterministic default.

**Architecture:** The existing graph extractor continues to build metadata and keyword graph records. A new `LlmTechnicalGraphExtractor` parses strict JSON from a chat client, validates evidence-backed technical entities and relations, then returns `GraphExtractionResult` records that can be merged into the rule graph. The build script exposes opt-in LLM extraction and falls back to rule-only graph construction when the LLM client is unavailable.

**Tech Stack:** Python 3.11+, Pydantic v2, NetworkX, pytest, existing `ChatClient` protocol, existing `GraphNode` / `GraphEdge` domain schemas.

---

## File Map

- Create `src/patent_rag/graph/llm_extractor.py`  
  Defines LLM graph extraction models, prompt construction, validation, evidence filtering, graph conversion, and extraction summary.
- Modify `src/patent_rag/graph/extractor.py`  
  Adds an optional `technical_extractor` integration point and safe merge helpers.
- Modify `src/patent_rag/graph/__init__.py`  
  Exports the new LLM extraction types.
- Modify `src/patent_rag/config.py`  
  Adds optional graph extraction settings.
- Modify `scripts/build_graph.py`  
  Adds `--llm-technical`, `--llm-graph-min-confidence`, and `--llm-graph-max-patents`.
- Modify `README.md`  
  Documents the optional LLM technical graph extraction path.
- Create `tests/test_llm_graph_extractor.py`  
  Offline tests for valid extraction and guardrails.
- Modify `tests/test_graph_extractor.py`  
  Tests rule graph plus injected technical graph merge.
- Create `tests/test_build_graph_script.py`  
  Tests CLI parser defaults and opt-in flags.

## Task 1: Add LLM Technical Extractor Success Path

**Files:**
- Create: `tests/test_llm_graph_extractor.py`
- Create: `src/patent_rag/graph/llm_extractor.py`

- [ ] **Step 1: Write the failing success-path test**

Create `tests/test_llm_graph_extractor.py`:

```python
import json
from pathlib import Path

from patent_rag.domain import PatentClaim, PatentDocument, PatentMetadata, PatentSection, PatentType
from patent_rag.graph.llm_extractor import LlmTechnicalGraphExtractor
from patent_rag.llm import ChatMessage


class FakeChatClient:
    model_name = "fake-graph-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[ChatMessage] = []

    def generate(self, messages: list[ChatMessage]) -> str:
        self.messages = messages
        return self.response


def make_document() -> PatentDocument:
    return PatentDocument(
        metadata=PatentMetadata(
            patent_id="CN000001U",
            title="Sample transport fixing frame",
            patent_type=PatentType.UTILITY_MODEL,
            applicants=["Example Hospital"],
            inventors=["Alice"],
            ipc_classes=["A61B10/00"],
            source_file=Path("patant/example.md"),
        ),
        abstract=(
            "Liquid nitrogen tanks can tip during transport. "
            "The fixing frame limits tank shaking. "
            "The fixing frame includes a base and a limiting ring. "
            "This improves transport stability."
        ),
        claims=[
            PatentClaim(
                claim_number=1,
                text="A fixing frame comprising a base and a limiting ring.",
            )
        ],
        sections=[
            PatentSection(
                name="abstract",
                text=(
                    "Liquid nitrogen tanks can tip during transport. "
                    "The fixing frame limits tank shaking. "
                    "The fixing frame includes a base and a limiting ring. "
                    "This improves transport stability."
                ),
            )
        ],
    )


def test_llm_technical_graph_extractor_creates_evidence_backed_nodes_and_edges() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Problem",
                        "name": "tank tipping during transport",
                        "evidence_text": "Liquid nitrogen tanks can tip during transport.",
                        "section": "abstract",
                        "confidence": 0.92,
                    },
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    },
                    {
                        "label": "Solution",
                        "name": "fixing frame limits tank shaking",
                        "evidence_text": "The fixing frame limits tank shaking.",
                        "section": "abstract",
                        "confidence": 0.88,
                    },
                    {
                        "label": "Effect",
                        "name": "improves transport stability",
                        "evidence_text": "This improves transport stability.",
                        "section": "abstract",
                        "confidence": 0.86,
                    },
                ],
                "relations": [
                    {
                        "source_name": "fixing frame limits tank shaking",
                        "relation": "USES_COMPONENT",
                        "target_name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.84,
                    }
                ],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    labels = {node.label for node in result.nodes}
    relations = {edge.relation for edge in result.edges}

    assert {"Problem", "Component", "Solution", "Effect"}.issubset(labels)
    assert "SOLVES_PROBLEM" in relations
    assert "USES_COMPONENT" in relations
    component = next(node for node in result.nodes if node.label == "Component")
    assert component.properties["patent_id"] == "CN000001U"
    assert component.properties["source"] == "llm"
    assert component.properties["confidence"] == "0.9"
    assert component.evidence[0].text == "The fixing frame includes a base and a limiting ring."
    assert extractor.summary.attempted_patents == 1
    assert extractor.summary.successful_patents == 1
    assert client.messages[0].role == "system"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_graph_extractor.py::test_llm_technical_graph_extractor_creates_evidence_backed_nodes_and_edges -q
```

Expected:

```text
ModuleNotFoundError: No module named 'patent_rag.graph.llm_extractor'
```

- [ ] **Step 3: Implement minimal extractor models and success path**

Create `src/patent_rag/graph/llm_extractor.py` with these public types and helpers:

```python
"""LLM-backed technical semantic graph extraction."""

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

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

DEFAULT_RELATION_BY_LABEL: dict[TechnicalEntityLabel, TechnicalRelationType] = {
    "TechnicalField": "HAS_TECHNICAL_FIELD",
    "Problem": "SOLVES_PROBLEM",
    "Component": "USES_COMPONENT",
    "Solution": "PROPOSES_SOLUTION",
    "Effect": "HAS_EFFECT",
}

LEGAL_CONCLUSION_TERMS = (
    "infringement",
    "infringe",
    "non-infringement",
    "invalidity conclusion",
    "legal conclusion",
    "patentability conclusion",
    "freedom to operate conclusion",
    "\u4fb5\u6743",
    "\u4e0d\u4fb5\u6743",
    "\u6cd5\u5f8b\u7ed3\u8bba",
    "\u4e13\u5229\u6027\u7ed3\u8bba",
)


class LlmTechnicalEntity(BaseModel):
    label: TechnicalEntityLabel
    name: str = Field(min_length=1, max_length=80)
    normalized_name: str | None = None
    evidence_text: str = Field(min_length=1)
    section: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class LlmTechnicalRelation(BaseModel):
    source_name: str = Field(min_length=1, max_length=120)
    relation: TechnicalRelationType
    target_name: str = Field(min_length=1, max_length=120)
    evidence_text: str = Field(min_length=1)
    section: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class LlmTechnicalGraphDecision(BaseModel):
    entities: list[LlmTechnicalEntity] = Field(default_factory=list, max_length=20)
    relations: list[LlmTechnicalRelation] = Field(default_factory=list, max_length=30)


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
    def __init__(self, chat_client: ChatClient, *, min_confidence: float = 0.65) -> None:
        self.chat_client = chat_client
        self.min_confidence = min_confidence
        self.summary = TechnicalGraphExtractionSummary()

    def extract(self, document: PatentDocument) -> GraphExtractionResult:
        self.summary.attempted_patents += 1
        context = _build_patent_context(document)
        try:
            raw_response = self.chat_client.generate(_build_messages(document, context))
            decision = LlmTechnicalGraphDecision.model_validate(json.loads(raw_response))
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            self.summary.errors += 1
            self.summary.error_messages.append(str(exc))
            return GraphExtractionResult()

        result = _decision_to_graph(document, decision, context, self.min_confidence, self.summary)
        if result.nodes or result.edges:
            self.summary.successful_patents += 1
        return result
```

Add conversion helpers in the same file:

```python
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
        if entity.confidence < min_confidence or _contains_legal_conclusion(entity.evidence_text):
            summary.dropped_entities += 1
            continue
        node = _entity_to_node(document, entity)
        nodes.append(node)
        entity_node_by_name[_name_key(entity.name)] = node
        if entity.normalized_name:
            entity_node_by_name[_name_key(entity.normalized_name)] = node
        edges.append(
            _edge(
                patent_node_id,
                DEFAULT_RELATION_BY_LABEL[entity.label],
                node.node_id,
                entity.evidence_text,
                entity.section,
                document,
                entity.confidence,
            )
        )

    for relation in decision.relations:
        if relation.confidence < min_confidence or not _valid_evidence(relation.evidence_text, context):
            summary.dropped_relations += 1
            continue
        if _contains_legal_conclusion(relation.evidence_text):
            summary.dropped_relations += 1
            continue
        source_id = _resolve_endpoint(relation.source_name, entity_node_by_name, patent_node_id)
        target_id = _resolve_endpoint(relation.target_name, entity_node_by_name, patent_node_id)
        if source_id is None or target_id is None:
            summary.dropped_relations += 1
            continue
        edges.append(
            _edge(
                source_id,
                relation.relation,
                target_id,
                relation.evidence_text,
                relation.section,
                document,
                relation.confidence,
            )
        )

    summary.added_entities += len(nodes)
    summary.added_relations += len(edges)
    return GraphExtractionResult(nodes=nodes, edges=edges)
```

Add utility helpers:

```python
def _build_messages(document: PatentDocument, context: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You extract technical semantic graph facts from patent text. "
                "Return strict JSON only with keys entities and relations. "
                "Allowed entity labels: TechnicalField, Problem, Component, Solution, Effect. "
                "Allowed relations: HAS_TECHNICAL_FIELD, SOLVES_PROBLEM, USES_COMPONENT, "
                "PROPOSES_SOLUTION, HAS_EFFECT, SUPPORTED_BY_CLAIM. "
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
        text=text.strip(),
    )


def _technical_node_id(patent_id: str, label: str, normalized_name: str) -> str:
    digest = hashlib.sha1(f"{patent_id}|{label}|{normalized_name}".encode()).hexdigest()[:12]
    return f"technical:{label.lower()}:{digest}"


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
) -> str | None:
    key = _name_key(name)
    if key in {"patent", "thispatent", patent_node_id.casefold()}:
        return patent_node_id
    node = entity_node_by_name.get(key)
    return node.node_id if node else None


def _valid_evidence(evidence_text: str, context: str) -> bool:
    return _compact(evidence_text) in _compact(context)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _contains_legal_conclusion(value: str) -> bool:
    normalized = value.casefold()
    return any(term.casefold() in normalized for term in LEGAL_CONCLUSION_TERMS)
```

- [ ] **Step 4: Run the success-path test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_graph_extractor.py::test_llm_technical_graph_extractor_creates_evidence_backed_nodes_and_edges -q
```

Expected:

```text
1 passed
```

## Task 2: Add Guardrail Tests for LLM Technical Extraction

**Files:**
- Modify: `tests/test_llm_graph_extractor.py`
- Modify: `src/patent_rag/graph/llm_extractor.py`

- [ ] **Step 1: Add failing guardrail tests**

Append to `tests/test_llm_graph_extractor.py`:

```python
def test_llm_technical_graph_extractor_drops_low_confidence_entities() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.2,
                    }
                ],
                "relations": [],
            }
        )
    )

    result = LlmTechnicalGraphExtractor(client, min_confidence=0.65).extract(make_document())

    assert result.nodes == []
    assert result.edges == []


def test_llm_technical_graph_extractor_drops_missing_evidence() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Effect",
                        "name": "reduces manufacturing cost",
                        "evidence_text": "This reduces manufacturing cost.",
                        "section": "abstract",
                        "confidence": 0.9,
                    }
                ],
                "relations": [],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    assert result.nodes == []
    assert extractor.summary.dropped_entities == 1


def test_llm_technical_graph_extractor_drops_unknown_relation_endpoint() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    }
                ],
                "relations": [
                    {
                        "source_name": "unknown solution",
                        "relation": "USES_COMPONENT",
                        "target_name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    }
                ],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    assert any(node.label == "Component" for node in result.nodes)
    assert all(edge.source_id != "unknown solution" for edge in result.edges)
    assert extractor.summary.dropped_relations == 1


def test_llm_technical_graph_extractor_returns_empty_on_invalid_json() -> None:
    extractor = LlmTechnicalGraphExtractor(FakeChatClient("not-json"), min_confidence=0.65)

    result = extractor.extract(make_document())

    assert result.nodes == []
    assert result.edges == []
    assert extractor.summary.errors == 1


def test_llm_technical_graph_extractor_drops_legal_conclusion_language() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Effect",
                        "name": "non infringement conclusion",
                        "evidence_text": "This is a non-infringement conclusion.",
                        "section": "abstract",
                        "confidence": 0.9,
                    }
                ],
                "relations": [],
            }
        )
    )
    document = make_document()
    document.abstract = f"{document.abstract} This is a non-infringement conclusion."

    result = LlmTechnicalGraphExtractor(client, min_confidence=0.65).extract(document)

    assert result.nodes == []
    assert result.edges == []
```

- [ ] **Step 2: Run guardrail tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_graph_extractor.py -q
```

Expected:

```text
6 passed
```

- [ ] **Step 3: Adjust implementation only for actual failures**

If the unknown endpoint test fails because default patent-to-entity edges are counted, keep
the default edge and assert only the dropped relation count:

```python
assert extractor.summary.dropped_relations == 1
```

Do not remove default patent-to-entity edges; they are required by the graph merge design.

## Task 3: Merge Optional Technical Graph Results into Rule Graph Extraction

**Files:**
- Modify: `tests/test_graph_extractor.py`
- Modify: `src/patent_rag/graph/extractor.py`

- [ ] **Step 1: Write failing merge test**

Append to `tests/test_graph_extractor.py`:

```python
from patent_rag.domain import EvidenceSpan, GraphEdge, GraphNode
from patent_rag.graph import GraphExtractionResult


class FakeTechnicalExtractor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract(self, document):
        self.calls.append(document.metadata.patent_id)
        problem_node = GraphNode(
            node_id=f"technical:problem:{document.metadata.patent_id}",
            label="Problem",
            name="transport instability",
            properties={
                "patent_id": document.metadata.patent_id,
                "source": "llm",
                "confidence": "0.9",
            },
            evidence=[
                EvidenceSpan(
                    source_file=document.metadata.source_file,
                    section="abstract",
                    text="transport instability",
                )
            ],
        )
        return GraphExtractionResult(
            nodes=[problem_node],
            edges=[
                GraphEdge(
                    edge_id=f"edge:problem:{document.metadata.patent_id}",
                    source_id=f"patent:{document.metadata.patent_id}",
                    target_id=problem_node.node_id,
                    relation="SOLVES_PROBLEM",
                    properties={
                        "patent_id": document.metadata.patent_id,
                        "source": "llm",
                        "confidence": "0.9",
                    },
                    evidence=[
                        EvidenceSpan(
                            source_file=document.metadata.source_file,
                            section="abstract",
                            text="transport instability",
                        )
                    ],
                )
            ],
        )


def test_extract_graph_merges_optional_technical_extractor_results() -> None:
    documents, _ = parse_patent_directory(Path("patant"))
    extractor = FakeTechnicalExtractor()

    result = extract_graph_from_documents(
        documents[:1],
        technical_extractor=extractor,
    )

    assert extractor.calls == [documents[0].metadata.patent_id]
    assert any(node.label == "Problem" for node in result.nodes)
    assert any(edge.relation == "SOLVES_PROBLEM" for edge in result.edges)
    assert any(node.label == "Patent" for node in result.nodes)
```

- [ ] **Step 2: Run merge test to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_graph_extractor.py::test_extract_graph_merges_optional_technical_extractor_results -q
```

Expected:

```text
TypeError: extract_graph_from_documents() got an unexpected keyword argument 'technical_extractor'
```

- [ ] **Step 3: Add optional extractor protocol and merge helpers**

Modify `src/patent_rag/graph/extractor.py`.

Add imports:

```python
from typing import Protocol
```

Add protocol after `GraphExtractionResult`:

```python
class TechnicalGraphExtractor(Protocol):
    """Optional provider of LLM-extracted technical graph records."""

    def extract(self, document: PatentDocument) -> GraphExtractionResult:
        """Extract technical semantic graph records for one patent."""
```

Change the function signature:

```python
def extract_graph_from_documents(
    documents: list[PatentDocument],
    *,
    keyword_limit_per_patent: int = 8,
    technical_extractor: TechnicalGraphExtractor | None = None,
    technical_max_patents: int | None = None,
) -> GraphExtractionResult:
```

At the end of each document loop, after keyword edges are added, insert:

```python
        if technical_extractor is not None and _within_technical_limit(
            technical_patent_count,
            technical_max_patents,
        ):
            technical_result = technical_extractor.extract(document)
            technical_patent_count += 1
            for node in technical_result.nodes:
                _add_node(nodes_by_id, node)
            for edge in technical_result.edges:
                _add_edge_object(edges_by_id, edge)
```

Initialize before the loop:

```python
    technical_patent_count = 0
```

Add helpers near `_add_edge`:

```python
def _add_edge_object(edges_by_id: dict[str, GraphEdge], edge: GraphEdge) -> None:
    if edge.edge_id not in edges_by_id:
        edges_by_id[edge.edge_id] = edge


def _within_technical_limit(current_count: int, max_patents: int | None) -> bool:
    return max_patents is None or max_patents <= 0 or current_count < max_patents
```

- [ ] **Step 4: Run graph extractor tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_graph_extractor.py -q
```

Expected:

```text
4 passed
```

## Task 4: Export LLM Graph Extractor Types

**Files:**
- Modify: `src/patent_rag/graph/__init__.py`
- Test: `tests/test_llm_graph_extractor.py`

- [ ] **Step 1: Add import smoke test**

Append to `tests/test_llm_graph_extractor.py`:

```python
def test_llm_graph_extractor_exports_public_types() -> None:
    from patent_rag.graph import LlmTechnicalGraphExtractor, TechnicalGraphExtractionSummary

    assert LlmTechnicalGraphExtractor.__name__ == "LlmTechnicalGraphExtractor"
    assert TechnicalGraphExtractionSummary().attempted_patents == 0
```

- [ ] **Step 2: Run export test to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_graph_extractor.py::test_llm_graph_extractor_exports_public_types -q
```

Expected:

```text
ImportError
```

- [ ] **Step 3: Export new types**

Modify `src/patent_rag/graph/__init__.py`:

```python
from patent_rag.graph.llm_extractor import (
    LlmTechnicalEntity,
    LlmTechnicalGraphDecision,
    LlmTechnicalGraphExtractor,
    LlmTechnicalRelation,
    TechnicalGraphExtractionSummary,
)
```

Add to `__all__`:

```python
    "LlmTechnicalEntity",
    "LlmTechnicalGraphDecision",
    "LlmTechnicalGraphExtractor",
    "LlmTechnicalRelation",
    "TechnicalGraphExtractionSummary",
```

- [ ] **Step 4: Run export and extractor tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_graph_extractor.py -q
```

Expected:

```text
7 passed
```

## Task 5: Add Settings and Build Script Opt-In

**Files:**
- Modify: `src/patent_rag/config.py`
- Modify: `scripts/build_graph.py`
- Create: `tests/test_build_graph_script.py`

- [ ] **Step 1: Write failing settings and parser tests**

Create `tests/test_build_graph_script.py`:

```python
from patent_rag.config import Settings
from scripts.build_graph import build_arg_parser


def test_llm_graph_settings_default_to_disabled(monkeypatch) -> None:
    monkeypatch.delenv("PATENT_RAG_ENABLE_LLM_GRAPH_EXTRACTION", raising=False)
    monkeypatch.delenv("PATENT_RAG_LLM_GRAPH_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("PATENT_RAG_LLM_GRAPH_MAX_PATENTS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.enable_llm_graph_extraction is False
    assert settings.llm_graph_min_confidence == 0.65
    assert settings.llm_graph_max_patents == 0


def test_build_graph_parser_defaults_to_rule_only() -> None:
    args = build_arg_parser().parse_args([])

    assert args.llm_technical is False
    assert args.llm_graph_min_confidence is None
    assert args.llm_graph_max_patents is None


def test_build_graph_parser_accepts_llm_technical_options() -> None:
    args = build_arg_parser().parse_args(
        [
            "--llm-technical",
            "--llm-graph-min-confidence",
            "0.72",
            "--llm-graph-max-patents",
            "3",
        ]
    )

    assert args.llm_technical is True
    assert args.llm_graph_min_confidence == 0.72
    assert args.llm_graph_max_patents == 3
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_build_graph_script.py -q
```

Expected:

```text
ImportError: cannot import name 'build_arg_parser'
```

or:

```text
AttributeError: 'Settings' object has no attribute 'enable_llm_graph_extraction'
```

- [ ] **Step 3: Add settings**

Modify `src/patent_rag/config.py` near existing LLM settings:

```python
    enable_llm_graph_extraction: bool = False
    llm_graph_min_confidence: float = 0.65
    llm_graph_max_patents: int = 0
```

- [ ] **Step 4: Refactor build script parser**

Modify `scripts/build_graph.py`.

Add imports:

```python
from patent_rag.config import get_settings  # noqa: E402
from patent_rag.graph import LlmTechnicalGraphExtractor  # noqa: E402
from patent_rag.llm import create_chat_client  # noqa: E402
```

Add parser function:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build patent knowledge graph.")
    parser.add_argument("--input", type=Path, default=Path("data") / "processed" / "patents.jsonl")
    parser.add_argument("--output", type=Path, default=Path("data") / "graph" / "patent_graph.json")
    parser.add_argument("--keyword-limit", type=int, default=8)
    parser.add_argument("--llm-technical", action="store_true")
    parser.add_argument("--llm-graph-min-confidence", type=float, default=None)
    parser.add_argument("--llm-graph-max-patents", type=int, default=None)
    return parser
```

Replace the parser construction in `main()` with:

```python
    parser = build_arg_parser()
    args = parser.parse_args()
```

- [ ] **Step 5: Wire optional extractor into script**

Inside `main()`, after `args = parser.parse_args()`:

```python
    settings = get_settings()
    enable_llm_technical = args.llm_technical or settings.enable_llm_graph_extraction
    min_confidence = (
        args.llm_graph_min_confidence
        if args.llm_graph_min_confidence is not None
        else settings.llm_graph_min_confidence
    )
    max_patents = (
        args.llm_graph_max_patents
        if args.llm_graph_max_patents is not None
        else settings.llm_graph_max_patents
    )
    technical_extractor = None
    if enable_llm_technical:
        try:
            technical_extractor = LlmTechnicalGraphExtractor(
                create_chat_client(temperature=0.0),
                min_confidence=min_confidence,
            )
        except Exception as exc:
            print(f"llm_technical_extraction_skipped={exc}")
```

Replace extraction call:

```python
    result = extract_graph_from_documents(
        documents,
        keyword_limit_per_patent=args.keyword_limit,
        technical_extractor=technical_extractor,
        technical_max_patents=max_patents,
    )
```

After printing stats:

```python
    if technical_extractor is not None:
        print(
            "technical_extraction="
            + json.dumps(
                technical_extractor.summary.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
```

- [ ] **Step 6: Run script tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_build_graph_script.py -q
```

Expected:

```text
3 passed
```

## Task 6: Document Optional LLM Technical Graph Extraction

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README section**

Add this near the existing graph build section:

```markdown
LLM technical graph extraction is optional. The default graph build remains rule-based and offline:

```powershell
python scripts\build_graph.py
```

To add evidence-backed technical semantic nodes such as `Problem`, `Component`, `Solution`, `Effect`, and `TechnicalField`, configure an LLM provider and run:

```dotenv
PATENT_RAG_DASHSCOPE_API_KEY=your-dashscope-api-key
PATENT_RAG_ENABLE_LLM_GRAPH_EXTRACTION=true
PATENT_RAG_LLM_GRAPH_MIN_CONFIDENCE=0.65
```

```powershell
python scripts\build_graph.py --llm-technical --llm-graph-max-patents 3
```

Invalid, low-confidence, unsupported, or unavailable LLM output is skipped. Rule graph extraction still completes.
```

- [ ] **Step 2: Check README renders as Markdown**

Run:

```powershell
Select-String -Path README.md -Pattern "LLM technical graph extraction|llm-technical|PATENT_RAG_ENABLE_LLM_GRAPH_EXTRACTION"
```

Expected: all three patterns are present.

## Task 7: Focused and Full Verification

**Files:**
- All changed files.

- [ ] **Step 1: Run focused graph tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_graph_extractor.py tests\test_graph_extractor.py tests\test_build_graph_script.py -q
```

Expected:

```text
passed
```

- [ ] **Step 2: Run existing RAG graph context tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rag_graph_context.py -q
```

Expected:

```text
passed
```

- [ ] **Step 3: Run full backend test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected:

```text
passed
```

- [ ] **Step 4: Run compile smoke**

Run:

```powershell
.\.venv\Scripts\python.exe -m compileall src scripts
```

Expected:

```text
Listing 'src'...
Listing 'scripts'...
```

- [ ] **Step 5: Run import smoke**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from patent_rag.graph import LlmTechnicalGraphExtractor, extract_graph_from_documents; print(LlmTechnicalGraphExtractor.__name__, extract_graph_from_documents.__name__)"
```

Expected:

```text
LlmTechnicalGraphExtractor extract_graph_from_documents
```

- [ ] **Step 6: Run targeted ruff**

Run:

```powershell
.\.venv\Scripts\ruff.exe check src\patent_rag\graph\llm_extractor.py src\patent_rag\graph\extractor.py src\patent_rag\graph\__init__.py src\patent_rag\config.py scripts\build_graph.py tests\test_llm_graph_extractor.py tests\test_graph_extractor.py tests\test_build_graph_script.py
```

Expected:

```text
All checks passed!
```

## Self-Review Checklist

- [ ] Spec requirement: rule graph remains default and deterministic.
- [ ] Spec requirement: LLM extraction is optional and disabled by default.
- [ ] Spec requirement: technical labels include `TechnicalField`, `Problem`, `Component`, `Solution`, and `Effect`.
- [ ] Spec requirement: technical relation types include `HAS_TECHNICAL_FIELD`, `SOLVES_PROBLEM`, `USES_COMPONENT`, `PROPOSES_SOLUTION`, `HAS_EFFECT`, and `SUPPORTED_BY_CLAIM`.
- [ ] Spec requirement: every accepted LLM node and edge has evidence and confidence metadata.
- [ ] Spec requirement: invalid JSON and unsafe output do not break graph construction.
- [ ] Spec requirement: tests are offline and use fake chat clients.
- [ ] Spec requirement: `scripts/build_graph.py` exposes opt-in LLM technical extraction.
- [ ] Type consistency: `LlmTechnicalGraphExtractor`, `LlmTechnicalGraphDecision`, and `TechnicalGraphExtractionSummary` names match across tasks.
- [ ] Completeness scan: no unresolved or vague implementation instructions remain.
