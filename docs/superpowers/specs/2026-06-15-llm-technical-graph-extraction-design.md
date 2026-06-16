# LLM Technical Graph Extraction Design

Date: 2026-06-15
Project: Patent KG Agent
Status: Draft for review

## 1. Purpose

The current knowledge graph is stable but shallow. It builds metadata relationships such as
`Patent -> Applicant`, `Patent -> Inventor`, `Patent -> IPC`, `Patent -> Section`,
`Patent -> Claim`, and rule-based `Keyword` nodes.

That is useful for first-pass GraphRAG, but it does not yet represent the technical meaning
inside a patent. The next phase adds an optional LLM extraction layer for technical semantic
entities and relations, while preserving the existing rule graph as the deterministic base.

The goal is not to replace rule extraction. The goal is to add evidence-backed technical
entities that can answer questions such as:

- What problem does this patent solve?
- What key components does this patent use?
- What solution or mechanism is proposed?
- What technical effects are claimed?
- Which patents share similar components, problems, or effects?

## 2. Current State

Existing graph extraction lives in `src/patent_rag/graph/extractor.py`.

Current node labels:

- `Patent`
- `Applicant`
- `Inventor`
- `IPC`
- `Section`
- `Claim`
- `Keyword`

Current relation types:

- `HAS_APPLICANT`
- `HAS_INVENTOR`
- `HAS_IPC`
- `HAS_SECTION`
- `HAS_CLAIM`
- `MENTIONS_KEYWORD`

Current graph storage uses `GraphNode` and `GraphEdge` from
`src/patent_rag/domain/schemas.py`, then persists a NetworkX node-link JSON file through
`src/patent_rag/graph/store.py`.

This storage shape already supports semantic expansion because both nodes and edges have:

- `properties: dict[str, str]`
- `evidence: list[EvidenceSpan]`

## 3. Scope

In scope:

- Add an optional LLM extractor for technical semantic graph facts.
- Keep rule graph extraction as the default path.
- Add technical node labels:
  - `TechnicalField`
  - `Problem`
  - `Component`
  - `Solution`
  - `Effect`
- Add technical relation types:
  - `HAS_TECHNICAL_FIELD`
  - `SOLVES_PROBLEM`
  - `USES_COMPONENT`
  - `PROPOSES_SOLUTION`
  - `HAS_EFFECT`
  - `SUPPORTED_BY_CLAIM`
- Require evidence text and confidence for every LLM-extracted entity and relation.
- Validate LLM output with Pydantic before adding it to the graph.
- Make LLM extraction disabled by default.
- Add offline tests with fake chat clients only.
- Extend `scripts/build_graph.py` with an opt-in flag for LLM technical extraction.

Out of scope:

- Neo4j migration.
- Frontend graph visualization changes.
- Similarity edge generation such as `SIMILAR_TO`.
- Human review UI.
- Long-running background job management.
- Automatically re-running extraction from the API.
- Replacing the existing `Keyword` extraction logic.

## 4. Recommended Approach

Use a hybrid graph extraction pipeline:

```text
PatentDocument list
-> rule extractor builds metadata graph
-> optional LLM technical extractor runs per patent
-> Pydantic validates extracted entities and relations
-> evidence and confidence guardrails filter unsafe output
-> validated semantic nodes and edges merge into the rule graph
-> NetworkX JSON writer persists the combined graph
```

This keeps local builds deterministic by default and lets LLM extraction be enabled only
when a model provider is configured.

## 5. Data Model

Add LLM-specific extraction models in a new graph module, not in the public domain schema.
The existing public `GraphNode` and `GraphEdge` are enough for storage.

Proposed internal models:

```python
TechnicalEntityLabel = Literal[
    "TechnicalField",
    "Problem",
    "Component",
    "Solution",
    "Effect",
]

TechnicalRelationType = Literal[
    "HAS_TECHNICAL_FIELD",
    "SOLVES_PROBLEM",
    "USES_COMPONENT",
    "PROPOSES_SOLUTION",
    "HAS_EFFECT",
    "SUPPORTED_BY_CLAIM",
]

class LlmTechnicalEntity(BaseModel):
    label: TechnicalEntityLabel
    name: str
    normalized_name: str | None = None
    evidence_text: str
    section: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

class LlmTechnicalRelation(BaseModel):
    source_name: str
    relation: TechnicalRelationType
    target_name: str
    evidence_text: str
    section: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

class LlmTechnicalGraphDecision(BaseModel):
    entities: list[LlmTechnicalEntity]
    relations: list[LlmTechnicalRelation]
```

After validation, the extractor converts these into `GraphNode` and `GraphEdge`.

Node properties should include:

- `patent_id`
- `source`
- `confidence`
- `normalized_name`

Edge properties should include:

- `patent_id`
- `source`
- `confidence`

## 6. Prompt Contract

The LLM prompt must be strict and evidence-first.

Rules:

- Return JSON only.
- Extract only facts supported by the provided patent text.
- Do not infer facts from general knowledge.
- Do not make infringement, patentability, validity, or legal conclusions.
- Use concise entity names.
- Every entity must include an exact evidence sentence or phrase from the input.
- Every relation must connect extracted entities or the patent itself to extracted entities.
- If evidence is insufficient, return empty arrays.

The prompt should provide:

- patent id
- title
- abstract
- first 2-3 claims
- selected background/summary/implementation sections, truncated to a safe length

The extractor should avoid sending the full raw patent document in this phase.

## 7. Guardrails

LLM output is accepted only when all guardrails pass:

- JSON parses successfully.
- Pydantic validation passes.
- Entity labels are known.
- Relation types are known.
- Entity names are non-empty and short enough for graph display.
- Evidence text is non-empty.
- Evidence text appears in the source context, or passes a conservative substring check after
  whitespace normalization.
- Confidence is at least the configured threshold.
- Relation endpoints can be resolved to either the patent node or extracted entity nodes.
- Output does not contain legal-conclusion language.

When validation fails for one entity or relation, drop that item and keep valid siblings.
When the whole LLM response is invalid, keep the rule graph unchanged and record an extraction
error summary.

## 8. Graph Merge Behavior

The rule extractor remains responsible for the base graph.

LLM technical extraction adds:

- `Patent -HAS_TECHNICAL_FIELD-> TechnicalField`
- `Patent -SOLVES_PROBLEM-> Problem`
- `Patent -USES_COMPONENT-> Component`
- `Patent -PROPOSES_SOLUTION-> Solution`
- `Patent -HAS_EFFECT-> Effect`
- `Claim -SUPPORTED_BY_CLAIM-> Problem | Component | Solution | Effect` when evidence comes
  from a claim and the claim number can be identified

Node IDs should be deterministic:

```text
technical:<label>:<sha1(patent_id|label|normalized_name)>
```

This keeps identical terms in different patents separate for this phase. Cross-patent entity
normalization is listed as a deferred follow-up after we have extraction quality data.

## 9. Configuration

Add settings:

```text
PATENT_RAG_ENABLE_LLM_GRAPH_EXTRACTION=false
PATENT_RAG_LLM_GRAPH_MIN_CONFIDENCE=0.65
PATENT_RAG_LLM_GRAPH_MAX_PATENTS=0
```

`PATENT_RAG_LLM_GRAPH_MAX_PATENTS=0` means no explicit cap. Tests should inject fake clients
and should not depend on these environment variables.

`scripts/build_graph.py` should support:

```powershell
python scripts\build_graph.py --llm-technical
python scripts\build_graph.py --llm-technical --llm-graph-max-patents 3
```

If LLM technical extraction is requested but the chat client cannot be created, the script
should explain the issue and continue with the rule graph. Strict failure mode is out of scope
for this phase.

## 10. Error Handling

Recoverable failures:

- Missing API key.
- LLM provider error.
- Invalid JSON.
- Unknown labels or relations.
- Evidence text not found in context.
- Confidence below threshold.

These failures should not break graph construction. The extractor should expose a small
summary object for tests and scripts, for example:

```python
{
    "attempted_patents": 20,
    "successful_patents": 18,
    "dropped_entities": 4,
    "dropped_relations": 7,
    "errors": 2,
}
```

This phase should store this summary in memory and print it from the build script. Persisting
an extraction report file is listed as a deferred follow-up.

## 11. Testing Strategy

Tests must remain offline.

Add tests for:

- The LLM technical extractor accepts valid JSON and creates technical nodes and edges.
- Low-confidence entities are dropped.
- Relations with unknown endpoints are dropped.
- Evidence not present in source context is dropped.
- Invalid JSON produces no semantic graph additions and records an error.
- `extract_graph_from_documents(..., technical_extractor=...)` merges rule and semantic graph
  results.
- `scripts/build_graph.py` remains rule-only by default.

Existing graph tests should keep passing unchanged when LLM extraction is disabled.

## 12. Success Criteria

This phase is complete when:

- The default graph build remains rule-only and deterministic.
- An injected fake LLM extractor can add `Problem`, `Component`, `Solution`, `Effect`, and
  `TechnicalField` nodes.
- Every LLM-added node and edge has evidence and confidence metadata.
- Invalid or unsafe LLM output cannot corrupt the graph.
- Backend tests pass.
- `scripts/build_graph.py` documents and supports opt-in technical extraction.

## 13. Deferred Follow-Up

After this phase, we can improve graph quality with:

- Cross-patent entity normalization and aliases.
- `SIMILAR_TO` edges based on shared components/problems/effects.
- GraphRAG ranking that prioritizes technical semantic nodes.
- Frontend display for technical entities and evidence.
- A persisted extraction quality report.
