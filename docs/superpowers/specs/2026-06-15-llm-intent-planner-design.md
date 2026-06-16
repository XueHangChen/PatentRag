# LLM Intent Planner Design

Date: 2026-06-15
Project: Patent KG Agent
Status: Draft for review

## 1. Purpose

The current Agent runtime is already backed by LangGraph, but task understanding still starts
from deterministic keyword rules. That is stable, but brittle: user requests can be indirect,
mixed-intent, colloquial, or phrased without the exact trigger words used by the rule planner.

This phase upgrades the planning layer so the Agent can use an LLM to produce a structured
intent and tool plan, while preserving rule planning as the offline fallback.

The goal is not to make the Agent free-form. The LLM must output constrained JSON that is
validated by Pydantic. If the output is invalid, unsafe, low confidence, or uses an unknown
tool, the runtime falls back to the existing rule planner.

## 2. Scope

In scope:

- Add an LLM-backed planner for user intent and initial tool plan generation.
- Keep the existing `PatentAgentPlanner` as the deterministic fallback.
- Support the existing intents:
  - `patent_qa`
  - `patent_summary`
  - `idea_analysis`
- Support only existing tools:
  - `search_patents`
  - `summarize_patent`
  - `graph_search`
  - `graph_rag_answer`
- Add confidence, rationale, and planner mode to the planning result.
- Integrate the LLM planner into the LangGraph runtime before tool execution.
- Add tests using fake chat clients only; no default test may call DashScope or another network LLM.
- Add config switches so local/offline runs can stay rule-only.

Out of scope:

- LLM-based technical knowledge graph extraction.
- New graph entity types such as Problem, Solution, Component, and Effect.
- New Agent tools.
- Multi-turn memory.
- Streaming responses.
- Durable LangGraph checkpoint persistence.

LLM technical graph extraction should be a separate phase after the planning layer is stable.

## 3. Recommended Approach

Use a hybrid planner:

```text
user query
-> rule planner builds a safe fallback plan
-> optional LLM planner proposes structured intent and steps
-> Pydantic validates the LLM plan
-> guardrails validate tool names, step count, confidence, and safety boundaries
-> LangGraph uses the LLM plan if valid, otherwise uses the rule plan
```

This keeps the system reliable for tests and demos while improving real user understanding
when an LLM key is configured.

## 4. Planner Modes

Planner mode should be explicit in the API response:

- `rule`: deterministic rule planner was used.
- `llm`: LLM planner generated the accepted initial plan.
- `hybrid`: rule plan was generated first, and LLM planning or replanning modified it.

For this phase, `llm` means the accepted initial plan came from the LLM planner. `hybrid`
remains available for later runtime replanning after partial execution.

## 5. Data Models

Add these planning models in the Agent layer.

```python
PlannerConfidence = Literal["low", "medium", "high"]

class LlmPlanDecision(BaseModel):
    intent: AgentIntent
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: PlannerConfidence
    rationale: str
    steps: list[AgentPlannedStep]
    safety_notes: list[str] = Field(default_factory=list)
```

The final public `AgentPlan` can remain the same shape, but the runtime should store metadata
in `AgentState`, for example:

```python
planning_metadata = {
    "source": "llm",
    "confidence": 0.86,
    "confidence_label": "high",
    "fallback_used": False,
}
```

The API can continue returning `planner_mode`, `plan`, `node_trace`, and `graph_state`.
No breaking response change is required.

## 6. Prompt Contract

The LLM planner prompt must be strict and short. It should tell the model:

- Return JSON only.
- Choose exactly one known intent.
- Choose only known tools.
- Do not invent patent facts.
- Do not make legal patentability or infringement conclusions.
- Prefer evidence-gathering tools before final answer generation.
- Use `summarize_patent` only when a concrete patent id, title, or summary request is present.
- Use `graph_rag_answer` as the final answer step when an answer should be generated.

Expected JSON:

```json
{
  "intent": "idea_analysis",
  "confidence": 0.86,
  "confidence_label": "high",
  "rationale": "The user wants design inspiration and improvement directions.",
  "steps": [
    {"tool_name": "search_patents", "reason": "Find similar patent evidence."},
    {"tool_name": "graph_search", "reason": "Expand relationships for matched patents."},
    {"tool_name": "graph_rag_answer", "reason": "Generate grounded improvement suggestions."}
  ],
  "safety_notes": []
}
```

## 7. Guardrails

The LLM plan is accepted only when all conditions pass:

- JSON parses successfully.
- Pydantic validation passes.
- `intent` is one of the known intents.
- Every `tool_name` is known.
- `steps` length is between 1 and 5.
- The final step is `graph_rag_answer` unless the plan intentionally only summarizes a patent.
- `confidence >= 0.65`.
- `rationale` is non-empty.
- The plan does not include legal conclusion language.

If any guardrail fails, use the rule planner result and record the fallback reason in
`node_trace` and `graph_state`.

## 8. LangGraph Integration

The current LangGraph runtime has:

```text
initialize
-> rule_plan
-> llm_replan
-> tool nodes
-> self_check
-> finalize
```

Update it to:

```text
initialize
-> rule_plan
-> llm_plan
-> llm_replan
-> tool nodes
-> self_check
-> finalize
```

`rule_plan` always runs first and creates a fallback. `llm_plan` can replace the pending
steps when enabled and valid. `llm_replan` remains a later optional adjustment boundary.

If LLM planning is disabled or no chat client is configured, `llm_plan` records a skipped trace
and keeps the rule plan.

## 9. Configuration

Add settings:

```text
PATENT_RAG_ENABLE_LLM_PLANNER=false
PATENT_RAG_LLM_PLANNER_MIN_CONFIDENCE=0.65
```

Default should be false so local tests and demos remain deterministic unless explicitly enabled.

API requests do not need a new field in this phase. Runtime behavior should come from settings
and dependency injection. Tests can inject a fake planner directly.

## 10. Error Handling

Failure modes:

- Missing API key: skip LLM planning and keep rule plan.
- LLM provider error: keep rule plan and record trace error.
- Invalid JSON: keep rule plan and record validation failure.
- Unknown tool: keep rule plan.
- Low confidence: keep rule plan.

These are recoverable planning failures. They should not cause `/agent/run` to fail if the
rule planner can still produce a valid plan.

## 11. Testing Strategy

Tests must remain offline.

Add tests for:

- LLM planner accepts valid JSON and returns an `AgentPlan`.
- LLM planner rejects invalid JSON and falls back to rule plan.
- LLM planner rejects unknown tool names.
- LLM planner rejects low confidence.
- LangGraph runtime records `llm_plan` node trace.
- LangGraph runtime uses LLM plan when enabled and valid.
- LangGraph runtime keeps rule plan when LLM planner fails.
- API response still includes existing fields plus planner observability.

Use fake chat clients and fake planners; do not call DashScope in tests.

## 12. Success Criteria

This phase is complete when:

- `/agent/run` can use an LLM-generated structured plan when enabled.
- Rule planning remains the default fallback and offline path.
- Invalid LLM output cannot crash the Agent.
- Unknown tools cannot be executed.
- The frontend can still display plan, tool trace, node trace, and answer.
- Backend tests pass.
- Frontend build passes.

## 13. Deferred Next Phase: LLM Technical Graph Extraction

After LLM planning is stable, implement LLM graph extraction as a separate design:

```text
Patent text
-> LLM extracts Problem / Solution / Component / Effect
-> Pydantic validates entities and relations
-> evidence text and confidence are attached
-> graph store persists technical relationships
-> GraphRAG prioritizes technical evidence
```

Keeping this separate prevents the planning upgrade from being blocked by graph schema,
evidence normalization, and extraction quality issues.
