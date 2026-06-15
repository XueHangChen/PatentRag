# LangGraph Agent Runtime Design

Date: 2026-06-15
Project: Patent KG Agent
Status: Approved for implementation planning

## 1. Purpose

The project already has a working rule-based patent Agent with tool calls, GraphRAG, FastAPI, and a React Agent workspace. The next architecture step is to replace the internal `/agent/run` execution path with a LangGraph-based runtime while keeping the public API compatible.

This design turns the current linear Agent service into a stateful graph:

- Each major Agent operation becomes a graph node.
- A single `AgentState` carries intent, plan, evidence, errors, traces, and final answer.
- A hybrid planner combines deterministic rules with optional LLM replanning.
- The response preserves existing frontend fields and adds LangGraph-specific observability fields.

The goal is not to rewrite business logic. Existing retrieval, graph, RAG, and summary tools remain the source of truth. LangGraph becomes the orchestration layer around them.

## 2. Current Baseline

Current implemented capabilities:

- Rule-based `PatentAgentPlanner`.
- Tool execution in `PatentAgentTools`.
- Agent orchestration in `PatentAgentService`.
- FastAPI endpoint `POST /agent/run`.
- React Agent workspace showing intent, plan, tool trace, answer, text sources, and graph sources.
- Unit tests for planner, service, API, retrieval, GraphRAG, graph extraction, LLM client, and vector retrieval.

Current verification status from project exploration:

- Backend tests: `51 passed`.
- Frontend build: `npm.cmd run build` passes.
- Keyword retrieval evaluation runs offline.
- Vector and hybrid evaluation may call DashScope embeddings and should remain separate from offline verification.

## 3. Scope

In scope:

- Replace the default `/agent/run` implementation with a LangGraph runtime.
- Keep response compatibility with the current `AgentRunResult`.
- Add optional response fields:
  - `planner_mode`
  - `node_trace`
  - `graph_state`
  - `checkpoint_id`
- Add `AgentState` schema for graph execution.
- Add hybrid planner flow:
  - Rule planner always runs first.
  - LLM replanner can adjust or extend the plan for complex tasks or evidence gaps.
  - Rule planner remains the fallback when LLM replanning fails or is disabled.
- Add structured node trace for interview and debugging visibility.
- Add tests that do not call external LLM services by default.
- Update README with the LangGraph Agent Runtime story and local verification commands.

Out of scope for this phase:

- MCP server exposure.
- Full frontend redesign.
- SSE or WebSocket streaming.
- Long-term durable checkpoint storage.
- LLM-as-judge evaluation.
- Full migration of RAG or graph internals.
- Replacing current tools with LangChain tool wrappers.

## 4. User-Facing API Contract

`POST /agent/run` remains the official endpoint.

Existing fields stay stable:

```text
query
intent
answer
retrieval_mode
top_k
use_graph
plan
steps
sources
graph_sources
```

New optional fields:

```text
planner_mode: "rule" | "llm" | "hybrid"
node_trace: list[AgentNodeTrace]
graph_state: dict[str, object]
checkpoint_id: str | null
```

Compatibility rules:

- Existing frontend code must continue to render even if it ignores new fields.
- Existing tests that assert `AgentRunResult` shape should continue to pass after expected extensions are added.
- New fields are additive and optional from the frontend perspective.
- API errors should still use clear HTTP failures for invalid environment or missing local artifacts, but recoverable graph-node failures should be represented in state and trace.

## 5. Architecture

Recommended graph:

```text
initialize
  -> rule_plan
  -> should_replan
       yes -> llm_replan
       no  -> route_tools
  -> execute_search
  -> execute_graph_search
  -> execute_summary
  -> execute_graph_rag
  -> self_check
       pass -> finalize
       evidence_missing -> llm_replan or supplemental retrieval
       unsafe_answer -> finalize_with_boundary
  -> finalize
```

The graph can execute only the nodes required by the plan. For example:

- Patent QA commonly runs search, optional graph search, and GraphRAG answer.
- Patent summary runs search or identifier extraction, then summary.
- Idea analysis runs search, optional graph search, and GraphRAG answer with an idea-analysis prompt.

The first implementation should keep node routing simple and deterministic. Dynamic replanning should be available but bounded.

## 6. Components

### 6.1 `langgraph_state.py`

Defines runtime state and trace structures.

Core state fields:

```text
query
intent
planner_mode
plan
pending_steps
completed_steps
sources
graph_sources
answer
errors
self_check
node_trace
checkpoint_id
metadata
```

`AgentNodeTrace` fields:

```text
node
status
input_summary
output_summary
latency_ms
error
```

The trace is concise by design. It should include enough information to explain execution without storing full raw chunk payloads.

### 6.2 `replanner.py`

Provides the hybrid planning boundary.

Responsibilities:

- Detect whether a query or current state needs LLM replanning.
- Build a constrained prompt from current state.
- Parse a structured replanning decision.
- Return one of:
  - keep current plan
  - append tool step
  - skip tool step
  - request supplemental retrieval
  - finalize with boundary explanation
- Fall back to the rule plan if the LLM planner is unavailable, malformed, or unsafe.

Initial implementation should support fake replanner injection for tests.

### 6.3 `langgraph_service.py`

Builds and runs the LangGraph workflow.

Responsibilities:

- Create initial `AgentState`.
- Compile the graph.
- Invoke nodes.
- Convert final state into the extended `AgentRunResult`.
- Preserve compatibility with existing `PatentAgentTools`.
- Add node-level trace entries.
- Generate a `checkpoint_id` for each run.

The first phase may use a run-level `checkpoint_id` without durable resume semantics. Durable checkpointer storage can be added later when the project introduces persisted sessions.

### 6.4 Existing `service.py`

The existing `PatentAgentService` should not be deleted immediately. It can provide helper logic or serve as a fallback during migration. The API default, however, should be `LangGraphPatentAgentService`.

## 7. Hybrid Planner Behavior

The rule planner is always the first decision layer.

Rule planner responsibilities:

- Intent classification.
- Initial plan generation.
- Safety constraints, including no legal infringement or patentability conclusions.
- Stable offline behavior for tests and demos.

LLM replanner responsibilities:

- Handle complex multi-part goals.
- Add missing tool steps when evidence is insufficient.
- Suggest supplemental retrieval.
- Explain why the graph should continue or finalize.

LLM replanner guardrails:

- May only choose known tool names.
- May not invent patent facts.
- May not produce final legal conclusions.
- Must preserve evidence-first behavior.
- Must fail closed to the rule plan.

Planner modes:

- `rule`: only rule planner executed.
- `hybrid`: rule planner executed, then LLM replanner made a valid change or confirmation.
- `llm`: reserved for future use. This phase does not make LLM planning the primary path.

## 8. Error Handling

Recoverable node failures should be recorded in `AgentState.errors` and `node_trace`.

Fallback rules:

- Search failure: record error and try keyword retrieval if possible.
- Graph missing or graph query failure: skip graph nodes and continue with text evidence.
- LLM replanner failure: keep rule plan and continue.
- GraphRAG answer failure: return a bounded fallback answer using available evidence.
- Evidence missing after planned steps: return an explicit knowledge-base limitation message.
- Unsafe answer self-check: finalize with a boundary statement instead of a risky answer.

Global failures should still fail clearly:

- Missing processed patent data.
- Missing chunk file for retrieval.
- Invalid API request shape.
- Missing required runtime dependency when `/agent/run` is called.

## 9. Testing Strategy

No default test should call DashScope or another external LLM service.

New tests:

```text
tests/test_langgraph_agent_service.py
tests/test_agent_replanner.py
```

Test coverage:

- LangGraph runtime returns an `AgentRunResult` compatible response.
- `node_trace` records expected node order and statuses.
- Rule-only path works without an LLM key.
- Hybrid path can be exercised with a fake replanner.
- Replanner malformed output falls back to rule plan.
- Graph missing path skips graph node and still returns an answer.
- API `/agent/run` uses the LangGraph service by default.
- Existing frontend-facing response fields remain populated.

Existing verification remains:

```powershell
.\.venv\Scripts\python.exe -m pytest
npm.cmd run build
```

## 10. Frontend Changes

Frontend changes are intentionally light.

`frontend/src/api/client.ts`:

- Extend `AgentRunResult` with optional `planner_mode`, `node_trace`, `graph_state`, and `checkpoint_id`.

`frontend/src/App.tsx`:

- Optionally show planner mode.
- Optionally show checkpoint id.
- Optionally render `node_trace` as a LangGraph execution trace.

The existing plan and tool trace UI remains valid.

## 11. Dependency Strategy

Because `/agent/run` will use LangGraph by default, LangGraph should become a first-class runtime dependency or be included in a clearly documented `agent` extra that is installed for the standard demo.

Recommended path:

- Move `langgraph` from the `llm` extra to main dependencies, or
- Add `agent = ["langgraph>=0.2.0"]` and update setup instructions to install `.[dev,agent]`.

For interview reliability, the demo environment should not require a network install during the interview.

## 12. Success Criteria

This phase is complete when:

- `POST /agent/run` is backed by LangGraph.
- Existing frontend behavior remains compatible.
- Response includes `planner_mode`, `node_trace`, `graph_state`, and `checkpoint_id`.
- Rule path runs without external LLM calls.
- Hybrid replanning can be tested with a fake replanner and used online when configuration is available.
- Recoverable node failures are represented in trace and do not crash the whole Agent run.
- Backend tests pass.
- Frontend production build passes.
- README explains the LangGraph Agent Runtime and interview talking points.

## 13. Interview Story

Suggested explanation:

> I first built a deterministic patent Agent with a rule planner and explicit tools because it is stable and testable. Then I upgraded the execution layer to LangGraph. The graph runtime carries a typed AgentState through planning, retrieval, graph expansion, answer generation, self-check, and finalization. Rule planning provides safety and offline reliability, while an optional LLM replanner can adjust the plan for complex tasks. The API remains compatible, but now returns node traces, planner mode, graph state, and checkpoint id, so the Agent is easier to debug and explain.

This shows:

- Agent workflow engineering, not only prompt engineering.
- State management and graph orchestration.
- Tool boundary discipline.
- Fallback and safety design.
- Observability through structured node traces.
- Backward-compatible API evolution.

## 14. References

- LangGraph overview: https://docs.langchain.com/oss/python/langgraph/overview
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
