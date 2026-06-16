# LangGraph Agent Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the internal `/agent/run` execution path with a LangGraph-backed runtime while preserving existing response fields and adding planner/runtime observability.
**Architecture:** Keep the existing rule planner, Agent tools, retrieval, GraphRAG, FastAPI route, and React workspace. Add a typed LangGraph state graph around them. The legacy linear `PatentAgentService` remains available for comparison and rollback, while `run_patent_agent()` becomes the compatibility entrypoint for the LangGraph service.
**Tech Stack:** Python 3.11+, Pydantic v2, FastAPI, LangGraph, pytest, React, Vite, TypeScript.

---

## Implementation Contract

- Public endpoint stays `POST /agent/run`.
- Existing response fields stay populated: `query`, `intent`, `answer`, `retrieval_mode`, `top_k`, `use_graph`, `plan`, `steps`, `sources`, `graph_sources`.
- New response fields are additive: `planner_mode`, `node_trace`, `graph_state`, `checkpoint_id`.
- Default tests do not call DashScope, Qwen, OpenAI, Chroma downloads, or any external network.
- Existing business logic remains the source of truth. LangGraph orchestrates the existing planner and tools.
- Recoverable node failures are stored in `errors`, `node_trace`, and `graph_state`; they do not crash a whole Agent run when an answer can still be produced.

## File Map

- Create `src/patent_rag/agent/execution.py` for shared execution helper functions currently embedded in `service.py`.
- Create `src/patent_rag/agent/langgraph_state.py` for `AgentState`, checkpoint creation, graph-state snapshots, and trace append helpers.
- Create `src/patent_rag/agent/replanner.py` for rule-only and LLM-backed replanning boundaries.
- Create `src/patent_rag/agent/langgraph_service.py` for `LangGraphPatentAgentService`.
- Modify `src/patent_rag/agent/schemas.py` to add public observability models.
- Modify `src/patent_rag/agent/service.py` to import helpers from `execution.py` and delegate `run_patent_agent()` to `LangGraphPatentAgentService`.
- Modify `src/patent_rag/agent/__init__.py` to export new public types and service.
- Modify `pyproject.toml` to make `langgraph>=0.2.0` a standard runtime dependency.
- Modify `frontend/src/api/client.ts` to type the additive response fields.
- Modify `frontend/src/App.tsx` and `frontend/src/styles.css` to render planner mode, checkpoint id, and node trace.
- Modify `README.md` to describe the LangGraph Agent runtime and verification commands.
- Add `tests/test_agent_replanner.py`.
- Add `tests/test_langgraph_agent_service.py`.
- Update `tests/test_api.py` to assert additive Agent fields pass through the API.
- Update `tests/test_agent_service.py` only if helper extraction changes import paths.

## Step 0: Create an Isolated Branch

- [ ] Run:

```powershell
git checkout -b codex/langgraph-agent-runtime
```

Expected output contains:

```text
Switched to a new branch 'codex/langgraph-agent-runtime'
```

- [ ] Confirm the branch:

```powershell
git status --short --branch
```

Expected output starts with:

```text
## codex/langgraph-agent-runtime
```

## Step 1: Promote LangGraph to a Standard Runtime Dependency

- [ ] Edit `pyproject.toml`.
- [ ] Add `langgraph>=0.2.0` to `[project].dependencies`.
- [ ] Remove `langgraph>=0.2.0` from the `llm` optional extra so the dependency is declared in one location.

Target dependency block:

```toml
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "pydantic>=2.7.0",
  "pydantic-settings>=2.3.0",
  "python-dotenv>=1.0.1",
  "typer>=0.12.0",
  "rich>=13.7.0",
  "markdown-it-py>=3.0.0",
  "rank-bm25>=0.2.2",
  "networkx>=3.3",
  "langgraph>=0.2.0",
]
```

Target `llm` extra:

```toml
llm = [
  "openai>=1.40.0",
]
```

- [ ] Refresh the local editable install:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,vector]"
```

Expected output contains:

```text
Successfully installed
```

- [ ] Verify LangGraph imports:

```powershell
.\.venv\Scripts\python.exe -c "from langgraph.graph import StateGraph; print(StateGraph.__name__)"
```

Expected output:

```text
StateGraph
```

## Step 2: Add Public Observability Schemas

- [ ] Add schema-level tests before implementation.

Create `tests/test_langgraph_agent_service.py` with this first test and imports:

```python
from patent_rag.agent.schemas import AgentNodeTrace, AgentRunResult


def test_agent_run_result_accepts_langgraph_observability(agent_result_factory):
    result = agent_result_factory(
        planner_mode="rule",
        node_trace=[
            AgentNodeTrace(
                node="rule_plan",
                status="success",
                input_summary="query length=12",
                output_summary="planned 3 steps",
                latency_ms=1.2,
            )
        ],
        graph_state={"pending_steps": 0, "completed_steps": 3, "error_count": 0},
        checkpoint_id="agent-test-checkpoint",
    )

    payload = result.model_dump(mode="json")

    assert payload["planner_mode"] == "rule"
    assert payload["node_trace"][0]["node"] == "rule_plan"
    assert payload["graph_state"]["completed_steps"] == 3
    assert payload["checkpoint_id"] == "agent-test-checkpoint"
```

- [ ] In the same test file, add a local factory fixture. Use existing schemas so this test does not need data files or external services:

```python
import pytest

from patent_rag.agent.schemas import AgentPlan, AgentPlannedStep, AgentToolStep


@pytest.fixture
def agent_result_factory():
    def build_result(**overrides):
        plan = AgentPlan(
            query="test patent task",
            intent="idea_analysis",
            rationale="rule planner selected idea analysis",
            steps=[
                AgentPlannedStep(
                    tool_name="search_patents",
                    reason="retrieve candidate patents",
                )
            ],
        )
        data = {
            "query": "test patent task",
            "intent": "idea_analysis",
            "answer": "bounded answer",
            "retrieval_mode": "keyword",
            "top_k": 3,
            "use_graph": True,
            "plan": plan,
            "steps": [
                AgentToolStep(
                    tool_name="search_patents",
                    tool_input={"query": "test patent task"},
                    status="success",
                    observation="retrieved 1 chunk",
                )
            ],
            "sources": [],
            "graph_sources": [],
        }
        data.update(overrides)
        return AgentRunResult(**data)

    return build_result
```

- [ ] Run the focused red test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_langgraph_agent_service.py
```

Expected failure contains:

```text
ImportError
```

or:

```text
AgentNodeTrace
```

- [ ] Modify `src/patent_rag/agent/schemas.py`.

Add these aliases near the existing literal aliases:

```python
PlannerMode = Literal["rule", "llm", "hybrid"]
AgentNodeStatus = Literal["success", "skipped", "error"]
```

Add this public trace model after `AgentToolStep`:

```python
class AgentNodeTrace(BaseModel):
    """A concise LangGraph node execution trace for debugging and interviews."""

    node: str
    status: AgentNodeStatus
    input_summary: str
    output_summary: str
    latency_ms: float = Field(ge=0)
    error: str | None = None
```

Add these fields to `AgentRunResult`:

```python
    planner_mode: PlannerMode = "rule"
    node_trace: list[AgentNodeTrace] = Field(default_factory=list)
    graph_state: dict[str, Any] = Field(default_factory=dict)
    checkpoint_id: str | None = None
```

Extend `AgentRunResult.from_rag_answer()` with optional keyword parameters:

```python
        planner_mode: PlannerMode = "rule",
        node_trace: list[AgentNodeTrace] | None = None,
        graph_state: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
```

and pass them into the returned model:

```python
            planner_mode=planner_mode,
            node_trace=node_trace or [],
            graph_state=graph_state or {},
            checkpoint_id=checkpoint_id,
```

- [ ] Export `AgentNodeTrace`, `AgentNodeStatus`, and `PlannerMode` from `src/patent_rag/agent/__init__.py`.
- [ ] Re-run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_langgraph_agent_service.py
```

Expected output:

```text
1 passed
```

## Step 3: Extract Shared Execution Helpers

- [ ] Create `src/patent_rag/agent/execution.py`.
- [ ] Move these functions from `src/patent_rag/agent/service.py` into `execution.py` without changing behavior:

```text
_choose_summary_identifier
_choose_graph_keyword
_top_hit_title_or_patent_id
_build_answer_query
_search_observation
_summary_observation
_graph_observation
_rag_observation
_fallback_answer
```

- [ ] Keep the same imports these helpers need:

```python
from patent_rag.agent.planner import extract_patent_id
from patent_rag.agent.schemas import AgentPlan, GraphSearchToolResult, PatentSearchToolResult, PatentSummaryToolResult
from patent_rag.rag import RagAnswer
from patent_rag.rag.service import RetrievalMode
from patent_rag.retrieval import SearchHit
```

- [ ] Modify `src/patent_rag/agent/service.py` to import the helpers from `execution.py`.
- [ ] Remove the moved helper function definitions from `service.py`.
- [ ] Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agent_service.py
```

Expected output:

```text
2 passed
```

## Step 4: Add AgentState and Trace Helpers

- [ ] Extend `tests/test_langgraph_agent_service.py` with state helper tests:

```python
from patent_rag.agent.langgraph_state import (
    AgentState,
    append_node_trace,
    new_checkpoint_id,
    snapshot_graph_state,
)


def test_langgraph_state_helpers_create_safe_observability_snapshot():
    checkpoint_id = new_checkpoint_id()
    state: AgentState = {
        "query": "test patent task",
        "pending_steps": [],
        "completed_steps": [],
        "errors": [],
        "node_trace": [],
        "checkpoint_id": checkpoint_id,
    }

    state = append_node_trace(
        state,
        node="initialize",
        status="success",
        input_summary="new request",
        output_summary="checkpoint created",
        latency_ms=0.5,
    )
    snapshot = snapshot_graph_state(state)

    assert checkpoint_id.startswith("agent-")
    assert snapshot == {
        "pending_steps": 0,
        "completed_steps": 0,
        "error_count": 0,
        "has_answer": False,
        "self_check": {},
    }
    assert state["node_trace"][0].node == "initialize"
```

- [ ] Run the focused red test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_langgraph_agent_service.py
```

Expected failure contains:

```text
langgraph_state
```

- [ ] Create `src/patent_rag/agent/langgraph_state.py`:

```python
"""State and trace helpers for the LangGraph Agent runtime."""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from patent_rag.agent.schemas import (
    AgentNodeStatus,
    AgentNodeTrace,
    AgentPlan,
    AgentPlannedStep,
    AgentToolStep,
    GraphSearchToolResult,
    PatentSearchToolResult,
    PatentSummaryToolResult,
    PlannerMode,
)
from patent_rag.rag import RagAnswer, RagGraphSource, RagSource
from patent_rag.rag.service import RetrievalMode


class AgentState(TypedDict, total=False):
    query: str
    top_k: int
    retrieval_mode: RetrievalMode
    use_graph: bool
    graph_top_k: int
    planner_mode: PlannerMode
    plan: AgentPlan
    pending_steps: list[AgentPlannedStep]
    completed_steps: list[AgentToolStep]
    search_result: PatentSearchToolResult | None
    summary_result: PatentSummaryToolResult | None
    graph_result: GraphSearchToolResult | None
    rag_answer: RagAnswer | None
    answer: str | None
    sources: list[RagSource]
    graph_sources: list[RagGraphSource]
    errors: list[str]
    self_check: dict[str, Any]
    node_trace: list[AgentNodeTrace]
    checkpoint_id: str
    metadata: dict[str, Any]


def new_checkpoint_id() -> str:
    return f"agent-{uuid4().hex}"


def append_node_trace(
    state: AgentState,
    *,
    node: str,
    status: AgentNodeStatus,
    input_summary: str,
    output_summary: str,
    latency_ms: float,
    error: str | None = None,
) -> AgentState:
    next_state = dict(state)
    traces = list(state.get("node_trace", []))
    traces.append(
        AgentNodeTrace(
            node=node,
            status=status,
            input_summary=input_summary,
            output_summary=output_summary,
            latency_ms=latency_ms,
            error=error,
        )
    )
    next_state["node_trace"] = traces
    return next_state


def snapshot_graph_state(state: AgentState) -> dict[str, Any]:
    return {
        "pending_steps": len(state.get("pending_steps", [])),
        "completed_steps": len(state.get("completed_steps", [])),
        "error_count": len(state.get("errors", [])),
        "has_answer": bool(state.get("answer")),
        "self_check": state.get("self_check", {}),
    }
```

- [ ] Re-run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_langgraph_agent_service.py
```

Expected output:

```text
2 passed
```

## Step 5: Add Replanner Boundary

- [ ] Create `tests/test_agent_replanner.py`:

```python
from patent_rag.agent.planner import PatentAgentPlanner
from patent_rag.agent.replanner import LlmAgentReplanner, RuleOnlyReplanner
from patent_rag.agent.schemas import AgentPlannedStep
from patent_rag.llm import ChatMessage


class FakeChatClient:
    model_name = "fake-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[ChatMessage] = []

    def generate(self, messages: list[ChatMessage]) -> str:
        self.messages = messages
        return self.response


def test_rule_only_replanner_keeps_rule_plan():
    plan = PatentAgentPlanner().plan("summarize CN206539886U")
    state = {"query": plan.query, "plan": plan, "pending_steps": plan.steps, "errors": []}

    decision = RuleOnlyReplanner().replan(state)

    assert decision.action == "keep"
    assert decision.planner_mode == "rule"
    assert decision.steps == plan.steps


def test_llm_replanner_accepts_known_tool_append_step():
    plan = PatentAgentPlanner().plan("analyze a cold-chain transport idea")
    client = FakeChatClient(
        '{"action":"append_step","reason":"need graph evidence",'
        '"steps":[{"tool_name":"graph_search","reason":"expand graph evidence"}]}'
    )
    state = {"query": plan.query, "plan": plan, "pending_steps": plan.steps, "errors": []}

    decision = LlmAgentReplanner(client).replan(state)

    assert decision.action == "append_step"
    assert decision.planner_mode == "hybrid"
    assert decision.steps[-1] == AgentPlannedStep(
        tool_name="graph_search",
        reason="expand graph evidence",
    )
    assert client.messages[0].role == "system"


def test_llm_replanner_fails_closed_on_invalid_json():
    plan = PatentAgentPlanner().plan("analyze a cold-chain transport idea")
    client = FakeChatClient("not-json")
    state = {"query": plan.query, "plan": plan, "pending_steps": plan.steps, "errors": []}

    decision = LlmAgentReplanner(client).replan(state)

    assert decision.action == "keep"
    assert decision.planner_mode == "rule"
    assert decision.steps == plan.steps
```

- [ ] Run the focused red test:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agent_replanner.py
```

Expected failure contains:

```text
replanner
```

- [ ] Create `src/patent_rag/agent/replanner.py` with these public classes:

```python
"""Hybrid replanning boundary for the LangGraph Agent runtime."""

from __future__ import annotations

import json
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from patent_rag.agent.langgraph_state import AgentState
from patent_rag.agent.schemas import AgentPlannedStep, AgentToolName, PlannerMode
from patent_rag.llm import ChatClient, ChatMessage

ReplanAction = Literal[
    "keep",
    "append_step",
    "skip_step",
    "supplemental_retrieval",
    "finalize_with_boundary",
]


class ReplanDecision(BaseModel):
    action: ReplanAction = "keep"
    planner_mode: PlannerMode = "rule"
    reason: str
    steps: list[AgentPlannedStep] = Field(default_factory=list)
    boundary_answer: str | None = None


class AgentReplanner(Protocol):
    def replan(self, state: AgentState) -> ReplanDecision:
        """Return a bounded replanning decision."""


class RuleOnlyReplanner:
    def replan(self, state: AgentState) -> ReplanDecision:
        return ReplanDecision(
            action="keep",
            planner_mode="rule",
            reason="Rule plan kept.",
            steps=list(state.get("pending_steps", [])),
        )


class LlmAgentReplanner:
    def __init__(self, chat_client: ChatClient) -> None:
        self.chat_client = chat_client

    def replan(self, state: AgentState) -> ReplanDecision:
        current_steps = list(state.get("pending_steps", []))
        try:
            raw = self.chat_client.generate(_build_replanner_messages(state))
            payload = json.loads(raw)
            decision = ReplanDecision.model_validate(payload)
            _validate_tool_steps(decision.steps)
            if decision.action == "keep":
                return decision.model_copy(update={"steps": current_steps})
            if decision.action == "append_step":
                return decision.model_copy(
                    update={"planner_mode": "hybrid", "steps": current_steps + decision.steps}
                )
            if decision.action == "skip_step":
                return decision.model_copy(
                    update={"planner_mode": "hybrid", "steps": decision.steps}
                )
            return decision.model_copy(update={"planner_mode": "hybrid"})
        except (json.JSONDecodeError, ValidationError, ValueError):
            return ReplanDecision(
                action="keep",
                planner_mode="rule",
                reason="LLM replanner failed validation; rule plan kept.",
                steps=current_steps,
            )


def _build_replanner_messages(state: AgentState) -> list[ChatMessage]:
    plan = state["plan"]
    allowed_tools = ", ".join(_allowed_tool_names())
    pending = [
        {"tool_name": step.tool_name, "reason": step.reason}
        for step in state.get("pending_steps", [])
    ]
    user_content = json.dumps(
        {
            "query": state["query"],
            "intent": plan.intent,
            "pending_steps": pending,
            "errors": state.get("errors", []),
        },
        ensure_ascii=False,
    )
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a bounded patent Agent replanner. Return strict JSON only. "
                f"Allowed tools: {allowed_tools}. "
                "Do not invent patent facts. Do not provide legal conclusions."
            ),
        ),
        ChatMessage(role="user", content=user_content),
    ]


def _validate_tool_steps(steps: list[AgentPlannedStep]) -> None:
    allowed = set(_allowed_tool_names())
    for step in steps:
        if step.tool_name not in allowed:
            raise ValueError(f"Unknown tool: {step.tool_name}")


def _allowed_tool_names() -> tuple[AgentToolName, AgentToolName, AgentToolName, AgentToolName]:
    return ("search_patents", "summarize_patent", "graph_search", "graph_rag_answer")
```

- [ ] Re-run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agent_replanner.py
```

Expected output:

```text
3 passed
```

## Step 6: Implement LangGraphPatentAgentService

- [ ] Extend `tests/test_langgraph_agent_service.py` with a fake tools class. Use deterministic outputs and no filesystem:

```python
from patent_rag.agent.langgraph_service import LangGraphPatentAgentService
from patent_rag.agent.schemas import (
    GraphKeywordMatch,
    GraphSearchToolResult,
    PatentSearchToolResult,
    PatentSummaryToolResult,
)
from patent_rag.rag import RagAnswer, RagGraphSource, RagSource
from patent_rag.retrieval import SearchHit


class FakeAgentTools:
    def __init__(self, *, fail_graph: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_graph = fail_graph

    def search_patents(self, query: str, *, top_k: int, retrieval_mode: str):
        self.calls.append("search_patents")
        return PatentSearchToolResult(
            query=query,
            retrieval_mode="keyword",
            hits=[make_hit()],
        )

    def summarize_patent(self, identifier: str):
        self.calls.append("summarize_patent")
        return PatentSummaryToolResult(
            found=True,
            patent_id=identifier,
            title="Vehicle liquid nitrogen tank fixing frame",
            patent_type="utility_model",
            applicants=["Nanjing Drum Tower Hospital"],
            claim_count=1,
            section_names=["background"],
        )

    def graph_search(self, keyword: str, *, limit: int):
        self.calls.append("graph_search")
        if self.fail_graph:
            raise RuntimeError("graph unavailable")
        return GraphSearchToolResult(
            keyword=keyword,
            matches=[
                GraphKeywordMatch(
                    patent_id="CN206539886U",
                    title="Vehicle liquid nitrogen tank fixing frame",
                    keywords=["fixing frame"],
                )
            ],
        )

    def graph_rag_answer(
        self,
        question: str,
        *,
        top_k: int,
        retrieval_mode: str,
        use_graph: bool,
        graph_top_k: int,
    ):
        self.calls.append("graph_rag_answer")
        return RagAnswer(
            question=question,
            answer="Use the existing fixing-frame structure as grounded evidence. [S1][G1]",
            retrieval_mode="keyword",
            top_k=top_k,
            use_graph=use_graph,
            sources=[
                RagSource(
                    source_id="S1",
                    chunk_id="CN206539886U:background:abc",
                    patent_id="CN206539886U",
                    title="Vehicle liquid nitrogen tank fixing frame",
                    section="background",
                    score=1.0,
                    snippet="A fixing frame reduces collision risk during transport.",
                    source_file="patant/example.md",
                )
            ],
            graph_sources=[
                RagGraphSource(
                    source_id="G1",
                    patent_id="CN206539886U",
                    title="Vehicle liquid nitrogen tank fixing frame",
                    score=1.0,
                    keywords=["fixing frame"],
                    relation_summary="Keyword: fixing frame",
                )
            ],
        )


def make_hit() -> SearchHit:
    return SearchHit(
        chunk_id="CN206539886U:background:abc",
        patent_id="CN206539886U",
        title="Vehicle liquid nitrogen tank fixing frame",
        section="background",
        score=0.8,
        snippet="A fixing frame reduces collision risk during transport.",
        source_file="patant/example.md",
    )
```

- [ ] Add a compatibility test:

```python
def test_langgraph_runtime_keeps_legacy_result_fields():
    tools = FakeAgentTools()
    service = LangGraphPatentAgentService(tools=tools)

    result = service.run(
        "analyze a cold-chain sample transport device",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    assert result.intent == "idea_analysis"
    assert result.answer
    assert result.plan.steps
    assert [step.tool_name for step in result.steps] == tools.calls
    assert result.sources[0].patent_id == "CN206539886U"
    assert result.graph_sources[0].source_id == "G1"
```

- [ ] Add an observability test:

```python
def test_langgraph_runtime_records_node_trace_and_checkpoint():
    service = LangGraphPatentAgentService(tools=FakeAgentTools())

    result = service.run(
        "analyze a cold-chain sample transport device",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    node_names = [trace.node for trace in result.node_trace]

    assert result.planner_mode == "rule"
    assert result.checkpoint_id is not None
    assert result.checkpoint_id.startswith("agent-")
    assert "initialize" in node_names
    assert "rule_plan" in node_names
    assert "execute_search" in node_names
    assert "execute_graph_rag" in node_names
    assert "finalize" in node_names
    assert result.graph_state["completed_steps"] == len(result.steps)
```

- [ ] Add a recoverable graph failure test:

```python
def test_langgraph_runtime_records_recoverable_graph_error():
    service = LangGraphPatentAgentService(tools=FakeAgentTools(fail_graph=True))

    result = service.run(
        "analyze a cold-chain sample transport device",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    assert result.answer
    assert result.graph_state["error_count"] == 1
    assert any(trace.node == "execute_graph_search" and trace.status == "error" for trace in result.node_trace)
    assert any(step.tool_name == "graph_search" and step.status == "error" for step in result.steps)
```

- [ ] Add a fake replanner test:

```python
from patent_rag.agent.replanner import ReplanDecision


class FakeAppendReplanner:
    def replan(self, state):
        return ReplanDecision(
            action="append_step",
            planner_mode="hybrid",
            reason="force one graph search",
            steps=[
                *state["pending_steps"],
                state["plan"].steps[0].model_copy(
                    update={"tool_name": "graph_search", "reason": "forced graph expansion"}
                ),
            ],
        )


def test_langgraph_runtime_accepts_fake_hybrid_replanner():
    service = LangGraphPatentAgentService(
        tools=FakeAgentTools(),
        replanner=FakeAppendReplanner(),
    )

    result = service.run(
        "summarize CN206539886U",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    assert result.planner_mode == "hybrid"
    assert any(trace.node == "llm_replan" for trace in result.node_trace)
```

- [ ] Run the focused red tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_langgraph_agent_service.py tests\test_agent_replanner.py
```

Expected failure contains:

```text
langgraph_service
```

- [ ] Create `src/patent_rag/agent/langgraph_service.py`.
- [ ] Implement `LangGraphPatentAgentService.__init__()` with the same construction parameters as `PatentAgentService`, plus `replanner`.
- [ ] Build a compiled graph once in `__init__()`:

```python
from langgraph.graph import END, StateGraph

workflow = StateGraph(AgentState)
workflow.add_node("initialize", self._initialize)
workflow.add_node("rule_plan", self._rule_plan)
workflow.add_node("should_replan", self._should_replan)
workflow.add_node("llm_replan", self._llm_replan)
workflow.add_node("execute_search", self._execute_search)
workflow.add_node("execute_summary", self._execute_summary)
workflow.add_node("execute_graph_search", self._execute_graph_search)
workflow.add_node("execute_graph_rag", self._execute_graph_rag)
workflow.add_node("self_check", self._self_check)
workflow.add_node("finalize", self._finalize)

workflow.set_entry_point("initialize")
workflow.add_edge("initialize", "rule_plan")
workflow.add_edge("rule_plan", "should_replan")
workflow.add_conditional_edges(
    "should_replan",
    self._route_after_replan_check,
    {"llm_replan": "llm_replan", "route_tools": "execute_search", "self_check": "self_check"},
)
workflow.add_edge("llm_replan", "execute_search")
workflow.add_conditional_edges(
    "execute_search",
    self._route_next_tool,
    {
        "execute_summary": "execute_summary",
        "execute_graph_search": "execute_graph_search",
        "execute_graph_rag": "execute_graph_rag",
        "self_check": "self_check",
    },
)
workflow.add_conditional_edges(
    "execute_summary",
    self._route_next_tool,
    {
        "execute_search": "execute_search",
        "execute_graph_search": "execute_graph_search",
        "execute_graph_rag": "execute_graph_rag",
        "self_check": "self_check",
    },
)
workflow.add_conditional_edges(
    "execute_graph_search",
    self._route_next_tool,
    {
        "execute_search": "execute_search",
        "execute_summary": "execute_summary",
        "execute_graph_rag": "execute_graph_rag",
        "self_check": "self_check",
    },
)
workflow.add_conditional_edges(
    "execute_graph_rag",
    self._route_next_tool,
    {"self_check": "self_check"},
)
workflow.add_edge("self_check", "finalize")
workflow.add_edge("finalize", END)
self.graph = workflow.compile()
```

- [ ] Keep routing deterministic by consuming `pending_steps` in execution nodes. A tool node skips itself when the next pending step is not its tool. `_route_next_tool()` returns the graph node for the next pending tool or `self_check`.
- [ ] Implement each tool node with a shared timing wrapper:

```python
def _with_trace(self, state: AgentState, node: str, func: Callable[[AgentState], AgentState]) -> AgentState:
    started = perf_counter()
    try:
        next_state = func(state)
        return append_node_trace(
            next_state,
            node=node,
            status="success",
            input_summary=self._summarize_input(state),
            output_summary=self._summarize_output(next_state),
            latency_ms=(perf_counter() - started) * 1000,
        )
    except Exception as exc:
        next_state = dict(state)
        errors = list(state.get("errors", []))
        errors.append(f"{node}: {exc}")
        next_state["errors"] = errors
        return append_node_trace(
            next_state,
            node=node,
            status="error",
            input_summary=self._summarize_input(state),
            output_summary="node failed and execution continued",
            latency_ms=(perf_counter() - started) * 1000,
            error=str(exc),
        )
```

- [ ] For recoverable tool exceptions, also append an `AgentToolStep` with `status="error"` and an empty `output`.
- [ ] For `graph_search` failures, continue to the next pending step so `graph_rag_answer` can still produce text evidence when available.
- [ ] Use helper functions from `execution.py` for identifier selection, graph keyword selection, answer-query construction, observations, and fallback answer creation.
- [ ] Convert final state to `AgentRunResult` in `run()`:

```python
return AgentRunResult.from_rag_answer(
    query=query,
    intent=final_state["plan"].intent,
    retrieval_mode=retrieval_mode,
    top_k=top_k,
    use_graph=use_graph,
    plan=final_state["plan"],
    steps=final_state.get("completed_steps", []),
    rag_answer=rag_answer,
    planner_mode=final_state.get("planner_mode", "rule"),
    node_trace=final_state.get("node_trace", []),
    graph_state=snapshot_graph_state(final_state),
    checkpoint_id=final_state.get("checkpoint_id"),
)
```

- [ ] Re-run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_langgraph_agent_service.py tests\test_agent_replanner.py
```

Expected output contains:

```text
passed
```

## Step 7: Make LangGraph the Default Agent Runtime

- [ ] Modify `src/patent_rag/agent/service.py`.
- [ ] Keep `PatentAgentService` unchanged except for helper imports.
- [ ] Replace the body of `run_patent_agent()` so it instantiates `LangGraphPatentAgentService`.
- [ ] Avoid a module-level circular import by importing inside the function:

```python
    from patent_rag.agent.langgraph_service import LangGraphPatentAgentService

    service = LangGraphPatentAgentService(
        patents_path=patents_path,
        chunks_path=chunks_path,
        index_path=index_path,
        graph_path=graph_path,
        collection_name=collection_name,
        chat_client=chat_client,
        embedding_provider=embedding_provider,
        embedding_model_name=embedding_model_name,
        embedding_dimension=embedding_dimension,
    )
```

- [ ] Export `LangGraphPatentAgentService`, `AgentReplanner`, `RuleOnlyReplanner`, `LlmAgentReplanner`, and `ReplanDecision` from `src/patent_rag/agent/__init__.py`.
- [ ] Update `tests/test_api.py`.
- [ ] In `test_agent_run_endpoint_returns_plan_and_trace`, extend the fake result with:

```python
            planner_mode="rule",
            node_trace=[],
            graph_state={"pending_steps": 0, "completed_steps": 1, "error_count": 0},
            checkpoint_id="agent-api-test",
```

- [ ] Add assertions:

```python
    assert payload["planner_mode"] == "rule"
    assert payload["graph_state"]["completed_steps"] == 1
    assert payload["checkpoint_id"] == "agent-api-test"
```

- [ ] Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_agent_service.py tests\test_langgraph_agent_service.py tests\test_agent_replanner.py tests\test_api.py
```

Expected output contains:

```text
passed
```

## Step 8: Update Frontend Types and Runtime Trace UI

- [ ] Modify `frontend/src/api/client.ts`.
- [ ] Add these types:

```ts
export type AgentPlannerMode = "rule" | "llm" | "hybrid";

export type AgentNodeTrace = {
  node: string;
  status: "success" | "skipped" | "error";
  input_summary: string;
  output_summary: string;
  latency_ms: number;
  error: string | null;
};
```

- [ ] Extend `AgentRunResult`:

```ts
  planner_mode?: AgentPlannerMode;
  node_trace?: AgentNodeTrace[];
  graph_state?: Record<string, unknown>;
  checkpoint_id?: string | null;
```

- [ ] Modify `frontend/src/App.tsx`.
- [ ] In the Agent summary row, add two summary tiles after evidence:

```tsx
              <article>
                <span>Planner</span>
                <strong>{agentResult.planner_mode ?? "rule"}</strong>
              </article>
              <article>
                <span>Checkpoint</span>
                <strong>{agentResult.checkpoint_id ?? "local-run"}</strong>
              </article>
```

- [ ] Change `.agent-summary-row` from 4 columns to responsive auto-fit:

```css
.agent-summary-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  padding: 0 18px;
}
```

- [ ] Add a third trace section under the existing Plan and Trace sections:

```tsx
              {agentResult.node_trace && agentResult.node_trace.length > 0 ? (
                <section>
                  <h3>LangGraph</h3>
                  <div className="agent-step-list">
                    {agentResult.node_trace.map((trace, index) => (
                      <article className="agent-step" key={`${trace.node}-${index}`}>
                        <div className="result-title-row">
                          <h4>{trace.node}</h4>
                          <span>{trace.status}</span>
                        </div>
                        <p>{trace.output_summary}</p>
                        {trace.error ? <p>{trace.error}</p> : null}
                      </article>
                    ))}
                  </div>
                </section>
              ) : null}
```

- [ ] Change `.agent-trace-grid` to support three columns on wide screens:

```css
.agent-trace-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
  padding: 0 18px;
}
```

- [ ] Run:

```powershell
cd frontend
npm.cmd run build
```

Expected output contains:

```text
built in
```

## Step 9: Update README Interview Story

- [ ] Modify `README.md`.
- [ ] Add a section named `LangGraph Agent Runtime`.
- [ ] Include this exact capability summary:

```markdown
## LangGraph Agent Runtime

`POST /agent/run` is backed by a LangGraph state graph. The graph carries a typed AgentState through rule planning, optional hybrid replanning, retrieval, graph expansion, GraphRAG answer generation, self-check, and finalization.

The API remains backward-compatible with the earlier Agent response and adds observability fields:

- `planner_mode`
- `node_trace`
- `graph_state`
- `checkpoint_id`

This makes the project easier to debug, demonstrate, and explain in Agent engineering interviews because every run exposes the plan, tool calls, graph nodes, evidence, and fallback behavior.
```

- [ ] Update the Agent execution flow text to:

```text
user goal
-> LangGraph initialize node
-> rule planner node
-> optional hybrid replanner node
-> search / summary / graph search / GraphRAG tool nodes
-> self-check node
-> finalize node
-> answer + plan + tool trace + node trace + sources + graph_sources
```

## Step 10: Verification Gate

- [ ] Run backend tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected output contains:

```text
passed
```

- [ ] Run frontend build:

```powershell
cd frontend
npm.cmd run build
```

Expected output contains:

```text
built in
```

- [ ] Run import smoke test:

```powershell
.\.venv\Scripts\python.exe -c "from patent_rag.agent import LangGraphPatentAgentService, run_patent_agent; print(LangGraphPatentAgentService.__name__, callable(run_patent_agent))"
```

Expected output:

```text
LangGraphPatentAgentService True
```

- [ ] Run git status:

```powershell
git status --short
```

Expected output lists only intentional source, test, frontend, README, and dependency changes.

## Step 11: Commit

- [ ] Stage implementation files:

```powershell
git add pyproject.toml README.md src/patent_rag/agent tests/test_agent_replanner.py tests/test_langgraph_agent_service.py tests/test_api.py frontend/src/api/client.ts frontend/src/App.tsx frontend/src/styles.css
```

- [ ] Commit:

```powershell
git commit -m "Add LangGraph agent runtime"
```

Expected output contains:

```text
Add LangGraph agent runtime
```

- [ ] Push the branch:

```powershell
git push -u origin codex/langgraph-agent-runtime
```

Expected output contains:

```text
codex/langgraph-agent-runtime
```

## Self-Review Checklist

- [ ] Every design requirement from `docs/superpowers/specs/2026-06-15-langgraph-agent-runtime-design.md` maps to at least one implementation step.
- [ ] `POST /agent/run` remains the only public Agent endpoint.
- [ ] Additive response fields do not break existing frontend rendering.
- [ ] Rule-only execution works without external LLM calls.
- [ ] Hybrid replanning is covered by fake replanner tests.
- [ ] Replanner invalid output fails closed to the rule plan.
- [ ] Recoverable graph node failure records trace and returns a bounded answer.
- [ ] Backend and frontend verification commands pass before commit.
