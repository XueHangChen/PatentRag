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
    rule_plan: AgentPlan
    planning_metadata: dict[str, Any]
    llm_plan_result: Any
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
    node_trace: list[AgentNodeTrace]
    runtime_checkpoint_id: str
    metadata: dict[str, Any]


def new_checkpoint_id() -> str:
    return f"agent-{uuid4().hex}"


def append_node_trace(
    state: AgentState,
    *,
    node_name: str,
    status: AgentNodeStatus,
    details: dict[str, Any] | None = None,
    latency_ms: float = 0.0,
    error: str | None = None,
) -> AgentState:
    next_state = dict(state)
    traces = list(state.get("node_trace", []))
    traces.append(
        AgentNodeTrace(
            node_name=node_name,
            status=status,
            details=details or {},
            latency_ms=latency_ms,
            error=error,
        )
    )
    next_state["node_trace"] = traces
    return next_state


def snapshot_graph_state(state: AgentState) -> dict[str, Any]:
    return {
        "pending_step_count": len(state.get("pending_steps", [])),
        "completed_step_count": len(state.get("completed_steps", [])),
        "errors": len(state.get("errors", [])),
        "has_answer": bool(state.get("answer")),
    }
