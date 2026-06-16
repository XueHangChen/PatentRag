"""Schemas for patent Agent planning and tool execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from patent_rag.rag import RagAnswer, RagGraphSource, RagSource
from patent_rag.rag.service import RetrievalMode
from patent_rag.retrieval import SearchHit

AgentIntent = Literal["patent_qa", "patent_summary", "idea_analysis"]
AgentToolName = Literal[
    "search_patents",
    "summarize_patent",
    "graph_search",
    "graph_rag_answer",
]
AgentStepStatus = Literal["success", "skipped", "error"]
PlannerMode = Literal["rule", "rule_only", "llm", "hybrid"]
AgentNodeStatus = Literal["success", "skipped", "error"]


class AgentPlannedStep(BaseModel):
    """A tool call planned before execution."""

    tool_name: AgentToolName
    reason: str


class AgentPlan(BaseModel):
    """A lightweight execution plan for a patent task."""

    query: str
    intent: AgentIntent
    rationale: str
    steps: list[AgentPlannedStep]


class AgentToolStep(BaseModel):
    """A completed tool call and its observation."""

    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    status: AgentStepStatus = "success"
    observation: str
    output: dict[str, Any] = Field(default_factory=dict)


class AgentNodeTrace(BaseModel):
    """A concise Agent runtime node trace for debugging and interviews."""

    node_name: str
    status: AgentNodeStatus
    details: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(default=0.0, ge=0)
    error: str | None = None


class PatentSearchToolResult(BaseModel):
    """Search tool output."""

    query: str
    retrieval_mode: RetrievalMode
    hits: list[SearchHit] = Field(default_factory=list)


class GraphKeywordMatch(BaseModel):
    """Patent match returned from keyword graph search."""

    patent_id: str
    title: str
    keywords: list[str] = Field(default_factory=list)


class GraphSearchToolResult(BaseModel):
    """Graph search tool output."""

    keyword: str
    matches: list[GraphKeywordMatch] = Field(default_factory=list)


class PatentSummaryToolResult(BaseModel):
    """A deterministic patent summary used as an Agent tool observation."""

    found: bool
    patent_id: str | None = None
    title: str | None = None
    patent_type: str | None = None
    applicants: list[str] = Field(default_factory=list)
    inventors: list[str] = Field(default_factory=list)
    ipc_classes: list[str] = Field(default_factory=list)
    abstract: str | None = None
    first_claim: str | None = None
    claim_count: int = 0
    section_names: list[str] = Field(default_factory=list)


class AgentRunResult(BaseModel):
    """Final Agent result returned to CLI/API/frontend callers."""

    query: str
    intent: AgentIntent
    answer: str
    retrieval_mode: RetrievalMode
    top_k: int
    use_graph: bool
    plan: AgentPlan
    steps: list[AgentToolStep] = Field(default_factory=list)
    sources: list[RagSource] = Field(default_factory=list)
    graph_sources: list[RagGraphSource] = Field(default_factory=list)
    planner_mode: PlannerMode = "rule"
    node_trace: list[AgentNodeTrace] = Field(default_factory=list)
    graph_state: dict[str, Any] = Field(default_factory=dict)
    checkpoint_id: str | None = None

    @classmethod
    def from_rag_answer(
        cls,
        *,
        query: str,
        intent: AgentIntent,
        retrieval_mode: RetrievalMode,
        top_k: int,
        use_graph: bool,
        plan: AgentPlan,
        steps: list[AgentToolStep],
        rag_answer: RagAnswer,
        planner_mode: PlannerMode = "rule",
        node_trace: list[AgentNodeTrace] | None = None,
        graph_state: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
    ) -> AgentRunResult:
        """Build an Agent result from the final GraphRAG answer."""

        return cls(
            query=query,
            intent=intent,
            answer=rag_answer.answer,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            use_graph=use_graph,
            plan=plan,
            steps=steps,
            sources=rag_answer.sources,
            graph_sources=rag_answer.graph_sources,
            planner_mode=planner_mode,
            node_trace=node_trace or [],
            graph_state=graph_state or {},
            checkpoint_id=checkpoint_id,
        )
