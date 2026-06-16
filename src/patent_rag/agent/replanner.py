"""Bounded replanning for Agent runtime execution."""

from __future__ import annotations

import json
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

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
    """A validated planner update decision."""

    action: ReplanAction = "keep"
    planner_mode: PlannerMode = "rule"
    reason: str
    steps: list[AgentPlannedStep] = Field(default_factory=list)
    boundary_answer: str | None = None


class AgentReplanner(Protocol):
    """Interface for optional runtime replanning."""

    def replan(self, state: dict) -> ReplanDecision:
        """Return a bounded replanning decision."""


class RuleOnlyReplanner:
    """Keep the deterministic rule plan unchanged."""

    def replan(self, state: dict) -> ReplanDecision:
        return ReplanDecision(
            action="keep",
            planner_mode="rule",
            reason="Rule plan kept.",
            steps=list(state.get("pending_steps", [])),
        )


class LlmAgentReplanner:
    """Ask an LLM for a constrained plan adjustment and validate it."""

    def __init__(self, chat_client: ChatClient) -> None:
        self.chat_client = chat_client

    def replan(self, state: dict) -> ReplanDecision:
        current_steps = list(state.get("pending_steps", []))
        try:
            raw_response = self.chat_client.generate(_build_replanner_messages(state))
            decision = ReplanDecision.model_validate(json.loads(raw_response))
            _validate_tool_steps(decision.steps)
        except (json.JSONDecodeError, ValidationError, ValueError):
            return ReplanDecision(
                action="keep",
                planner_mode="rule",
                reason="LLM replanner failed validation; rule plan kept.",
                steps=current_steps,
            )

        if decision.action == "keep":
            return decision.model_copy(update={"steps": current_steps})
        if decision.action == "append_step":
            return decision.model_copy(
                update={
                    "planner_mode": "hybrid",
                    "steps": current_steps + decision.steps,
                }
            )
        if decision.action == "skip_step":
            return decision.model_copy(
                update={
                    "planner_mode": "hybrid",
                    "steps": decision.steps,
                }
            )
        return decision.model_copy(update={"planner_mode": "hybrid"})


def _build_replanner_messages(state: dict) -> list[ChatMessage]:
    plan = state["plan"]
    pending_steps = [
        {"tool_name": step.tool_name, "reason": step.reason}
        for step in state.get("pending_steps", [])
    ]
    user_payload = {
        "query": state.get("query", plan.query),
        "intent": plan.intent,
        "pending_steps": pending_steps,
        "errors": state.get("errors", []),
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a bounded patent Agent replanner. Return strict JSON only. "
                f"Allowed tools: {', '.join(_allowed_tool_names())}. "
                "Do not invent patent facts or legal conclusions."
            ),
        ),
        ChatMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False)),
    ]


def _validate_tool_steps(steps: list[AgentPlannedStep]) -> None:
    allowed_tools = set(_allowed_tool_names())
    for step in steps:
        if step.tool_name not in allowed_tools:
            raise ValueError(f"Unknown tool: {step.tool_name}")


def _allowed_tool_names() -> tuple[AgentToolName, AgentToolName, AgentToolName, AgentToolName]:
    return ("search_patents", "summarize_patent", "graph_search", "graph_rag_answer")
