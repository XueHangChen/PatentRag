"""LLM-backed intent and tool planning."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from patent_rag.agent.schemas import AgentIntent, AgentPlan, AgentPlannedStep
from patent_rag.llm import ChatClient, ChatMessage

PlannerConfidence = Literal["low", "medium", "high"]

_INTENT_ALIASES: dict[str, AgentIntent] = {
    "patent_qa": "patent_qa",
    "patent_search": "patent_qa",
    "patent_summary": "patent_summary",
    "idea_analysis": "idea_analysis",
    "patent_analysis": "idea_analysis",
    "technology_exploration": "idea_analysis",
}
_TOOL_ALIASES = {
    "search_patents": "search_patents",
    "vector_search": "search_patents",
    "summarize_patent": "summarize_patent",
    "graph_search": "graph_search",
    "graph_rag_answer": "graph_rag_answer",
    "graph_rag": "graph_rag_answer",
}
_LEGAL_CONCLUSION_TERMS = (
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


class LlmPlanDecision(BaseModel):
    """Validated structured plan emitted by an LLM."""

    model_config = ConfigDict(extra="ignore")

    intent: AgentIntent
    confidence: PlannerConfidence
    confidence_score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    steps: list[AgentPlannedStep] = Field(min_length=1, max_length=5)
    safety_notes: list[str] = Field(default_factory=list)


class LlmPlanResult(BaseModel):
    """Outcome of LLM planning, including rejected raw output for fallback."""

    accepted: bool
    plan: AgentPlan | None = None
    decision: LlmPlanDecision | None = None
    raw_response: str | None = None
    rejection_reason: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class LlmIntentPlanner:
    """Ask an LLM for a constrained patent workflow plan and validate it."""

    def __init__(self, chat_client: ChatClient, *, min_confidence: float = 0.65) -> None:
        self.chat_client = chat_client
        self.min_confidence = min_confidence

    def plan(self, query: str) -> LlmPlanResult:
        """Return an accepted executable plan or a rejected fallback result."""

        raw_response: str | None = None
        try:
            raw_response = self.chat_client.generate(_build_messages(query))
            payload = _normalize_payload(json.loads(raw_response))
            decision = LlmPlanDecision.model_validate(payload)
        except json.JSONDecodeError as exc:
            return _rejected_result(
                raw_response=raw_response,
                reason="invalid_json",
                model_name=_model_name(self.chat_client),
                error=str(exc),
            )
        except (ValidationError, ValueError) as exc:
            return _rejected_result(
                raw_response=raw_response,
                reason=str(exc),
                model_name=_model_name(self.chat_client),
                error=str(exc),
            )
        except Exception as exc:
            return _rejected_result(
                raw_response=raw_response,
                reason="chat_client_error",
                model_name=_model_name(self.chat_client),
                error=str(exc),
            )

        rejection_reason = _validate_decision(decision, self.min_confidence)
        if rejection_reason is not None:
            return LlmPlanResult(
                accepted=False,
                decision=decision,
                raw_response=raw_response,
                rejection_reason=rejection_reason,
                metadata=_metadata(decision, _model_name(self.chat_client), fallback_used=True),
            )

        plan = AgentPlan(
            query=query,
            intent=decision.intent,
            rationale=decision.rationale,
            steps=decision.steps,
        )
        return LlmPlanResult(
            accepted=True,
            plan=plan,
            decision=decision,
            raw_response=raw_response,
            metadata=_metadata(decision, _model_name(self.chat_client), fallback_used=False),
        )


def _build_messages(query: str) -> list[ChatMessage]:
    schema_hint = {
        "intent": "patent_qa | patent_summary | idea_analysis",
        "confidence": "low | medium | high",
        "confidence_score": "0.0-1.0",
        "rationale": "short planning rationale, no patent facts or legal conclusions",
        "steps": [
            {"tool_name": "search_patents", "reason": "why this tool is needed"},
            {"tool_name": "graph_search", "reason": "optional graph expansion"},
            {"tool_name": "graph_rag_answer", "reason": "final grounded answer"},
        ],
        "safety_notes": [],
    }
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a patent Agent intent planner. Return strict JSON only. "
                "Allowed intents: patent_qa, patent_summary, idea_analysis. "
                "Accepted aliases: patent_search->patent_qa, "
                "patent_analysis/technology_exploration->idea_analysis. "
                "Allowed tools: search_patents, summarize_patent, graph_search, "
                "graph_rag_answer. Accepted aliases: vector_search->search_patents, "
                "graph_rag->graph_rag_answer. Do not invent patent facts. "
                "Do not provide infringement, validity, patentability, or other legal "
                "conclusions."
            ),
        ),
        ChatMessage(
            role="user",
            content=json.dumps(
                {
                    "query": query,
                    "required_json_schema": schema_hint,
                },
                ensure_ascii=False,
            ),
        ),
    ]


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("LLM planner response must be a JSON object.")

    normalized = dict(payload)
    raw_intent = normalized.get("intent")
    if isinstance(raw_intent, str) and raw_intent in _INTENT_ALIASES:
        normalized["intent"] = _INTENT_ALIASES[raw_intent]

    raw_confidence = normalized.get("confidence")
    if isinstance(raw_confidence, int | float):
        normalized["confidence_score"] = float(raw_confidence)
        normalized["confidence"] = normalized.get("confidence_label") or _label_for_score(
            float(raw_confidence)
        )
    elif "confidence_label" in normalized and "confidence" not in normalized:
        normalized["confidence"] = normalized["confidence_label"]

    if "steps" not in normalized and "tool_sequence" in normalized:
        normalized["steps"] = normalized["tool_sequence"]
    normalized["steps"] = _normalize_steps(normalized.get("steps", []))
    return normalized


def _normalize_steps(raw_steps: Any) -> list[dict[str, str]]:
    if not isinstance(raw_steps, list):
        raise ValueError("LLM planner steps must be a list.")

    normalized_steps: list[dict[str, str]] = []
    for raw_step in raw_steps:
        if isinstance(raw_step, str):
            tool_name = raw_step
            reason = f"Use {raw_step} for the planned workflow."
        elif isinstance(raw_step, dict):
            tool_name = str(raw_step.get("tool_name") or raw_step.get("tool") or "")
            reason = str(raw_step.get("reason") or raw_step.get("rationale") or "")
        else:
            raise ValueError("Each LLM planner step must be a string or object.")

        normalized_steps.append(
            {
                "tool_name": _TOOL_ALIASES.get(tool_name, tool_name),
                "reason": reason or f"Use {tool_name} for the planned workflow.",
            }
        )
    return normalized_steps


def _validate_decision(decision: LlmPlanDecision, min_confidence: float) -> str | None:
    if decision.confidence == "low" or decision.confidence_score < min_confidence:
        return "confidence_below_threshold"

    if _contains_legal_conclusion(decision.rationale):
        return "unsafe_legal_conclusion"
    if any(_contains_legal_conclusion(note) for note in decision.safety_notes):
        return "unsafe_legal_conclusion"
    if any(_contains_legal_conclusion(step.reason) for step in decision.steps):
        return "unsafe_legal_conclusion"

    tool_names = [step.tool_name for step in decision.steps]
    if tool_names[-1] != "graph_rag_answer":
        return "missing_final_answer_step"
    if decision.intent == "patent_summary" and "summarize_patent" not in tool_names:
        return "intent_tool_mismatch"
    if decision.intent == "idea_analysis" and "search_patents" not in tool_names:
        return "intent_tool_mismatch"
    return None


def _metadata(
    decision: LlmPlanDecision,
    model_name: str,
    *,
    fallback_used: bool,
) -> dict[str, object]:
    return {
        "source": "llm",
        "model_name": model_name,
        "intent": decision.intent,
        "confidence": decision.confidence_score,
        "confidence_label": decision.confidence,
        "fallback_used": fallback_used,
    }


def _rejected_result(
    *,
    raw_response: str | None,
    reason: str,
    model_name: str,
    error: str,
) -> LlmPlanResult:
    return LlmPlanResult(
        accepted=False,
        raw_response=raw_response,
        rejection_reason=reason,
        metadata={
            "source": "llm",
            "model_name": model_name,
            "fallback_used": True,
            "error": error,
        },
    )


def _contains_legal_conclusion(value: str) -> bool:
    normalized = value.casefold()
    return any(term.casefold() in normalized for term in _LEGAL_CONCLUSION_TERMS)


def _label_for_score(score: float) -> PlannerConfidence:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _model_name(chat_client: ChatClient) -> str:
    return getattr(chat_client, "model_name", "unknown")
