# LLM Intent Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional LLM-backed intent and tool planner for `/agent/run`, with rule-planner fallback and LangGraph observability.

**Architecture:** Keep `PatentAgentPlanner` as the deterministic fallback. Add a new `LlmIntentPlanner` that parses constrained JSON from a `ChatClient`, validates it with Pydantic and guardrails, then lets `LangGraphPatentAgentService` optionally replace the rule plan before tool execution. LLM planning is disabled by default through settings, so offline tests and local demos remain deterministic.

**Tech Stack:** Python 3.11+, Pydantic v2, FastAPI, LangGraph, pytest, React/Vite TypeScript.

---

## File Map

- Create `src/patent_rag/agent/llm_planner.py`  
  Defines `LlmPlanDecision`, `LlmPlanResult`, `LlmIntentPlanner`, JSON parsing, prompt construction, and guardrail validation.
- Modify `src/patent_rag/config.py`  
  Adds `enable_llm_planner` and `llm_planner_min_confidence`.
- Modify `src/patent_rag/agent/langgraph_state.py`  
  Adds optional `rule_plan`, `planning_metadata`, and `llm_plan_result` state fields.
- Modify `src/patent_rag/agent/langgraph_service.py`  
  Adds `llm_planner`, `enable_llm_planner`, and `llm_planner_min_confidence` constructor options, adds an `llm_plan` graph node between `rule_plan` and `llm_replan`, records skipped/error/success trace details, and uses the accepted LLM plan when valid.
- Modify `src/patent_rag/agent/schemas.py`  
  No response shape change is required, but keep `PlannerMode` supporting `llm`.
- Modify `src/patent_rag/agent/__init__.py`  
  Exports the new planner classes.
- Modify `README.md`  
  Documents the optional LLM planner settings and fallback behavior.
- Create `tests/test_llm_intent_planner.py`  
  Offline tests for valid, invalid, low-confidence, and unsafe LLM plans.
- Modify `tests/test_langgraph_agent_service.py`  
  Adds fake planner/runtime tests for the `llm_plan` node.
- Modify `tests/test_api.py` only if API assertions need to confirm planner fields remain serializable.
- Modify `frontend/src/api/client.ts` only if type comments are desired; no shape change is required.

## Task 1: Add LLM Planner Models and Parser

**Files:**
- Create: `src/patent_rag/agent/llm_planner.py`
- Test: `tests/test_llm_intent_planner.py`

- [ ] **Step 1: Write failing tests for accepted LLM JSON**

Create `tests/test_llm_intent_planner.py` with:

```python
from patent_rag.agent.llm_planner import LlmIntentPlanner
from patent_rag.llm import ChatMessage


class FakeChatClient:
    model_name = "fake-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[ChatMessage] = []

    def generate(self, messages: list[ChatMessage]) -> str:
        self.messages = messages
        return self.response


def test_llm_intent_planner_accepts_valid_structured_plan() -> None:
    client = FakeChatClient(
        """
        {
          "intent": "idea_analysis",
          "confidence": 0.91,
          "confidence_label": "high",
          "rationale": "User asks for design inspiration and improvements.",
          "steps": [
            {"tool_name": "search_patents", "reason": "Find similar patent evidence."},
            {"tool_name": "graph_search", "reason": "Expand graph relationships."},
            {"tool_name": "graph_rag_answer", "reason": "Generate grounded suggestions."}
          ],
          "safety_notes": []
        }
        """
    )
    planner = LlmIntentPlanner(client, min_confidence=0.65)

    result = planner.plan("我想做一个低温样本运输装置，有什么已有方案可以参考？")

    assert result.accepted is True
    assert result.plan is not None
    assert result.plan.intent == "idea_analysis"
    assert [step.tool_name for step in result.plan.steps] == [
        "search_patents",
        "graph_search",
        "graph_rag_answer",
    ]
    assert result.metadata["confidence"] == 0.91
    assert client.messages[0].role == "system"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_intent_planner.py::test_llm_intent_planner_accepts_valid_structured_plan -q
```

Expected:

```text
ModuleNotFoundError: No module named 'patent_rag.agent.llm_planner'
```

- [ ] **Step 3: Implement minimal planner models and success path**

Create `src/patent_rag/agent/llm_planner.py`:

```python
"""LLM-backed intent and tool planning."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from patent_rag.agent.schemas import AgentIntent, AgentPlan, AgentPlannedStep
from patent_rag.llm import ChatClient, ChatMessage

PlannerConfidence = Literal["low", "medium", "high"]


class LlmPlanDecision(BaseModel):
    intent: AgentIntent
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: PlannerConfidence
    rationale: str = Field(min_length=1)
    steps: list[AgentPlannedStep] = Field(min_length=1, max_length=5)
    safety_notes: list[str] = Field(default_factory=list)


class LlmPlanResult(BaseModel):
    accepted: bool
    plan: AgentPlan | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    rejection_reason: str | None = None


class LlmIntentPlanner:
    def __init__(self, chat_client: ChatClient, *, min_confidence: float = 0.65) -> None:
        self.chat_client = chat_client
        self.min_confidence = min_confidence

    def plan(self, query: str) -> LlmPlanResult:
        try:
            raw_response = self.chat_client.generate(_build_messages(query))
            decision = LlmPlanDecision.model_validate(json.loads(raw_response))
            rejection_reason = _validate_decision(decision, self.min_confidence)
            if rejection_reason:
                return LlmPlanResult(
                    accepted=False,
                    rejection_reason=rejection_reason,
                    metadata=_metadata(decision, fallback_used=True),
                )
            return LlmPlanResult(
                accepted=True,
                plan=AgentPlan(
                    query=query,
                    intent=decision.intent,
                    rationale=decision.rationale,
                    steps=decision.steps,
                ),
                metadata=_metadata(decision, fallback_used=False),
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            return LlmPlanResult(
                accepted=False,
                rejection_reason=str(exc),
                metadata={"source": "llm", "fallback_used": True},
            )


def _build_messages(query: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content=(
                "You are a patent Agent intent planner. Return strict JSON only. "
                "Known intents: patent_qa, patent_summary, idea_analysis. "
                "Known tools: search_patents, summarize_patent, graph_search, graph_rag_answer. "
                "Do not invent patent facts. Do not provide legal conclusions."
            ),
        ),
        ChatMessage(role="user", content=query),
    ]


def _validate_decision(decision: LlmPlanDecision, min_confidence: float) -> str | None:
    if decision.confidence < min_confidence:
        return "confidence_below_threshold"
    if _contains_legal_conclusion(decision.rationale) or any(
        _contains_legal_conclusion(note) for note in decision.safety_notes
    ):
        return "unsafe_legal_conclusion"
    last_tool = decision.steps[-1].tool_name
    if last_tool != "graph_rag_answer" and decision.intent != "patent_summary":
        return "missing_final_answer_step"
    return None


def _metadata(decision: LlmPlanDecision, *, fallback_used: bool) -> dict[str, object]:
    return {
        "source": "llm",
        "confidence": decision.confidence,
        "confidence_label": decision.confidence_label,
        "fallback_used": fallback_used,
    }


def _contains_legal_conclusion(value: str) -> bool:
    lowered = value.lower()
    risky_terms = ("侵权", "可专利性结论", "patentable", "infringement")
    return any(term in lowered for term in risky_terms)
```

- [ ] **Step 4: Run success-path test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_intent_planner.py::test_llm_intent_planner_accepts_valid_structured_plan -q
```

Expected:

```text
1 passed
```

## Task 2: Add LLM Planner Guardrail Tests

**Files:**
- Modify: `tests/test_llm_intent_planner.py`
- Modify: `src/patent_rag/agent/llm_planner.py`

- [ ] **Step 1: Add failing guardrail tests**

Append to `tests/test_llm_intent_planner.py`:

```python
def test_llm_intent_planner_rejects_invalid_json() -> None:
    planner = LlmIntentPlanner(FakeChatClient("not-json"), min_confidence=0.65)

    result = planner.plan("请总结这篇专利")

    assert result.accepted is False
    assert result.plan is None
    assert result.metadata["fallback_used"] is True


def test_llm_intent_planner_rejects_unknown_tool_name() -> None:
    client = FakeChatClient(
        """
        {
          "intent": "idea_analysis",
          "confidence": 0.9,
          "confidence_label": "high",
          "rationale": "Need to browse the web.",
          "steps": [{"tool_name": "web_search", "reason": "Unknown tool."}],
          "safety_notes": []
        }
        """
    )
    planner = LlmIntentPlanner(client, min_confidence=0.65)

    result = planner.plan("帮我分析一个新设计")

    assert result.accepted is False
    assert result.plan is None
    assert "web_search" in result.rejection_reason


def test_llm_intent_planner_rejects_low_confidence() -> None:
    client = FakeChatClient(
        """
        {
          "intent": "patent_qa",
          "confidence": 0.2,
          "confidence_label": "low",
          "rationale": "Unclear user request.",
          "steps": [{"tool_name": "graph_rag_answer", "reason": "Try answering."}],
          "safety_notes": []
        }
        """
    )
    planner = LlmIntentPlanner(client, min_confidence=0.65)

    result = planner.plan("这个咋弄")

    assert result.accepted is False
    assert result.rejection_reason == "confidence_below_threshold"


def test_llm_intent_planner_rejects_legal_conclusion_language() -> None:
    client = FakeChatClient(
        """
        {
          "intent": "idea_analysis",
          "confidence": 0.95,
          "confidence_label": "high",
          "rationale": "This can provide an infringement conclusion.",
          "steps": [{"tool_name": "graph_rag_answer", "reason": "Answer directly."}],
          "safety_notes": []
        }
        """
    )
    planner = LlmIntentPlanner(client, min_confidence=0.65)

    result = planner.plan("我的方案会不会侵权？")

    assert result.accepted is False
    assert result.rejection_reason == "unsafe_legal_conclusion"
```

- [ ] **Step 2: Run guardrail tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_intent_planner.py -q
```

Expected: unknown tool and unsafe language tests may fail if validation messages differ.

- [ ] **Step 3: Adjust implementation only if needed**

If unknown tool rejection does not include the unknown name, update the exception branch:

```python
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            return LlmPlanResult(
                accepted=False,
                rejection_reason=str(exc),
                metadata={"source": "llm", "fallback_used": True},
            )
```

This is already present in Task 1. If tests still fail, inspect the exact Pydantic error and assert a stable substring such as `"Input should be"`.

- [ ] **Step 4: Run all planner tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_intent_planner.py -q
```

Expected:

```text
5 passed
```

## Task 3: Add Settings and Exports

**Files:**
- Modify: `src/patent_rag/config.py`
- Modify: `src/patent_rag/agent/__init__.py`
- Test: `tests/test_llm_intent_planner.py`

- [ ] **Step 1: Add settings test**

Append to `tests/test_llm_intent_planner.py`:

```python
from patent_rag.config import Settings


def test_llm_planner_settings_default_to_disabled() -> None:
    settings = Settings()

    assert settings.enable_llm_planner is False
    assert settings.llm_planner_min_confidence == 0.65
```

- [ ] **Step 2: Run settings test to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_intent_planner.py::test_llm_planner_settings_default_to_disabled -q
```

Expected:

```text
AttributeError
```

- [ ] **Step 3: Add settings fields**

Modify `src/patent_rag/config.py` inside `Settings`:

```python
    enable_llm_planner: bool = False
    llm_planner_min_confidence: float = 0.65
```

Place these near the existing LLM settings.

- [ ] **Step 4: Export planner types**

Modify `src/patent_rag/agent/__init__.py`:

```python
from patent_rag.agent.llm_planner import LlmIntentPlanner, LlmPlanDecision, LlmPlanResult
```

Add to `__all__`:

```python
    "LlmIntentPlanner",
    "LlmPlanDecision",
    "LlmPlanResult",
```

- [ ] **Step 5: Run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_intent_planner.py -q
```

Expected:

```text
6 passed
```

## Task 4: Integrate LLM Planning Node into LangGraph Runtime

**Files:**
- Modify: `src/patent_rag/agent/langgraph_state.py`
- Modify: `src/patent_rag/agent/langgraph_service.py`
- Test: `tests/test_langgraph_agent_service.py`

- [ ] **Step 1: Add fake LLM planner and runtime tests**

Append to `tests/test_langgraph_agent_service.py`:

```python
from patent_rag.agent.schemas import AgentPlan, AgentPlannedStep


class FakeLlmPlanner:
    def __init__(self, *, accepted: bool = True) -> None:
        self.accepted = accepted
        self.calls: list[str] = []

    def plan(self, query: str):
        self.calls.append(query)
        if not self.accepted:
            from patent_rag.agent.llm_planner import LlmPlanResult

            return LlmPlanResult(
                accepted=False,
                rejection_reason="fake_failure",
                metadata={"source": "llm", "fallback_used": True},
            )

        from patent_rag.agent.llm_planner import LlmPlanResult

        return LlmPlanResult(
            accepted=True,
            plan=AgentPlan(
                query=query,
                intent="patent_qa",
                rationale="LLM selected direct patent QA.",
                steps=[
                    AgentPlannedStep(
                        tool_name="search_patents",
                        reason="Find text evidence.",
                    ),
                    AgentPlannedStep(
                        tool_name="graph_rag_answer",
                        reason="Answer with grounded evidence.",
                    ),
                ],
            ),
            metadata={
                "source": "llm",
                "confidence": 0.88,
                "confidence_label": "high",
                "fallback_used": False,
            },
        )


def test_langgraph_runtime_uses_enabled_llm_initial_plan() -> None:
    tools = FakeAgentTools()
    llm_planner = FakeLlmPlanner()
    service = LangGraphPatentAgentService(
        tools=tools,
        llm_planner=llm_planner,
        enable_llm_planner=True,
    )

    result = service.run(
        "随便问一个不含规则关键词但需要专利问答的问题",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    node_names = [trace.node_name for trace in result.node_trace]

    assert result.intent == "patent_qa"
    assert result.planner_mode == "llm"
    assert [step.tool_name for step in result.plan.steps] == [
        "search_patents",
        "graph_rag_answer",
    ]
    assert tools.calls == ["search_patents", "graph_rag_answer"]
    assert "llm_plan" in node_names
    assert llm_planner.calls


def test_langgraph_runtime_keeps_rule_plan_when_llm_plan_rejected() -> None:
    tools = FakeAgentTools()
    service = LangGraphPatentAgentService(
        tools=tools,
        llm_planner=FakeLlmPlanner(accepted=False),
        enable_llm_planner=True,
    )

    result = service.run(
        "我想设计一个低温样本运输装置，可以怎么改进？",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    llm_trace = next(trace for trace in result.node_trace if trace.node_name == "llm_plan")

    assert result.intent == "idea_analysis"
    assert result.planner_mode == "rule"
    assert llm_trace.details["fallback_used"] is True
    assert llm_trace.details["rejection_reason"] == "fake_failure"
```

- [ ] **Step 2: Run new runtime tests to verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_langgraph_agent_service.py::test_langgraph_runtime_uses_enabled_llm_initial_plan tests\test_langgraph_agent_service.py::test_langgraph_runtime_keeps_rule_plan_when_llm_plan_rejected -q
```

Expected:

```text
TypeError: LangGraphPatentAgentService.__init__() got an unexpected keyword argument 'llm_planner'
```

- [ ] **Step 3: Add state fields**

Modify `src/patent_rag/agent/langgraph_state.py`:

```python
    rule_plan: AgentPlan
    planning_metadata: dict[str, Any]
    llm_plan_result: Any
```

- [ ] **Step 4: Modify LangGraph service imports**

In `src/patent_rag/agent/langgraph_service.py`, add:

```python
from patent_rag.config import get_settings
from patent_rag.agent.llm_planner import LlmIntentPlanner
```

- [ ] **Step 5: Extend constructor**

Add constructor parameters:

```python
        llm_planner: LlmIntentPlanner | None = None,
        enable_llm_planner: bool | None = None,
        llm_planner_min_confidence: float | None = None,
```

Inside `__init__`, resolve settings:

```python
        settings = get_settings()
        self.enable_llm_planner = (
            settings.enable_llm_planner
            if enable_llm_planner is None
            else enable_llm_planner
        )
        self.llm_planner = llm_planner
        if self.llm_planner is None and self.enable_llm_planner and chat_client is not None:
            self.llm_planner = LlmIntentPlanner(
                chat_client,
                min_confidence=(
                    llm_planner_min_confidence
                    if llm_planner_min_confidence is not None
                    else settings.llm_planner_min_confidence
                ),
            )
```

- [ ] **Step 6: Add `llm_plan` node**

In `_build_graph()`, add:

```python
        workflow.add_node("llm_plan", self._llm_plan)
```

Change:

```python
        workflow.add_edge("rule_plan", "llm_replan")
```

to:

```python
        workflow.add_edge("rule_plan", "llm_plan")
        workflow.add_edge("llm_plan", "llm_replan")
```

Add method:

```python
    def _llm_plan(self, state: AgentState) -> AgentState:
        if not self.enable_llm_planner or self.llm_planner is None:
            return append_node_trace(
                state,
                node_name="llm_plan",
                status="skipped",
                details={**self._trace_details(state), "enabled": self.enable_llm_planner},
            )

        def run(active_state: AgentState) -> AgentState:
            result = self.llm_planner.plan(active_state["query"])
            next_state: AgentState = {
                **active_state,
                "llm_plan_result": result,
                "planning_metadata": result.metadata,
            }
            if result.accepted and result.plan is not None:
                next_state["plan"] = result.plan
                next_state["pending_steps"] = list(result.plan.steps)
                next_state["planner_mode"] = "llm"
            else:
                metadata = dict(result.metadata)
                metadata["rejection_reason"] = result.rejection_reason
                metadata["fallback_used"] = True
                next_state["planning_metadata"] = metadata
            return next_state

        return self._with_trace(state, "llm_plan", run)
```

- [ ] **Step 7: Preserve rule plan in `_rule_plan`**

Update `_rule_plan` return:

```python
                "plan": plan,
                "rule_plan": plan,
                "pending_steps": list(plan.steps),
                "planning_metadata": {"source": "rule", "fallback_used": False},
```

- [ ] **Step 8: Include planning metadata in trace details**

Update `_trace_details()` to include:

```python
            "fallback_used": bool(state.get("planning_metadata", {}).get("fallback_used", False)),
```

For `rejection_reason`, either add it to `_trace_details()` when present, or add it directly in `_llm_plan` after result rejection:

```python
                metadata["rejection_reason"] = result.rejection_reason
```

Then `append_node_trace` from `_with_trace` will include it if `_trace_details()` reads it:

```python
        planning_metadata = state.get("planning_metadata", {})
        details = {
            ...
        }
        if "rejection_reason" in planning_metadata:
            details["rejection_reason"] = planning_metadata["rejection_reason"]
        return details
```

- [ ] **Step 9: Run runtime tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_langgraph_agent_service.py -q
```

Expected:

```text
5 passed
```

## Task 5: Ensure API and README Compatibility

**Files:**
- Modify: `README.md`
- Test: `tests/test_api.py`

- [ ] **Step 1: Add API assertion for planner mode still serializing**

In `tests/test_api.py::test_agent_run_endpoint_returns_plan_and_trace`, no new shape is needed because the test already asserts `planner_mode`, `node_trace`, `graph_state`, and `checkpoint_id`. Ensure it still passes.

- [ ] **Step 2: Update README**

Add this paragraph near the existing `LangGraph Agent Runtime` section:

```markdown
LLM intent planning is optional. By default the Agent uses the deterministic rule planner for offline reliability. To enable LLM structured planning, configure:

```dotenv
PATENT_RAG_ENABLE_LLM_PLANNER=true
PATENT_RAG_LLM_PLANNER_MIN_CONFIDENCE=0.65
PATENT_RAG_DASHSCOPE_API_KEY=your-dashscope-api-key
```

When enabled, the LLM planner must return validated JSON containing `intent`, `confidence`, `rationale`, and known tool steps. Invalid, low-confidence, or unsafe output falls back to the rule planner.
```

- [ ] **Step 3: Run API tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api.py::test_agent_run_endpoint_returns_plan_and_trace -q
```

Expected:

```text
1 passed
```

## Task 6: Verification Gate

**Files:**
- All changed files from prior tasks.

- [ ] **Step 1: Run planner and LangGraph focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_intent_planner.py tests\test_langgraph_agent_service.py tests\test_agent_replanner.py -q
```

Expected:

```text
passed
```

- [ ] **Step 2: Run full backend test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected:

```text
passed
```

- [ ] **Step 3: Run frontend build**

Run:

```powershell
cd frontend
npm.cmd run build
```

Expected:

```text
✓ built
```

- [ ] **Step 4: Run compile smoke**

Run:

```powershell
python -m compileall src
```

Expected:

```text
Listing 'src'...
```

- [ ] **Step 5: Run import smoke**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from patent_rag.agent import LlmIntentPlanner, LangGraphPatentAgentService; print(LlmIntentPlanner.__name__, LangGraphPatentAgentService.__name__)"
```

Expected:

```text
LlmIntentPlanner LangGraphPatentAgentService
```

## Self-Review Checklist

- [ ] Spec requirement: LLM planner is optional and disabled by default.
- [ ] Spec requirement: rule planner remains fallback.
- [ ] Spec requirement: LLM output is strict JSON and Pydantic-validated.
- [ ] Spec requirement: unknown tools and low confidence are rejected.
- [ ] Spec requirement: LangGraph has an `llm_plan` node before tool execution.
- [ ] Spec requirement: tests are offline and use fake chat clients/planners.
- [ ] Spec requirement: API response remains compatible.
- [ ] Placeholder scan: no `TBD`, `TODO`, or vague implementation instructions remain.
- [ ] Type consistency: `LlmPlanDecision`, `LlmPlanResult`, and `LlmIntentPlanner` names match across tasks.
