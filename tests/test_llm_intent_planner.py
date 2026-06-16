import json

from patent_rag.agent.llm_planner import LlmIntentPlanner
from patent_rag.config import Settings
from patent_rag.llm import ChatMessage


class FakeChatClient:
    model_name = "fake-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[ChatMessage] = []

    def generate(self, messages: list[ChatMessage]) -> str:
        self.messages = messages
        return self.response


class FailingChatClient:
    model_name = "fake-model"

    def generate(self, messages: list[ChatMessage]) -> str:
        raise RuntimeError("planner service unavailable")


def test_llm_intent_planner_accepts_valid_structured_plan() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "intent": "idea_analysis",
                "confidence": 0.91,
                "confidence_label": "high",
                "rationale": "User asks for design inspiration and improvements.",
                "steps": [
                    {"tool_name": "search_patents", "reason": "Find similar patent evidence."},
                    {"tool_name": "graph_search", "reason": "Expand graph relationships."},
                    {"tool_name": "graph_rag_answer", "reason": "Generate grounded suggestions."},
                ],
                "safety_notes": [],
            }
        )
    )
    planner = LlmIntentPlanner(client, min_confidence=0.65)

    result = planner.plan("How can I improve a cold-chain sample transport device?")

    assert result.accepted is True
    assert result.plan is not None
    assert result.plan.intent == "idea_analysis"
    assert [step.tool_name for step in result.plan.steps] == [
        "search_patents",
        "graph_search",
        "graph_rag_answer",
    ]
    assert result.metadata["confidence"] == 0.91
    assert result.metadata["confidence_label"] == "high"
    assert result.raw_response
    assert client.messages[0].role == "system"


def test_llm_intent_planner_rejects_invalid_json() -> None:
    planner = LlmIntentPlanner(FakeChatClient("not-json"), min_confidence=0.65)

    result = planner.plan("Summarize this patent")

    assert result.accepted is False
    assert result.plan is None
    assert result.raw_response == "not-json"
    assert result.metadata["fallback_used"] is True


def test_llm_intent_planner_rejects_unknown_tool_name() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "intent": "idea_analysis",
                "confidence": "high",
                "confidence_score": 0.9,
                "rationale": "Need to browse the web.",
                "steps": [{"tool_name": "web_search", "reason": "Unknown tool."}],
                "safety_notes": [],
            }
        )
    )
    planner = LlmIntentPlanner(client, min_confidence=0.65)

    result = planner.plan("Analyze a new design")

    assert result.accepted is False
    assert result.plan is None
    assert result.rejection_reason is not None
    assert "web_search" in result.rejection_reason


def test_llm_intent_planner_rejects_low_confidence() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "intent": "patent_qa",
                "confidence": "low",
                "confidence_score": 0.9,
                "rationale": "Unclear user request.",
                "steps": [{"tool_name": "graph_rag_answer", "reason": "Try answering."}],
                "safety_notes": [],
            }
        )
    )
    planner = LlmIntentPlanner(client, min_confidence=0.65)

    result = planner.plan("This is unclear")

    assert result.accepted is False
    assert result.rejection_reason == "confidence_below_threshold"


def test_llm_intent_planner_rejects_score_below_threshold() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "intent": "patent_qa",
                "confidence": "high",
                "confidence_score": 0.2,
                "rationale": "The request may be patent QA.",
                "steps": [{"tool_name": "graph_rag_answer", "reason": "Try answering."}],
                "safety_notes": [],
            }
        )
    )
    planner = LlmIntentPlanner(client, min_confidence=0.65)

    result = planner.plan("Maybe patent QA")

    assert result.accepted is False
    assert result.rejection_reason == "confidence_below_threshold"


def test_llm_intent_planner_rejects_legal_conclusion_language() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "intent": "idea_analysis",
                "confidence": "high",
                "confidence_score": 0.95,
                "rationale": "This can provide an infringement conclusion.",
                "steps": [{"tool_name": "graph_rag_answer", "reason": "Answer directly."}],
                "safety_notes": [],
            }
        )
    )
    planner = LlmIntentPlanner(client, min_confidence=0.65)

    result = planner.plan("Will my design infringe?")

    assert result.accepted is False
    assert result.rejection_reason == "unsafe_legal_conclusion"


def test_llm_intent_planner_rejects_chat_client_errors() -> None:
    planner = LlmIntentPlanner(FailingChatClient(), min_confidence=0.65)

    result = planner.plan("Find related patents")

    assert result.accepted is False
    assert result.plan is None
    assert result.metadata["fallback_used"] is True
    assert "planner service unavailable" in str(result.metadata["error"])


def test_llm_planner_settings_default_to_disabled(monkeypatch) -> None:
    monkeypatch.delenv("PATENT_RAG_ENABLE_LLM_PLANNER", raising=False)
    monkeypatch.delenv("PATENT_RAG_LLM_PLANNER_MIN_CONFIDENCE", raising=False)
    settings = Settings(_env_file=None)

    assert settings.enable_llm_planner is False
    assert settings.llm_planner_min_confidence == 0.65
