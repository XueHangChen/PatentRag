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


def test_rule_only_replanner_keeps_rule_plan() -> None:
    plan = PatentAgentPlanner().plan("summarize CN206539886U")
    state = {"query": plan.query, "plan": plan, "pending_steps": plan.steps, "errors": []}

    decision = RuleOnlyReplanner().replan(state)

    assert decision.action == "keep"
    assert decision.planner_mode == "rule"
    assert decision.steps == plan.steps


def test_llm_replanner_accepts_known_tool_append_step() -> None:
    plan = PatentAgentPlanner().plan("analyze a cold-chain transport idea")
    client = FakeChatClient(
        '{"action":"append_step","reason":"need graph evidence",'
        '"steps":[{"tool_name":"graph_search","reason":"expand graph evidence"}]}'
    )
    state = {"query": plan.query, "plan": plan, "pending_steps": plan.steps, "errors": []}

    decision = LlmAgentReplanner(client).replan(state)

    assert decision.action == "append_step"
    assert decision.planner_mode == "hybrid"
    assert decision.steps == [
        *plan.steps,
        AgentPlannedStep(tool_name="graph_search", reason="expand graph evidence"),
    ]
    assert client.messages[0].role == "system"


def test_llm_replanner_fails_closed_on_invalid_json() -> None:
    plan = PatentAgentPlanner().plan("analyze a cold-chain transport idea")
    client = FakeChatClient("not-json")
    state = {"query": plan.query, "plan": plan, "pending_steps": plan.steps, "errors": []}

    decision = LlmAgentReplanner(client).replan(state)

    assert decision.action == "keep"
    assert decision.planner_mode == "rule"
    assert decision.steps == plan.steps
