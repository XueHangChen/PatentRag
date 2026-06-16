from patent_rag.agent.langgraph_service import LangGraphPatentAgentService
from patent_rag.agent.planner import PatentAgentPlanner
from patent_rag.agent.schemas import (
    AgentPlan,
    AgentPlannedStep,
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
            title="车载液氮罐固定架",
            patent_type="utility_model",
            applicants=["南京鼓楼医院"],
            claim_count=1,
            section_names=["背景技术"],
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
                    title="车载液氮罐固定架",
                    keywords=["液氮罐固定架"],
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
            answer="可参考固定架限位结构降低运输碰撞风险。[S1][G1]",
            retrieval_mode="keyword",
            top_k=top_k,
            use_graph=use_graph,
            sources=[
                RagSource(
                    source_id="S1",
                    chunk_id="CN206539886U:background:abc",
                    patent_id="CN206539886U",
                    title="车载液氮罐固定架",
                    section="背景技术",
                    score=1.0,
                    snippet="固定架降低运输过程中的碰撞风险。",
                    source_file="patant/实用新型1.md",
                )
            ],
            graph_sources=[
                RagGraphSource(
                    source_id="G1",
                    patent_id="CN206539886U",
                    title="车载液氮罐固定架",
                    score=1.0,
                    keywords=["液氮罐固定架"],
                    relation_summary="关键词：液氮罐固定架",
                )
            ],
        )


class FakeLlmPlanner:
    def __init__(self, *, accepted: bool = True) -> None:
        self.accepted = accepted
        self.calls: list[str] = []

    def plan(self, query: str):
        from patent_rag.agent.llm_planner import LlmPlanResult

        self.calls.append(query)
        if not self.accepted:
            return LlmPlanResult(
                accepted=False,
                rejection_reason="fake_failure",
                metadata={"source": "llm", "fallback_used": True},
            )

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


def make_hit() -> SearchHit:
    return SearchHit(
        chunk_id="CN206539886U:background:abc",
        patent_id="CN206539886U",
        title="车载液氮罐固定架",
        section="背景技术",
        score=0.8,
        snippet="固定架降低运输过程中的碰撞风险。",
        source_file="patant/实用新型1.md",
    )


def test_langgraph_runtime_keeps_legacy_result_fields() -> None:
    tools = FakeAgentTools()
    service = LangGraphPatentAgentService(tools=tools)

    result = service.run(
        "我想设计一个低温样本运输装置，可以怎么改进？",
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


def test_langgraph_runtime_records_node_trace_and_checkpoint() -> None:
    service = LangGraphPatentAgentService(tools=FakeAgentTools())

    result = service.run(
        "我想设计一个低温样本运输装置，可以怎么改进？",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    node_names = [trace.node_name for trace in result.node_trace]

    assert result.planner_mode == "rule"
    assert result.checkpoint_id is not None
    assert result.checkpoint_id.startswith("agent-")
    assert "initialize" in node_names
    assert "rule_plan" in node_names
    assert "execute_search" in node_names
    assert "execute_graph_rag" in node_names
    assert "finalize" in node_names
    assert result.graph_state["completed_step_count"] == len(result.steps)


def test_langgraph_runtime_records_recoverable_graph_error() -> None:
    service = LangGraphPatentAgentService(tools=FakeAgentTools(fail_graph=True))

    result = service.run(
        "我想设计一个低温样本运输装置，可以怎么改进？",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    assert result.answer
    assert result.graph_state["errors"] == 1
    assert any(
        trace.node_name == "execute_graph_search" and trace.status == "error"
        for trace in result.node_trace
    )
    assert any(step.tool_name == "graph_search" and step.status == "error" for step in result.steps)


def test_langgraph_runtime_uses_enabled_llm_initial_plan() -> None:
    tools = FakeAgentTools()
    llm_planner = FakeLlmPlanner()
    service = LangGraphPatentAgentService(
        tools=tools,
        llm_planner=llm_planner,
        enable_llm_planner=True,
    )

    result = service.run(
        "闅忎究闂竴涓笉鍚鍒欏叧閿瘝浣嗛渶瑕佷笓鍒╅棶绛旂殑闂",
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
        "鎴戞兂璁捐涓€涓綆娓╂牱鏈繍杈撹缃紝鍙互鎬庝箞鏀硅繘锛?",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    llm_trace = next(trace for trace in result.node_trace if trace.node_name == "llm_plan")
    rule_plan = PatentAgentPlanner().plan(result.query)

    assert result.intent == rule_plan.intent
    assert [step.tool_name for step in result.plan.steps] == [
        step.tool_name for step in rule_plan.steps
    ]
    assert result.planner_mode == "rule"
    assert llm_trace.details["fallback_used"] is True
    assert llm_trace.details["rejection_reason"] == "fake_failure"
