from patent_rag.agent import PatentAgentService
from patent_rag.agent.schemas import (
    GraphKeywordMatch,
    GraphSearchToolResult,
    PatentSearchToolResult,
    PatentSummaryToolResult,
)
from patent_rag.rag import RagAnswer, RagGraphSource, RagSource
from patent_rag.retrieval import SearchHit


class FakeAgentTools:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search_patents(
        self,
        query: str,
        *,
        top_k: int,
        retrieval_mode: str,
    ) -> PatentSearchToolResult:
        self.calls.append("search_patents")
        return PatentSearchToolResult(
            query=query,
            retrieval_mode="keyword",
            hits=[_make_hit()],
        )

    def summarize_patent(self, identifier: str) -> PatentSummaryToolResult:
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

    def graph_search(self, keyword: str, *, limit: int) -> GraphSearchToolResult:
        self.calls.append("graph_search")
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
    ) -> RagAnswer:
        self.calls.append("graph_rag_answer")
        return RagAnswer(
            question=question,
            answer="可以参考车载液氮罐固定架，并结合图谱关系分析改进方向。[S1][G1]",
            retrieval_mode="keyword",
            top_k=top_k,
            use_graph=use_graph,
            sources=[
                RagSource(
                    source_id="S1",
                    chunk_id="CN206539886U:背景技术:section:abc",
                    patent_id="CN206539886U",
                    title="车载液氮罐固定架",
                    section="背景技术",
                    score=1.0,
                    snippet="液氮运输罐在运输过程中容易发生倾斜、碰撞。",
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


def test_agent_runs_idea_analysis_tool_sequence() -> None:
    tools = FakeAgentTools()
    service = PatentAgentService(tools=tools)

    result = service.run(
        "我想设计一个低温样本运输装置，可以怎么改进？",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    assert result.intent == "idea_analysis"
    assert tools.calls == ["search_patents", "graph_search", "graph_rag_answer"]
    assert [step.tool_name for step in result.steps] == tools.calls
    assert result.sources[0].patent_id == "CN206539886U"
    assert result.graph_sources[0].source_id == "G1"


def test_agent_runs_summary_tool_sequence() -> None:
    tools = FakeAgentTools()
    service = PatentAgentService(tools=tools)

    result = service.run(
        "总结 CN206539886U 这篇专利",
        top_k=3,
        retrieval_mode="keyword",
        use_graph=True,
    )

    assert result.intent == "patent_summary"
    assert tools.calls == [
        "search_patents",
        "summarize_patent",
        "graph_search",
        "graph_rag_answer",
    ]
    assert "已读取" in result.steps[1].observation


def _make_hit() -> SearchHit:
    return SearchHit(
        chunk_id="CN206539886U:背景技术:section:abc",
        patent_id="CN206539886U",
        title="车载液氮罐固定架",
        section="背景技术",
        score=0.8,
        snippet="液氮运输罐在运输过程中会因车辆颠簸发生倾斜、碰撞。",
        source_file="patant/实用新型1.md",
    )
