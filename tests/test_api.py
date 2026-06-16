from pathlib import Path

from patent_rag.agent import (
    AgentNodeTrace,
    AgentPlan,
    AgentPlannedStep,
    AgentRunResult,
    AgentToolStep,
)
from patent_rag.api.main import (
    AgentRunRequest,
    RagAskRequest,
    SearchRequest,
    _resolve_graph_path,
    create_app,
)
from patent_rag.domain import PatentDocument
from patent_rag.graph import extract_graph_from_documents, write_graph_json
from patent_rag.ingestion.jsonl import read_jsonl
from patent_rag.ingestion.pipeline import ingest_patent_directory
from patent_rag.rag import RagAnswer, RagGraphSource, RagSource
from patent_rag.retrieval import build_chunks_from_patents_file

PROCESSED_PATENTS_PATH = Path("data/processed/patents.jsonl")
PROCESSED_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
GRAPH_PATH = Path("data/graph/patent_graph.json")


def _ensure_api_data() -> None:
    if not PROCESSED_PATENTS_PATH.exists():
        ingest_patent_directory(Path("patant"), PROCESSED_PATENTS_PATH)
    if not PROCESSED_CHUNKS_PATH.exists():
        build_chunks_from_patents_file(PROCESSED_PATENTS_PATH, PROCESSED_CHUNKS_PATH)


def _ensure_graph_data() -> None:
    _ensure_api_data()
    if GRAPH_PATH.exists():
        return
    documents = read_jsonl(PROCESSED_PATENTS_PATH, PatentDocument)
    result = extract_graph_from_documents(documents)
    write_graph_json(result, GRAPH_PATH)


def test_health_endpoint() -> None:
    app = create_app()
    endpoint = _get_route_endpoint(app, "/health")

    payload = endpoint()

    assert payload == {"status": "ok"}


def test_list_patents_endpoint() -> None:
    _ensure_api_data()
    app = create_app()
    endpoint = _get_route_endpoint(app, "/patents")

    payload = endpoint()

    assert len(payload) == 20
    assert payload[0].patent_id
    assert payload[0].title
    assert payload[0].claim_count > 0


def test_search_endpoint_finds_relevant_patent() -> None:
    _ensure_api_data()
    app = create_app()
    endpoint = _get_route_endpoint(app, "/search")

    payload = endpoint(SearchRequest(query="液氮罐运输固定", top_k=3))

    assert payload["query"] == "液氮罐运输固定"
    assert payload["hits"]
    assert payload["hits"][0]["patent_id"] == "CN206539886U"


def test_rag_ask_endpoint_returns_grounded_answer(monkeypatch) -> None:
    _ensure_api_data()

    def fake_answer_patent_question(*args, **kwargs) -> RagAnswer:
        return RagAnswer(
            question=args[0],
            answer="可以通过固定架限位来降低运输碰撞风险。[S1]",
            retrieval_mode=kwargs["retrieval_mode"],
            top_k=kwargs["top_k"],
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
        )

    monkeypatch.setattr("patent_rag.api.main.answer_patent_question", fake_answer_patent_question)
    app = create_app()
    endpoint = _get_route_endpoint(app, "/rag/ask")

    payload = endpoint(
        RagAskRequest(
            question="低温样本运输时如何避免容器碰撞？",
            top_k=3,
            retrieval_mode="keyword",
        )
    )

    assert payload["question"] == "低温样本运输时如何避免容器碰撞？"
    assert payload["answer"]
    assert payload["sources"][0]["patent_id"] == "CN206539886U"
    assert payload["retrieval_mode"] == "keyword"


def test_rag_ask_endpoint_forwards_graph_options(monkeypatch) -> None:
    _ensure_api_data()
    _ensure_graph_data()
    captured_kwargs = {}

    def fake_answer_patent_question(*args, **kwargs) -> RagAnswer:
        captured_kwargs.update(kwargs)
        return RagAnswer(
            question=args[0],
            answer="图谱显示该专利关联液氮罐固定架。[G1]",
            retrieval_mode=kwargs["retrieval_mode"],
            top_k=kwargs["top_k"],
            use_graph=kwargs["use_graph"],
            sources=[],
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

    monkeypatch.setattr("patent_rag.api.main.answer_patent_question", fake_answer_patent_question)
    app = create_app()
    endpoint = _get_route_endpoint(app, "/rag/ask")

    payload = endpoint(
        RagAskRequest(
            question="液氮罐固定架有哪些相关专利？",
            top_k=3,
            retrieval_mode="keyword",
            use_graph=True,
            graph_top_k=2,
        )
    )

    assert captured_kwargs["use_graph"] is True
    assert captured_kwargs["graph_top_k"] == 2
    assert payload["use_graph"] is True
    assert payload["graph_sources"][0]["source_id"] == "G1"


def test_graph_stats_endpoint_returns_summary() -> None:
    _ensure_graph_data()
    app = create_app()
    endpoint = _get_route_endpoint(app, "/graph/stats")

    payload = endpoint()

    assert payload["node_count"] > 0
    assert payload["edge_count"] > 0
    assert payload["node_labels"]["Patent"] == 20
    assert "MENTIONS_KEYWORD" in payload["edge_relations"]


def test_resolve_graph_path_prefers_llm_enhanced_graph(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    default_graph = graph_dir / "patent_graph.json"
    enhanced_graph = graph_dir / "patent_graph_llm_full20_fixed.json"
    default_graph.write_text("{}", encoding="utf-8")
    enhanced_graph.write_text("{}", encoding="utf-8")

    assert _resolve_graph_path(graph_dir) == enhanced_graph


def test_resolve_graph_path_falls_back_to_default_graph(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    default_graph = graph_dir / "patent_graph.json"
    default_graph.write_text("{}", encoding="utf-8")

    assert _resolve_graph_path(graph_dir) == default_graph


def test_graph_search_endpoint_finds_keyword_patent() -> None:
    _ensure_graph_data()
    app = create_app()
    endpoint = _get_route_endpoint(app, "/graph/search")

    payload = endpoint(keyword="液氮罐", limit=5)

    assert payload["keyword"] == "液氮罐"
    assert any(match["patent_id"] == "CN206539886U" for match in payload["matches"])


def test_agent_run_endpoint_returns_plan_and_trace(monkeypatch) -> None:
    _ensure_api_data()
    captured_kwargs = {}

    def fake_run_patent_agent(query: str, **kwargs) -> AgentRunResult:
        captured_kwargs.update(kwargs)
        plan = AgentPlan(
            query=query,
            intent="idea_analysis",
            rationale="测试规划",
            steps=[
                AgentPlannedStep(tool_name="search_patents", reason="检索候选专利"),
                AgentPlannedStep(tool_name="graph_rag_answer", reason="生成最终回答"),
            ],
        )
        return AgentRunResult(
            query=query,
            intent="idea_analysis",
            answer="可以参考现有固定架方案进行改进。[S1]",
            retrieval_mode=kwargs["retrieval_mode"],
            top_k=kwargs["top_k"],
            use_graph=kwargs["use_graph"],
            plan=plan,
            steps=[
                AgentToolStep(
                    tool_name="search_patents",
                    tool_input={"query": query},
                    observation="检索到 1 条候选 chunk。",
                )
            ],
            planner_mode="rule_only",
            node_trace=[
                AgentNodeTrace(
                    node_name="initialize",
                    status="success",
                    details={"query": query},
                )
            ],
            graph_state={"completed_step_count": 1, "errors": 0},
            checkpoint_id="agent-test-checkpoint",
        )

    monkeypatch.setattr("patent_rag.api.main.run_patent_agent", fake_run_patent_agent)
    app = create_app()
    endpoint = _get_route_endpoint(app, "/agent/run")

    payload = endpoint(
        AgentRunRequest(
            query="我想设计一个低温样本运输装置，可以怎么改进？",
            top_k=3,
            retrieval_mode="keyword",
            use_graph=False,
        )
    )

    assert captured_kwargs["retrieval_mode"] == "keyword"
    assert captured_kwargs["use_graph"] is False
    assert payload["intent"] == "idea_analysis"
    assert payload["plan"]["steps"][0]["tool_name"] == "search_patents"
    assert payload["steps"][0]["observation"] == "检索到 1 条候选 chunk。"
    assert payload["planner_mode"] == "rule_only"
    assert payload["node_trace"][0]["node_name"] == "initialize"
    assert payload["graph_state"]["completed_step_count"] == 1
    assert payload["checkpoint_id"] == "agent-test-checkpoint"


def _get_route_endpoint(app, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"Route not found: {path}")
