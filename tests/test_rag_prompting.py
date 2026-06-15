from patent_rag.llm import ChatMessage
from patent_rag.rag.graph_context import GraphEvidence
from patent_rag.rag.prompting import build_rag_messages, render_evidence_context
from patent_rag.rag.service import generate_answer_from_hits
from patent_rag.retrieval import SearchHit


class FakeChatClient:
    model_name = "fake-chat-model"

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def generate(self, messages: list[ChatMessage]) -> str:
        self.messages = messages
        return "可以通过限位和固定结构降低低温容器运输碰撞风险。[S1]"


def test_render_evidence_context_numbers_sources() -> None:
    hits = [_make_hit()]

    context = render_evidence_context(hits)

    assert "[S1]" in context
    assert "CN206539886U" in context
    assert "车载液氮罐固定架" in context
    assert "背景技术" in context


def test_build_rag_messages_contains_question_and_citation_instruction() -> None:
    messages = build_rag_messages("如何避免低温容器运输碰撞？", [_make_hit()])

    assert messages[0].role == "system"
    assert "只依据用户提供的专利证据" in messages[0].content
    assert messages[1].role == "user"
    assert "如何避免低温容器运输碰撞" in messages[1].content
    assert "[S1]" in messages[1].content


def test_build_rag_messages_contains_graph_evidence() -> None:
    messages = build_rag_messages(
        "如何避免低温容器运输碰撞？",
        [_make_hit()],
        graph_evidence=[_make_graph_evidence()],
    )

    assert "知识图谱证据" in messages[1].content
    assert "[G1]" in messages[1].content
    assert "关键词：液氮罐固定架、车载液氮罐固定架" in messages[1].content


def test_generate_answer_from_hits_uses_chat_client() -> None:
    client = FakeChatClient()

    answer = generate_answer_from_hits("如何避免低温容器运输碰撞？", [_make_hit()], client)

    assert "降低低温容器运输碰撞风险" in answer
    assert client.messages
    assert "CN206539886U" in client.messages[1].content


def test_generate_answer_from_hits_accepts_graph_evidence() -> None:
    client = FakeChatClient()

    generate_answer_from_hits(
        "如何避免低温容器运输碰撞？",
        [_make_hit()],
        client,
        graph_evidence=[_make_graph_evidence()],
    )

    assert "[G1]" in client.messages[1].content
    assert "关联原文 chunk" in client.messages[1].content


def _make_hit() -> SearchHit:
    return SearchHit(
        chunk_id="CN206539886U:背景技术:section:abc",
        patent_id="CN206539886U",
        title="车载液氮罐固定架",
        section="背景技术",
        score=0.6017,
        snippet="液氮运输罐在运输过程中会因车辆颠簸发生倾斜、碰撞。",
        source_file="patant/实用新型1.md",
    )


def _make_graph_evidence() -> GraphEvidence:
    return GraphEvidence(
        source_id="G1",
        patent_id="CN206539886U",
        title="车载液氮罐固定架",
        score=1.5,
        matched_terms=["液氮罐"],
        supporting_chunk_ids=["CN206539886U:背景技术:section:abc"],
        keywords=["液氮罐固定架", "车载液氮罐固定架"],
        applicants=["南京鼓楼医院"],
        inventors=["张三"],
        ipc_classes=["A61B10/00"],
        section_names=["背景技术", "实用新型内容"],
        claim_numbers=[1, 2, 3],
        relation_summary="关键词：液氮罐固定架、车载液氮罐固定架；申请人：南京鼓楼医院",
    )
