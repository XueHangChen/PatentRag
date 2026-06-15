import json
from pathlib import Path

import networkx as nx

from patent_rag.rag import retrieve_graph_evidence
from patent_rag.retrieval import SearchHit


def test_retrieve_graph_evidence_expands_hit_patent_neighborhood(tmp_path: Path) -> None:
    graph_path = tmp_path / "patent_graph.json"
    _write_test_graph(graph_path)

    evidence = retrieve_graph_evidence(
        "低温样本运输时如何避免容器碰撞？",
        [_make_hit()],
        graph_path=graph_path,
        top_k=2,
    )

    assert len(evidence) == 1
    assert evidence[0].source_id == "G1"
    assert evidence[0].patent_id == "CN206539886U"
    assert "液氮罐固定架" in evidence[0].keywords
    assert "南京鼓楼医院" in evidence[0].applicants
    assert "A61B10/00" in evidence[0].ipc_classes
    assert evidence[0].claim_numbers == [1]
    assert evidence[0].supporting_chunk_ids == ["CN206539886U:背景技术:section:abc"]


def test_retrieve_graph_evidence_can_match_question_keyword(tmp_path: Path) -> None:
    graph_path = tmp_path / "patent_graph.json"
    _write_test_graph(graph_path)

    evidence = retrieve_graph_evidence(
        "液氮罐固定架有哪些相关专利？",
        [],
        graph_path=graph_path,
        top_k=2,
    )

    assert len(evidence) == 1
    assert evidence[0].patent_id == "CN206539886U"
    assert "液氮罐" in evidence[0].matched_terms


def test_retrieve_graph_evidence_returns_empty_when_graph_missing(tmp_path: Path) -> None:
    evidence = retrieve_graph_evidence(
        "液氮罐固定架有哪些相关专利？",
        [_make_hit()],
        graph_path=tmp_path / "missing.json",
    )

    assert evidence == []


def _write_test_graph(graph_path: Path) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "patent:CN206539886U",
        label="Patent",
        name="车载液氮罐固定架",
        properties={"patent_id": "CN206539886U"},
    )
    graph.add_node("keyword:liquid-nitrogen", label="Keyword", name="液氮罐固定架")
    graph.add_node("applicant:nanjing", label="Applicant", name="南京鼓楼医院")
    graph.add_node("ipc:a61b", label="IPC", name="A61B10/00")
    graph.add_node(
        "section:background",
        label="Section",
        name="背景技术",
        properties={"patent_id": "CN206539886U"},
    )
    graph.add_node(
        "claim:1",
        label="Claim",
        name="CN206539886U 权利要求1",
        properties={"patent_id": "CN206539886U", "claim_number": "1"},
    )
    graph.add_edge(
        "patent:CN206539886U",
        "keyword:liquid-nitrogen",
        key="edge:keyword",
        relation="MENTIONS_KEYWORD",
    )
    graph.add_edge(
        "patent:CN206539886U",
        "applicant:nanjing",
        key="edge:applicant",
        relation="HAS_APPLICANT",
    )
    graph.add_edge(
        "patent:CN206539886U",
        "ipc:a61b",
        key="edge:ipc",
        relation="HAS_IPC",
    )
    graph.add_edge(
        "patent:CN206539886U",
        "section:background",
        key="edge:section",
        relation="HAS_SECTION",
    )
    graph.add_edge(
        "patent:CN206539886U",
        "claim:1",
        key="edge:claim",
        relation="HAS_CLAIM",
    )
    payload = {
        "format": "networkx_node_link",
        "directed": True,
        "multigraph": True,
        "graph": nx.node_link_data(graph, edges="edges"),
    }
    graph_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


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
