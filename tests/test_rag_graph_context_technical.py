import json
from pathlib import Path

import networkx as nx

from patent_rag.rag import retrieve_graph_evidence
from patent_rag.retrieval import SearchHit


def test_retrieve_graph_evidence_includes_technical_semantic_nodes(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "patent_graph.json"
    _write_technical_graph(graph_path)

    evidence = retrieve_graph_evidence(
        "How does the transport frame avoid tipping?",
        [_make_hit()],
        graph_path=graph_path,
        top_k=2,
    )

    assert len(evidence) == 1
    assert evidence[0].technical_fields == ["medical transport equipment"]
    assert evidence[0].problems == ["tank tips during transport"]
    assert evidence[0].components == ["limiting ring"]
    assert evidence[0].solutions == ["fixing frame limits tank shaking"]
    assert evidence[0].effects == ["improves transport stability"]
    assert "Technical problems: tank tips during transport" in evidence[0].relation_summary
    assert "Key components: limiting ring" in evidence[0].relation_summary


def test_retrieve_graph_evidence_can_match_technical_node_terms(tmp_path: Path) -> None:
    graph_path = tmp_path / "patent_graph.json"
    _write_technical_graph(graph_path)

    evidence = retrieve_graph_evidence(
        "Which patent uses a limiting ring?",
        [],
        graph_path=graph_path,
        top_k=2,
    )

    assert len(evidence) == 1
    assert evidence[0].patent_id == "CN206539886U"
    assert evidence[0].components == ["limiting ring"]
    assert "limiting" in evidence[0].matched_terms


def _write_technical_graph(graph_path: Path) -> None:
    graph = nx.MultiDiGraph()
    graph.add_node(
        "patent:CN206539886U",
        label="Patent",
        name="Vehicle-mounted liquid nitrogen tank fixing frame",
        properties={"patent_id": "CN206539886U"},
    )
    graph.add_node("technical:field", label="TechnicalField", name="medical transport equipment")
    graph.add_node("technical:problem", label="Problem", name="tank tips during transport")
    graph.add_node("technical:component", label="Component", name="limiting ring")
    graph.add_node(
        "technical:solution",
        label="Solution",
        name="fixing frame limits tank shaking",
    )
    graph.add_node("technical:effect", label="Effect", name="improves transport stability")
    graph.add_edge(
        "patent:CN206539886U",
        "technical:field",
        key="edge:technical-field",
        relation="HAS_TECHNICAL_FIELD",
    )
    graph.add_edge(
        "patent:CN206539886U",
        "technical:problem",
        key="edge:problem",
        relation="SOLVES_PROBLEM",
    )
    graph.add_edge(
        "patent:CN206539886U",
        "technical:component",
        key="edge:component",
        relation="USES_COMPONENT",
    )
    graph.add_edge(
        "patent:CN206539886U",
        "technical:solution",
        key="edge:solution",
        relation="PROPOSES_SOLUTION",
    )
    graph.add_edge(
        "patent:CN206539886U",
        "technical:effect",
        key="edge:effect",
        relation="HAS_EFFECT",
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
        chunk_id="CN206539886U:abstract:abc",
        patent_id="CN206539886U",
        title="Vehicle-mounted liquid nitrogen tank fixing frame",
        section="abstract",
        score=0.8,
        snippet="The frame limits shaking during transport.",
        source_file="patant/example.md",
    )
