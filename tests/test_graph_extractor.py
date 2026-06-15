from pathlib import Path

from patent_rag.graph import (
    extract_graph_from_documents,
    extract_keywords_for_document,
    find_patents_by_keyword,
    read_graph_json,
    summarize_graph,
    write_graph_json,
)
from patent_rag.ingestion.pipeline import parse_patent_directory


def test_extract_graph_from_current_patents() -> None:
    documents, errors = parse_patent_directory(Path("patant"))

    assert errors == {}

    result = extract_graph_from_documents(documents)
    labels = _count_by_label(result.nodes)
    relations = _count_by_relation(result.edges)

    assert labels["Patent"] == 20
    assert labels["Keyword"] > 0
    assert labels["IPC"] > 0
    assert relations["HAS_IPC"] > 0
    assert relations["MENTIONS_KEYWORD"] > 0


def test_keyword_extraction_finds_liquid_nitrogen_fixed_frame() -> None:
    documents, _ = parse_patent_directory(Path("patant"))
    document = next(item for item in documents if item.metadata.patent_id == "CN206539886U")

    keywords = extract_keywords_for_document(document)

    assert any("液氮罐" in keyword for keyword in keywords)
    assert any("固定架" in keyword for keyword in keywords)


def test_write_read_and_query_graph(tmp_path: Path) -> None:
    documents, _ = parse_patent_directory(Path("patant"))
    result = extract_graph_from_documents(documents)
    graph_path = tmp_path / "patent_graph.json"

    stats = write_graph_json(result, graph_path)
    graph = read_graph_json(graph_path)
    loaded_stats = summarize_graph(graph)
    matches = find_patents_by_keyword(graph, "液氮罐")

    assert stats.node_count == loaded_stats.node_count
    assert stats.edge_count == loaded_stats.edge_count
    assert any(match["patent_id"] == "CN206539886U" for match in matches)


def _count_by_label(nodes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.label] = counts.get(node.label, 0) + 1
    return counts


def _count_by_relation(edges) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edge in edges:
        counts[edge.relation] = counts.get(edge.relation, 0) + 1
    return counts
