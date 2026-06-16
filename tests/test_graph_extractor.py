from collections.abc import Callable
from pathlib import Path

from patent_rag.domain import GraphEdge, GraphNode, PatentDocument
from patent_rag.graph import (
    GraphExtractionResult,
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


def test_extract_graph_merges_optional_technical_extractor_results() -> None:
    documents, _ = parse_patent_directory(Path("patant"))
    document = documents[0]
    patent_id = document.metadata.patent_id
    problem_node_id = f"problem:{patent_id}"
    edge_id = f"edge:{patent_id}:solves_problem"
    fake_extractor = FakeTechnicalExtractor(problem_node_id, edge_id)

    result = extract_graph_from_documents(
        documents[:1],
        technical_extractor=fake_extractor,
    )

    node_ids = {node.node_id for node in result.nodes}
    edge_ids = {edge.edge_id for edge in result.edges}

    assert fake_extractor.called_patent_ids == [patent_id]
    assert problem_node_id in node_ids
    assert edge_id in edge_ids
    assert f"patent:{patent_id}" in node_ids


def test_technical_max_patents_positive_caps_technical_extraction_to_first_document() -> None:
    documents, _ = parse_patent_directory(Path("patant"))
    fake_extractor = FakeTechnicalExtractor("problem:capped", "edge:capped")

    extract_graph_from_documents(
        documents[:2],
        technical_extractor=fake_extractor,
        technical_max_patents=1,
    )

    assert fake_extractor.called_patent_ids == [documents[0].metadata.patent_id]


def test_technical_max_patents_non_positive_values_are_uncapped() -> None:
    documents, _ = parse_patent_directory(Path("patant"))
    expected_patent_ids = [document.metadata.patent_id for document in documents[:2]]

    for technical_max_patents in (0, -1):
        fake_extractor = FakeTechnicalExtractor(
            f"problem:uncapped:{technical_max_patents}",
            f"edge:uncapped:{technical_max_patents}",
        )

        extract_graph_from_documents(
            documents[:2],
            technical_extractor=fake_extractor,
            technical_max_patents=technical_max_patents,
        )

        assert fake_extractor.called_patent_ids == expected_patent_ids


def test_technical_graph_merge_keeps_existing_rule_records_on_id_collision() -> None:
    documents, _ = parse_patent_directory(Path("patant"))
    document = documents[0]
    patent_id = document.metadata.patent_id
    rule_only_result = extract_graph_from_documents([document])
    original_patent_node = next(
        node for node in rule_only_result.nodes if node.node_id == f"patent:{patent_id}"
    )
    original_edge = rule_only_result.edges[0]

    def colliding_result(document: PatentDocument) -> GraphExtractionResult:
        return GraphExtractionResult(
            nodes=[
                GraphNode(
                    node_id=f"patent:{document.metadata.patent_id}",
                    label="TechnicalPatent",
                    name="Technical extractor should not overwrite this node",
                    properties={"source": "technical"},
                )
            ],
            edges=[
                GraphEdge(
                    edge_id=original_edge.edge_id,
                    source_id=original_edge.source_id,
                    target_id="technical:collision-target",
                    relation="TECHNICAL_COLLISION",
                )
            ],
        )

    fake_extractor = FakeTechnicalExtractor(result_factory=colliding_result)

    result = extract_graph_from_documents([document], technical_extractor=fake_extractor)
    merged_patent_node = next(
        node for node in result.nodes if node.node_id == original_patent_node.node_id
    )
    merged_edge = next(edge for edge in result.edges if edge.edge_id == original_edge.edge_id)

    assert merged_patent_node == original_patent_node
    assert merged_edge == original_edge


class FakeTechnicalExtractor:
    def __init__(
        self,
        problem_node_id: str = "problem:sample",
        edge_id: str = "edge:sample",
        result_factory: Callable[[PatentDocument], GraphExtractionResult] | None = None,
    ) -> None:
        self.problem_node_id = problem_node_id
        self.edge_id = edge_id
        self.result_factory = result_factory
        self.called_patent_ids: list[str] = []

    def extract(self, document: PatentDocument) -> GraphExtractionResult:
        patent_id = document.metadata.patent_id
        self.called_patent_ids.append(patent_id)
        if self.result_factory is not None:
            return self.result_factory(document)

        return GraphExtractionResult(
            nodes=[
                GraphNode(
                    node_id=self.problem_node_id,
                    label="Problem",
                    name="Sample technical problem",
                    properties={"patent_id": patent_id},
                )
            ],
            edges=[
                GraphEdge(
                    edge_id=self.edge_id,
                    source_id=f"patent:{patent_id}",
                    target_id=self.problem_node_id,
                    relation="SOLVES_PROBLEM",
                )
            ],
        )


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
