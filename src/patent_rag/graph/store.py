"""NetworkX graph storage helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from pydantic import BaseModel

from patent_rag.graph.extractor import GraphExtractionResult


class GraphStats(BaseModel):
    """Basic graph statistics for quick inspection."""

    node_count: int
    edge_count: int
    node_labels: dict[str, int]
    edge_relations: dict[str, int]


def build_networkx_graph(result: GraphExtractionResult) -> nx.MultiDiGraph:
    """Build a NetworkX graph from extracted graph records."""

    graph = nx.MultiDiGraph()
    for node in result.nodes:
        graph.add_node(
            node.node_id,
            label=node.label,
            name=node.name,
            properties=node.properties,
            evidence=[item.model_dump(mode="json") for item in node.evidence],
        )
    for edge in result.edges:
        graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.edge_id,
            edge_id=edge.edge_id,
            relation=edge.relation,
            properties=edge.properties,
            evidence=[item.model_dump(mode="json") for item in edge.evidence],
        )
    return graph


def write_graph_json(result: GraphExtractionResult, output_path: Path) -> GraphStats:
    """Persist graph records as NetworkX node-link JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = build_networkx_graph(result)
    payload = {
        "format": "networkx_node_link",
        "directed": True,
        "multigraph": True,
        "graph": nx.node_link_data(graph, edges="edges"),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summarize_graph(graph)


def read_graph_json(input_path: Path) -> nx.MultiDiGraph:
    """Load a NetworkX graph from node-link JSON."""

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    graph_payload = payload["graph"] if "graph" in payload else payload
    return nx.node_link_graph(graph_payload, edges="edges")


def summarize_graph(graph: nx.MultiDiGraph) -> GraphStats:
    """Summarize node labels and edge relations."""

    node_labels: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        label = str(data.get("label", "Unknown"))
        node_labels[label] = node_labels.get(label, 0) + 1

    edge_relations: dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        relation = str(data.get("relation", "UNKNOWN"))
        edge_relations[relation] = edge_relations.get(relation, 0) + 1

    return GraphStats(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        node_labels=dict(sorted(node_labels.items())),
        edge_relations=dict(sorted(edge_relations.items())),
    )


def find_patents_by_keyword(
    graph: nx.MultiDiGraph,
    keyword: str,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find patents connected to keyword nodes containing the given text."""

    results_by_patent_id: dict[str, dict[str, Any]] = {}
    for node_id, data in graph.nodes(data=True):
        if data.get("label") != "Keyword" or keyword not in str(data.get("name", "")):
            continue
        for source_id, _, edge_data in graph.in_edges(node_id, data=True):
            if edge_data.get("relation") != "MENTIONS_KEYWORD":
                continue
            patent_data = graph.nodes[source_id]
            patent_id = patent_data.get("properties", {}).get("patent_id", source_id)
            result = results_by_patent_id.setdefault(
                patent_id,
                {
                    "patent_id": patent_id,
                    "title": patent_data.get("name", ""),
                    "keywords": [],
                },
            )
            if data.get("name", "") not in result["keywords"]:
                result["keywords"].append(data.get("name", ""))
            if len(results_by_patent_id) >= limit:
                return list(results_by_patent_id.values())
    return list(results_by_patent_id.values())
