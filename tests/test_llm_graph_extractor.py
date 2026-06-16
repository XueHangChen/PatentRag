import hashlib
import json
from pathlib import Path

from patent_rag.domain import PatentClaim, PatentDocument, PatentMetadata, PatentSection
from patent_rag.domain.schemas import PatentType
from patent_rag.graph import LlmTechnicalGraphExtractor, TechnicalGraphExtractionSummary
from patent_rag.llm import ChatMessage


class FakeChatClient:
    model_name = "fake-graph-model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[ChatMessage] = []

    def generate(self, messages: list[ChatMessage]) -> str:
        self.messages = messages
        return self.response


class RaisingChatClient:
    model_name = "fake-graph-model"

    def generate(self, messages: list[ChatMessage]) -> str:
        raise RuntimeError("provider unavailable")


def make_document() -> PatentDocument:
    return PatentDocument(
        metadata=PatentMetadata(
            patent_id="CN000001U",
            title="Sample transport fixing frame",
            patent_type=PatentType.UTILITY_MODEL,
            applicants=["Example Hospital"],
            inventors=["Alice"],
            ipc_classes=["A61B10/00"],
            source_file=Path("patant/example.md"),
        ),
        abstract=(
            "Liquid nitrogen tanks can tip during transport. "
            "The fixing frame limits tank shaking. "
            "The fixing frame includes a base and a limiting ring. "
            "This improves transport stability."
        ),
        claims=[
            PatentClaim(
                claim_number=1,
                text="A fixing frame comprising a base and a limiting ring.",
            )
        ],
        sections=[
            PatentSection(
                name="abstract",
                text=(
                    "Liquid nitrogen tanks can tip during transport. "
                    "The fixing frame limits tank shaking. "
                    "The fixing frame includes a base and a limiting ring. "
                    "This improves transport stability."
                ),
            )
        ],
    )


def expected_claim_node_id(patent_id: str, claim_number: int) -> str:
    digest = hashlib.sha1(f"Claim|{patent_id}:{claim_number}".encode()).hexdigest()[:12]
    return f"claim:{digest}"


def test_llm_graph_extractor_public_types_are_exported() -> None:
    summary = TechnicalGraphExtractionSummary()

    assert LlmTechnicalGraphExtractor.__name__ == "LlmTechnicalGraphExtractor"
    assert summary == TechnicalGraphExtractionSummary()


def test_llm_technical_graph_extractor_creates_evidence_backed_nodes_and_edges() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Problem",
                        "name": "tank tipping during transport",
                        "evidence_text": "Liquid nitrogen tanks can tip during transport.",
                        "section": "abstract",
                        "confidence": 0.92,
                    },
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    },
                    {
                        "label": "Solution",
                        "name": "fixing frame limits tank shaking",
                        "evidence_text": "The fixing frame limits tank shaking.",
                        "section": "abstract",
                        "confidence": 0.88,
                    },
                    {
                        "label": "Effect",
                        "name": "improves transport stability",
                        "evidence_text": "This improves transport stability.",
                        "section": "abstract",
                        "confidence": 0.86,
                    },
                ],
                "relations": [
                    {
                        "source_name": "fixing frame limits tank shaking",
                        "relation": "USES_COMPONENT",
                        "target_name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.84,
                    }
                ],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    nodes_by_label = {node.label: node for node in result.nodes}
    patent_id = "CN000001U"
    expected_nodes = {
        "Problem": {
            "confidence": "0.92",
            "normalized_name": "tank tipping during transport",
            "evidence_text": "Liquid nitrogen tanks can tip during transport.",
        },
        "Component": {
            "confidence": "0.9",
            "normalized_name": "limiting ring",
            "evidence_text": "The fixing frame includes a base and a limiting ring.",
        },
        "Solution": {
            "confidence": "0.88",
            "normalized_name": "fixing frame limits tank shaking",
            "evidence_text": "The fixing frame limits tank shaking.",
        },
        "Effect": {
            "confidence": "0.86",
            "normalized_name": "improves transport stability",
            "evidence_text": "This improves transport stability.",
        },
    }

    assert set(nodes_by_label) == set(expected_nodes)
    assert len(result.nodes) == 4

    for label, expected in expected_nodes.items():
        node = nodes_by_label[label]
        assert node.properties["patent_id"] == patent_id
        assert node.properties["source"] == "llm"
        assert node.properties["confidence"] == expected["confidence"]
        assert node.properties["normalized_name"] == expected["normalized_name"]
        assert node.evidence[0].text == expected["evidence_text"]

    patent_node_id = f"patent:{patent_id}"
    default_edges = {
        (patent_node_id, nodes_by_label["Problem"].node_id, "SOLVES_PROBLEM"),
        (patent_node_id, nodes_by_label["Component"].node_id, "USES_COMPONENT"),
        (patent_node_id, nodes_by_label["Solution"].node_id, "PROPOSES_SOLUTION"),
        (patent_node_id, nodes_by_label["Effect"].node_id, "HAS_EFFECT"),
    }
    edge_triples = {
        (edge.source_id, edge.target_id, edge.relation)
        for edge in result.edges
    }

    assert default_edges.issubset(edge_triples)
    assert (
        nodes_by_label["Solution"].node_id,
        nodes_by_label["Component"].node_id,
        "USES_COMPONENT",
    ) in edge_triples
    assert sum(1 for edge in result.edges if edge.relation == "USES_COMPONENT") == 2
    assert {edge.relation for edge in result.edges} == {
        "SOLVES_PROBLEM",
        "USES_COMPONENT",
        "PROPOSES_SOLUTION",
        "HAS_EFFECT",
    }
    assert extractor.summary.attempted_patents == 1
    assert extractor.summary.successful_patents == 1
    assert client.messages[0].role == "system"


def test_llm_technical_graph_extractor_deduplicates_repeated_nodes_and_edges() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    },
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    },
                    {
                        "label": "Solution",
                        "name": "fixing frame limits tank shaking",
                        "evidence_text": "The fixing frame limits tank shaking.",
                        "section": "abstract",
                        "confidence": 0.88,
                    },
                ],
                "relations": [
                    {
                        "source_name": "fixing frame limits tank shaking",
                        "relation": "USES_COMPONENT",
                        "target_name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.84,
                    },
                    {
                        "source_name": "fixing frame limits tank shaking",
                        "relation": "USES_COMPONENT",
                        "target_name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.84,
                    },
                ],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    node_ids = [node.node_id for node in result.nodes]
    edge_ids = [edge.edge_id for edge in result.edges]

    assert node_ids == list(dict.fromkeys(node_ids))
    assert edge_ids == list(dict.fromkeys(edge_ids))
    assert len(result.nodes) == 2
    assert len(result.edges) == 3
    assert extractor.summary.added_entities == 2
    assert extractor.summary.added_relations == 3


def test_llm_technical_graph_extractor_drops_low_confidence_entities() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.64,
                    }
                ],
                "relations": [],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    assert result.nodes == []
    assert result.edges == []
    assert extractor.summary.dropped_entities == 1
    assert extractor.summary.added_entities == 0
    assert extractor.summary.added_relations == 0


def test_llm_technical_graph_extractor_drops_entities_without_evidence() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Component",
                        "name": "unmentioned locking clamp",
                        "evidence_text": "The frame includes an unmentioned locking clamp.",
                        "section": "abstract",
                        "confidence": 0.9,
                    }
                ],
                "relations": [],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    assert result.nodes == []
    assert result.edges == []
    assert extractor.summary.dropped_entities == 1
    assert extractor.summary.added_entities == 0
    assert extractor.summary.added_relations == 0


def test_llm_technical_graph_extractor_drops_relations_with_unknown_endpoints() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    }
                ],
                "relations": [
                    {
                        "source_name": "unknown shock absorber",
                        "relation": "USES_COMPONENT",
                        "target_name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.84,
                    }
                ],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    assert len(result.nodes) == 1
    assert result.nodes[0].label == "Component"
    assert len(result.edges) == 1
    assert result.edges[0].relation == "USES_COMPONENT"
    assert result.edges[0].source_id == "patent:CN000001U"
    assert result.edges[0].target_id == result.nodes[0].node_id
    assert extractor.summary.dropped_relations == 1
    assert extractor.summary.added_relations == 1


def test_llm_technical_graph_extractor_resolves_supported_by_claim_endpoint() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    }
                ],
                "relations": [
                    {
                        "source_name": "claim 1",
                        "relation": "SUPPORTED_BY_CLAIM",
                        "target_name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.84,
                    }
                ],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    component = result.nodes[0]
    expected_claim_id = expected_claim_node_id("CN000001U", 1)
    edge_triples = {
        (edge.source_id, edge.target_id, edge.relation)
        for edge in result.edges
    }

    assert (expected_claim_id, component.node_id, "SUPPORTED_BY_CLAIM") in edge_triples
    assert extractor.summary.dropped_relations == 0


def test_llm_technical_graph_extractor_drops_missing_supported_by_claim_endpoint() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    }
                ],
                "relations": [
                    {
                        "source_name": "claim 99",
                        "relation": "SUPPORTED_BY_CLAIM",
                        "target_name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.84,
                    }
                ],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    assert len(result.nodes) == 1
    assert result.edges[0].relation == "USES_COMPONENT"
    assert extractor.summary.dropped_relations == 1
    assert extractor.summary.added_relations == 1


def test_llm_technical_graph_extractor_accepts_common_model_field_aliases() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Component",
                        "text": "limiting ring",
                        "evidence": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    },
                    {
                        "label": "Solution",
                        "text": "fixing frame limits tank shaking",
                        "evidence": "The fixing frame limits tank shaking.",
                        "section": "abstract",
                        "confidence": 0.88,
                    },
                ],
                "relations": [
                    {
                        "source": "fixing frame limits tank shaking",
                        "type": "USES_COMPONENT",
                        "target": "limiting ring",
                        "evidence": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.84,
                    }
                ],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    nodes_by_label = {node.label: node for node in result.nodes}
    edge_triples = {
        (edge.source_id, edge.target_id, edge.relation)
        for edge in result.edges
    }

    assert set(nodes_by_label) == {"Component", "Solution"}
    assert nodes_by_label["Component"].name == "limiting ring"
    assert nodes_by_label["Component"].evidence[0].text == (
        "The fixing frame includes a base and a limiting ring."
    )
    assert (
        nodes_by_label["Solution"].node_id,
        nodes_by_label["Component"].node_id,
        "USES_COMPONENT",
    ) in edge_triples
    assert extractor.summary.successful_patents == 1


def test_llm_technical_graph_extractor_accepts_supported_by_claim_relation_alias() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Component",
                        "name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.9,
                    }
                ],
                "relations": [
                    {
                        "source_name": "claim 1",
                        "relation": "SUPPORTS_CLAIM",
                        "target_name": "limiting ring",
                        "evidence_text": "The fixing frame includes a base and a limiting ring.",
                        "section": "abstract",
                        "confidence": 0.84,
                    }
                ],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    component = result.nodes[0]
    expected_claim_id = expected_claim_node_id("CN000001U", 1)
    edge_triples = {
        (edge.source_id, edge.target_id, edge.relation)
        for edge in result.edges
    }

    assert (expected_claim_id, component.node_id, "SUPPORTED_BY_CLAIM") in edge_triples
    assert extractor.summary.errors == 0


def test_llm_technical_graph_extractor_returns_empty_result_for_invalid_json() -> None:
    client = FakeChatClient("{not valid json")
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    assert result.nodes == []
    assert result.edges == []
    assert extractor.summary.errors == 1
    assert extractor.summary.successful_patents == 0


def test_llm_technical_graph_extractor_returns_empty_result_when_provider_unavailable() -> None:
    extractor = LlmTechnicalGraphExtractor(RaisingChatClient(), min_confidence=0.65)

    result = extractor.extract(make_document())

    assert result.nodes == []
    assert result.edges == []
    assert extractor.summary.errors == 1
    assert extractor.summary.error_messages == ["provider unavailable"]


def test_llm_technical_graph_extractor_drops_legal_conclusion_language() -> None:
    document = make_document()
    document.sections.append(
        PatentSection(
            name="analysis",
            text="The sample contains a non-infringement conclusion.",
        )
    )
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Effect",
                        "name": "non-infringement conclusion",
                        "evidence_text": "non-infringement conclusion",
                        "section": "analysis",
                        "confidence": 0.95,
                    }
                ],
                "relations": [],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(document)

    assert result.nodes == []
    assert result.edges == []
    assert extractor.summary.dropped_entities == 1
    assert extractor.summary.added_entities == 0
    assert extractor.summary.added_relations == 0


def test_llm_technical_graph_extractor_drops_chinese_legal_conclusion_language() -> None:
    document = make_document()
    document.sections.append(
        PatentSection(
            name="analysis",
            text="该段落包含不侵权法律结论。",
        )
    )
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Effect",
                        "name": "不侵权法律结论",
                        "evidence_text": "不侵权法律结论",
                        "section": "analysis",
                        "confidence": 0.95,
                    }
                ],
                "relations": [],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(document)

    assert result.nodes == []
    assert result.edges == []
    assert extractor.summary.dropped_entities == 1
    assert extractor.summary.added_entities == 0
    assert extractor.summary.added_relations == 0


def test_llm_technical_graph_extractor_drops_legal_conclusion_entity_names() -> None:
    client = FakeChatClient(
        json.dumps(
            {
                "entities": [
                    {
                        "label": "Effect",
                        "name": "transport stability",
                        "normalized_name": "non-infringement conclusion",
                        "evidence_text": "This improves transport stability.",
                        "section": "abstract",
                        "confidence": 0.95,
                    }
                ],
                "relations": [],
            }
        )
    )
    extractor = LlmTechnicalGraphExtractor(client, min_confidence=0.65)

    result = extractor.extract(make_document())

    assert result.nodes == []
    assert result.edges == []
    assert extractor.summary.dropped_entities == 1
    assert extractor.summary.added_entities == 0
    assert extractor.summary.added_relations == 0
