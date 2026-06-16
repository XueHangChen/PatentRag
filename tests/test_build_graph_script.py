import sys

from patent_rag.config import Settings
from patent_rag.graph import GraphExtractionResult, GraphStats
from scripts import build_graph
from scripts.build_graph import build_arg_parser


def test_settings_default_llm_graph_extraction_is_disabled(monkeypatch) -> None:
    monkeypatch.delenv("PATENT_RAG_ENABLE_LLM_GRAPH_EXTRACTION", raising=False)
    monkeypatch.delenv("PATENT_RAG_LLM_GRAPH_MIN_CONFIDENCE", raising=False)
    monkeypatch.delenv("PATENT_RAG_LLM_GRAPH_MAX_PATENTS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.enable_llm_graph_extraction is False
    assert settings.llm_graph_min_confidence == 0.65
    assert settings.llm_graph_max_patents == 0


def test_build_arg_parser_defaults_to_rule_only() -> None:
    args = build_arg_parser().parse_args([])

    assert args.llm_technical is False
    assert args.llm_graph_min_confidence is None
    assert args.llm_graph_max_patents is None


def test_build_arg_parser_accepts_llm_graph_options() -> None:
    args = build_arg_parser().parse_args(
        [
            "--llm-technical",
            "--llm-graph-min-confidence",
            "0.72",
            "--llm-graph-max-patents",
            "3",
        ]
    )

    assert args.llm_technical is True
    assert args.llm_graph_min_confidence == 0.72
    assert args.llm_graph_max_patents == 3


def test_main_default_rule_only_does_not_create_chat_client(monkeypatch, tmp_path) -> None:
    input_path = tmp_path / "patents.jsonl"
    output_path = tmp_path / "graph.json"
    captured_kwargs = {}

    monkeypatch.setenv("PATENT_RAG_ENABLE_LLM_GRAPH_EXTRACTION", "false")
    monkeypatch.setenv("PATENT_RAG_LLM_GRAPH_MIN_CONFIDENCE", "0.65")
    monkeypatch.setenv("PATENT_RAG_LLM_GRAPH_MAX_PATENTS", "0")
    monkeypatch.setattr(
        build_graph,
        "get_settings",
        lambda: Settings(_env_file=None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_graph.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    monkeypatch.setattr(build_graph, "read_jsonl", lambda path, model: [])

    def fake_extract_graph_from_documents(documents, **kwargs):
        captured_kwargs.update(kwargs)
        return GraphExtractionResult()

    monkeypatch.setattr(
        build_graph,
        "extract_graph_from_documents",
        fake_extract_graph_from_documents,
    )
    monkeypatch.setattr(
        build_graph,
        "write_graph_json",
        lambda result, path: GraphStats(
            node_count=0,
            edge_count=0,
            node_labels={},
            edge_relations={},
        ),
    )

    def fail_create_chat_client(*args, **kwargs):
        raise AssertionError("create_chat_client should not be called for rule-only graph builds")

    monkeypatch.setattr(build_graph, "create_chat_client", fail_create_chat_client)

    assert build_graph.main() == 0
    assert captured_kwargs["technical_extractor"] is None
    assert captured_kwargs["technical_max_patents"] == 0
