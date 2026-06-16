"""Build a rule-based patent knowledge graph from processed patents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patent_rag.config import get_settings  # noqa: E402
from patent_rag.domain import PatentDocument  # noqa: E402
from patent_rag.graph import (  # noqa: E402
    LlmTechnicalGraphExtractor,
    TechnicalGraphExtractor,
    extract_graph_from_documents,
    write_graph_json,
)
from patent_rag.ingestion.jsonl import read_jsonl  # noqa: E402
from patent_rag.llm import create_chat_client  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Build patent knowledge graph.")
    parser.add_argument("--input", type=Path, default=Path("data") / "processed" / "patents.jsonl")
    parser.add_argument("--output", type=Path, default=Path("data") / "graph" / "patent_graph.json")
    parser.add_argument("--keyword-limit", type=int, default=8)
    parser.add_argument("--llm-technical", action="store_true")
    parser.add_argument("--llm-graph-min-confidence", type=float, default=None)
    parser.add_argument("--llm-graph-max-patents", type=int, default=None)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    settings = get_settings()

    min_confidence = (
        args.llm_graph_min_confidence
        if args.llm_graph_min_confidence is not None
        else settings.llm_graph_min_confidence
    )
    max_patents = (
        args.llm_graph_max_patents
        if args.llm_graph_max_patents is not None
        else settings.llm_graph_max_patents
    )
    technical_extractor: TechnicalGraphExtractor | None = None
    if args.llm_technical or settings.enable_llm_graph_extraction:
        try:
            technical_extractor = LlmTechnicalGraphExtractor(
                create_chat_client(temperature=0.0),
                min_confidence=min_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"llm_technical_extraction_skipped={exc}")

    documents = read_jsonl(args.input, PatentDocument)
    result = extract_graph_from_documents(
        documents,
        keyword_limit_per_patent=args.keyword_limit,
        technical_extractor=technical_extractor,
        technical_max_patents=max_patents,
    )
    stats = write_graph_json(result, args.output)

    print(json.dumps(stats.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if technical_extractor is not None:
        print(
            "technical_extraction="
            + json.dumps(
                technical_extractor.summary.model_dump(mode="json"),
                ensure_ascii=False,
            )
        )
    print(f"output_path={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
