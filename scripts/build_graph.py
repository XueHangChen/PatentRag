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

from patent_rag.domain import PatentDocument  # noqa: E402
from patent_rag.graph import extract_graph_from_documents, write_graph_json  # noqa: E402
from patent_rag.ingestion.jsonl import read_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build patent knowledge graph.")
    parser.add_argument("--input", type=Path, default=Path("data") / "processed" / "patents.jsonl")
    parser.add_argument("--output", type=Path, default=Path("data") / "graph" / "patent_graph.json")
    parser.add_argument("--keyword-limit", type=int, default=8)
    args = parser.parse_args()

    documents = read_jsonl(args.input, PatentDocument)
    result = extract_graph_from_documents(documents, keyword_limit_per_patent=args.keyword_limit)
    stats = write_graph_json(result, args.output)

    print(json.dumps(stats.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print(f"output_path={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
