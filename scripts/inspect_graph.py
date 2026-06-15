"""Inspect a local patent knowledge graph JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patent_rag.graph import find_patents_by_keyword, read_graph_json, summarize_graph  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect patent knowledge graph.")
    parser.add_argument("--graph", type=Path, default=Path("data") / "graph" / "patent_graph.json")
    parser.add_argument("--keyword", default=None)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    graph = read_graph_json(args.graph)
    stats = summarize_graph(graph)
    print(json.dumps(stats.model_dump(mode="json"), ensure_ascii=False, indent=2))

    if args.keyword:
        matches = find_patents_by_keyword(graph, args.keyword, limit=args.limit)
        print(
            json.dumps(
                {"keyword": args.keyword, "matches": matches},
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
