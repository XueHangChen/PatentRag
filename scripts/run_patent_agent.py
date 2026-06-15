"""Run the patent Agent workflow from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patent_rag.agent import run_patent_agent  # noqa: E402
from patent_rag.llm import create_chat_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a patent Agent task.")
    parser.add_argument("query")
    parser.add_argument(
        "--patents",
        type=Path,
        default=Path("data") / "processed" / "patents.jsonl",
    )
    parser.add_argument("--chunks", type=Path, default=Path("data") / "processed" / "chunks.jsonl")
    parser.add_argument("--index", type=Path, default=Path("data") / "indexes" / "chroma")
    parser.add_argument("--graph", type=Path, default=Path("data") / "graph" / "patent_graph.json")
    parser.add_argument("--collection", default="patent_chunks")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-mode",
        choices=["keyword", "vector", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--graph-top-k", type=int, default=3)
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dimension", type=int, default=None)
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    chat_client = create_chat_client(
        provider=args.llm_provider,
        model_name=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    result = run_patent_agent(
        args.query,
        chat_client=chat_client,
        patents_path=args.patents,
        chunks_path=args.chunks,
        index_path=args.index,
        graph_path=args.graph,
        collection_name=args.collection,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        use_graph=not args.no_graph,
        graph_top_k=args.graph_top_k,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
    )

    print(f"Intent: {result.intent}")
    print()
    print("Plan:")
    for index, step in enumerate(result.plan.steps, start=1):
        print(f"{index}. {step.tool_name} - {step.reason}")
    print()
    print("Tool Trace:")
    for step in result.steps:
        print(f"- {step.tool_name}: {step.observation}")
    print()
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
