"""Ask a patent question with retrieval-augmented generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patent_rag.llm import create_chat_client  # noqa: E402
from patent_rag.rag import answer_patent_question  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a patent RAG question.")
    parser.add_argument("question")
    parser.add_argument("--chunks", type=Path, default=Path("data") / "processed" / "chunks.jsonl")
    parser.add_argument("--index", type=Path, default=Path("data") / "indexes" / "chroma")
    parser.add_argument("--graph", type=Path, default=Path("data") / "graph" / "patent_graph.json")
    parser.add_argument("--collection", default="patent_chunks")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-graph", action="store_true")
    parser.add_argument("--graph-top-k", type=int, default=3)
    parser.add_argument(
        "--retrieval-mode",
        choices=["keyword", "vector", "hybrid"],
        default="hybrid",
    )
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
    result = answer_patent_question(
        args.question,
        chat_client=chat_client,
        chunks_path=args.chunks,
        index_path=args.index,
        graph_path=args.graph,
        collection_name=args.collection,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        use_graph=args.use_graph,
        graph_top_k=args.graph_top_k,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
    )

    print(result.answer)
    print()
    print("Sources:")
    for source in result.sources:
        print(
            f"- [{source.source_id}] {source.title} [{source.patent_id}] "
            f"{source.section} score={source.score}"
        )
    if result.graph_sources:
        print()
        print("Graph Sources:")
        for source in result.graph_sources:
            print(
                f"- [{source.source_id}] {source.title} [{source.patent_id}] "
                f"score={source.score} relations={source.relation_summary}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
