"""Search the local Chroma vector index from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patent_rag.retrieval import search_vector_chunks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Search patent chunks with Chroma vector search.")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=Path("data") / "indexes" / "chroma")
    parser.add_argument("--collection", default="patent_chunks")
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dimension", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    hits = search_vector_chunks(
        args.index,
        args.query,
        top_k=args.top_k,
        collection_name=args.collection,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
    )
    for index, hit in enumerate(hits, start=1):
        print(f"{index}. {hit.title} [{hit.patent_id}] {hit.section} score={hit.score}")
        print(f"   {hit.snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
