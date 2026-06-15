"""Build a local Chroma vector index from processed patent chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patent_rag.retrieval import build_vector_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Chroma vector index from patent chunks.")
    parser.add_argument("--chunks", type=Path, default=Path("data") / "processed" / "chunks.jsonl")
    parser.add_argument("--index", type=Path, default=Path("data") / "indexes" / "chroma")
    parser.add_argument("--collection", default="patent_chunks")
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dimension", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    count = build_vector_index(
        args.chunks,
        args.index,
        collection_name=args.collection,
        embedding_provider=args.embedding_provider,
        embedding_model_name=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        reset=not args.no_reset,
        batch_size=args.batch_size,
    )
    print(f"indexed_chunk_count={count}")
    print(f"index_path={args.index}")
    print(f"collection={args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
