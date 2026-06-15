"""Build retrieval chunks from processed patent JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patent_rag.retrieval import build_chunks_from_patents_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build retrieval chunks from processed patents.")
    parser.add_argument("--input", type=Path, default=Path("data") / "processed" / "patents.jsonl")
    parser.add_argument("--output", type=Path, default=Path("data") / "processed" / "chunks.jsonl")
    args = parser.parse_args()

    count = build_chunks_from_patents_file(args.input, args.output)
    print(f"chunk_count={count}")
    print(f"output_path={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

