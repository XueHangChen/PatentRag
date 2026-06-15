"""Run patent markdown ingestion from the repository root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patent_rag.ingestion.pipeline import ingest_patent_directory  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse patent markdown files into JSONL.")
    parser.add_argument("--raw-dir", type=Path, default=Path("patant"))
    parser.add_argument("--output", type=Path, default=Path("data") / "processed" / "patents.jsonl")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()

    result = ingest_patent_directory(args.raw_dir, args.output, encoding=args.encoding)
    print(f"source_count={result.source_count}")
    print(f"parsed_count={result.parsed_count}")
    print(f"output_path={result.output_path}")

    if result.errors:
        print("errors:")
        for path, error in result.errors.items():
            print(f"- {path}: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
