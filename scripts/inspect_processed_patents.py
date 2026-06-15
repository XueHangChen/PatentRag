"""Inspect processed patent JSONL quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patent_rag.quality import build_quality_report, render_quality_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect processed patent JSONL quality.")
    parser.add_argument("--input", type=Path, default=Path("data") / "processed" / "patents.jsonl")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with code 1 when warnings are present.",
    )
    args = parser.parse_args()

    report = build_quality_report(args.input)
    if args.json:
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
    else:
        print(render_quality_report(report))

    if report.error_count > 0:
        return 1
    if args.fail_on_warning and report.warning_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

