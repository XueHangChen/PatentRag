"""Evaluate GraphRAG behavior on a small curated question set."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from patent_rag.config import get_settings  # noqa: E402
from patent_rag.rag import RagAnswer, RagService  # noqa: E402
from patent_rag.rag.graph_context import retrieve_graph_evidence  # noqa: E402


class GraphRagEvalCase(BaseModel):
    case_id: str
    category: str
    question: str
    expected_patent_ids: list[str] = Field(min_length=1)
    expected_terms: list[str] = Field(default_factory=list)
    note: str | None = None


class GraphRagEvalResult(BaseModel):
    case_id: str
    category: str
    question: str
    expected_patent_ids: list[str]
    source_patent_ids: list[str]
    graph_patent_ids: list[str]
    expected_in_sources: list[str]
    expected_in_graph_sources: list[str]
    missing_expected_patents: list[str]
    graph_source_count: int
    source_count: int
    has_source_citation: bool
    has_graph_citation: bool
    answer_length: int
    answer_preview: str
    graph_evidence_preview: list[dict[str, Any]]


class GraphRagEvalReport(BaseModel):
    created_at: str
    case_count: int
    retrieval_mode: str
    top_k: int
    graph_top_k: int
    graph_path: str
    answer_generation: bool
    expected_graph_hit_rate: float
    expected_any_hit_rate: float
    graph_citation_rate: float
    source_citation_rate: float
    average_answer_length: float
    cases: list[GraphRagEvalResult]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG on curated cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data") / "evaluation" / "graphrag_cases.jsonl",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("docs") / "evaluation" / "graphrag_eval_latest.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("docs") / "evaluation" / "graphrag_eval_latest.md",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--graph-top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-mode",
        choices=["keyword", "vector", "hybrid"],
        default="keyword",
    )
    parser.add_argument(
        "--skip-answer",
        action="store_true",
        help="Only evaluate retrieval and graph evidence, without calling the chat model.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cases = _read_cases(args.cases)
    settings = get_settings()
    service = RagService(graph_path=settings.graph_path)
    results = [
        _evaluate_case(
            service,
            case,
            top_k=args.top_k,
            graph_top_k=args.graph_top_k,
            retrieval_mode=args.retrieval_mode,
            generate_answer=not args.skip_answer,
        )
        for case in cases
    ]
    report = _build_report(
        results,
        graph_path=settings.graph_path,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
        graph_top_k=args.graph_top_k,
        answer_generation=not args.skip_answer,
    )
    _write_json_report(report, args.json_output)
    _write_markdown_report(report, args.markdown_output)
    print(report.model_dump_json(indent=2, ensure_ascii=False))
    return 0


def _read_cases(path: Path) -> list[GraphRagEvalCase]:
    return [
        GraphRagEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _evaluate_case(
    service: RagService,
    case: GraphRagEvalCase,
    *,
    top_k: int,
    graph_top_k: int,
    retrieval_mode: str,
    generate_answer: bool,
) -> GraphRagEvalResult:
    hits = service.retrieve(case.question, top_k=top_k, retrieval_mode=retrieval_mode)
    graph_evidence = retrieve_graph_evidence(
        case.question,
        hits,
        graph_path=service.graph_path,
        top_k=graph_top_k,
    )
    if generate_answer:
        answer = service.answer(
            case.question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            use_graph=True,
            graph_top_k=graph_top_k,
        )
    else:
        answer = RagAnswer(
            question=case.question,
            answer="",
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
            top_k=top_k,
            use_graph=True,
            sources=[],
            graph_sources=[],
        )

    source_patent_ids = _unique([hit.patent_id for hit in hits])
    graph_patent_ids = _unique([item.patent_id for item in graph_evidence])
    found_any = set(source_patent_ids).union(graph_patent_ids)
    return GraphRagEvalResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        expected_patent_ids=case.expected_patent_ids,
        source_patent_ids=source_patent_ids,
        graph_patent_ids=graph_patent_ids,
        expected_in_sources=[
            patent_id
            for patent_id in case.expected_patent_ids
            if patent_id in source_patent_ids
        ],
        expected_in_graph_sources=[
            patent_id for patent_id in case.expected_patent_ids if patent_id in graph_patent_ids
        ],
        missing_expected_patents=[
            patent_id for patent_id in case.expected_patent_ids if patent_id not in found_any
        ],
        graph_source_count=len(graph_evidence),
        source_count=len(hits),
        has_source_citation="[S" in answer.answer,
        has_graph_citation="[G" in answer.answer,
        answer_length=len(answer.answer),
        answer_preview=_preview(answer.answer, 700),
        graph_evidence_preview=[
            {
                "source_id": item.source_id,
                "patent_id": item.patent_id,
                "title": item.title,
                "technical_fields": item.technical_fields[:3],
                "problems": item.problems[:3],
                "components": item.components[:6],
                "solutions": item.solutions[:3],
                "effects": item.effects[:3],
                "relation_summary": _preview(item.relation_summary, 300),
            }
            for item in graph_evidence
        ],
    )


def _build_report(
    results: list[GraphRagEvalResult],
    *,
    graph_path: Path,
    retrieval_mode: str,
    top_k: int,
    graph_top_k: int,
    answer_generation: bool,
) -> GraphRagEvalReport:
    case_count = len(results)
    if case_count == 0:
        return GraphRagEvalReport(
            created_at=_now(),
            case_count=0,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            graph_top_k=graph_top_k,
            graph_path=str(graph_path),
            answer_generation=answer_generation,
            expected_graph_hit_rate=0.0,
            expected_any_hit_rate=0.0,
            graph_citation_rate=0.0,
            source_citation_rate=0.0,
            average_answer_length=0.0,
            cases=[],
        )

    return GraphRagEvalReport(
        created_at=_now(),
        case_count=case_count,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        graph_top_k=graph_top_k,
        graph_path=str(graph_path),
        answer_generation=answer_generation,
        expected_graph_hit_rate=_mean(
            1.0 if result.expected_in_graph_sources else 0.0 for result in results
        ),
        expected_any_hit_rate=_mean(
            1.0 if len(result.missing_expected_patents) < len(result.expected_patent_ids) else 0.0
            for result in results
        ),
        graph_citation_rate=_mean(
            1.0 if result.has_graph_citation else 0.0 for result in results
        ),
        source_citation_rate=_mean(
            1.0 if result.has_source_citation else 0.0 for result in results
        ),
        average_answer_length=round(
            sum(result.answer_length for result in results) / case_count,
            2,
        ),
        cases=results,
    )


def _write_json_report(report: GraphRagEvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _write_markdown_report(report: GraphRagEvalReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GraphRAG Evaluation Report",
        "",
        f"- Created at: {report.created_at}",
        f"- Graph path: `{report.graph_path}`",
        f"- Cases: {report.case_count}",
        f"- Retrieval mode: `{report.retrieval_mode}`",
        f"- top_k / graph_top_k: {report.top_k} / {report.graph_top_k}",
        f"- Answer generation: {report.answer_generation}",
        f"- Expected graph hit rate: {report.expected_graph_hit_rate}",
        f"- Expected any hit rate: {report.expected_any_hit_rate}",
        f"- Graph citation rate: {report.graph_citation_rate}",
        f"- Source citation rate: {report.source_citation_rate}",
        f"- Average answer length: {report.average_answer_length}",
        "",
    ]
    for result in report.cases:
        lines.extend(
            [
                f"## {result.case_id}",
                "",
                f"- Category: {result.category}",
                f"- Question: {result.question}",
                f"- Expected patents: {', '.join(result.expected_patent_ids)}",
                f"- Source patents: {', '.join(result.source_patent_ids)}",
                f"- Graph patents: {', '.join(result.graph_patent_ids)}",
                f"- Expected in graph: {', '.join(result.expected_in_graph_sources) or 'none'}",
                f"- Missing expected: {', '.join(result.missing_expected_patents) or 'none'}",
                (
                    f"- Citations: source={result.has_source_citation}, "
                    f"graph={result.has_graph_citation}"
                ),
                f"- Answer length: {result.answer_length}",
                "",
                "Answer preview:",
                "",
                result.answer_preview or "(skipped)",
                "",
                "Graph evidence preview:",
                "",
            ]
        )
        for evidence in result.graph_evidence_preview:
            lines.append(
                "- "
                + f"{evidence['source_id']} {evidence['patent_id']}: "
                + f"components={evidence['components']}; "
                + f"problems={evidence['problems']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _preview(value: str, max_chars: int) -> str:
    text = value.strip().replace("\r\n", "\n")
    return text if len(text) <= max_chars else f"{text[:max_chars]}..."


def _mean(values: Any) -> float:
    items = list(values)
    return round(sum(items) / len(items), 4) if items else 0.0


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
