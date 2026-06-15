"""Patent-level retrieval evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from patent_rag.retrieval import SearchHit


class RetrievalCase(BaseModel):
    """A retrieval evaluation question with expected patent ids."""

    question: str
    expected_patent_ids: list[str] = Field(min_length=1)
    expected_keywords: list[str] = Field(default_factory=list)
    note: str | None = None


class RetrievalCaseResult(BaseModel):
    """Evaluation result for a single question."""

    question: str
    expected_patent_ids: list[str]
    retrieved_patent_ids: list[str]
    hit_at_k: dict[str, float]
    recall_at_k: dict[str, float]
    reciprocal_rank: float
    first_relevant_rank: int | None = None


class RetrievalEvaluationReport(BaseModel):
    """Aggregate retrieval metrics across evaluation cases."""

    case_count: int
    top_k_values: list[int]
    mean_hit_at_k: dict[str, float]
    mean_recall_at_k: dict[str, float]
    mean_reciprocal_rank: float
    cases: list[RetrievalCaseResult]


def evaluate_retrieval_cases(
    cases: list[RetrievalCase],
    retrieve: Callable[[str, int], list[SearchHit]],
    *,
    top_k_values: list[int] | None = None,
) -> RetrievalEvaluationReport:
    """Evaluate retrieval function over multiple cases."""

    active_top_k_values = sorted(top_k_values or [1, 3, 5])
    max_k = max(active_top_k_values)
    case_results = [
        evaluate_retrieval_hits(case, retrieve(case.question, max_k), active_top_k_values)
        for case in cases
    ]
    return _aggregate_results(case_results, active_top_k_values)


def evaluate_retrieval_hits(
    case: RetrievalCase,
    hits: list[SearchHit],
    top_k_values: list[int] | None = None,
) -> RetrievalCaseResult:
    """Evaluate already retrieved hits for a single case."""

    active_top_k_values = sorted(top_k_values or [1, 3, 5])
    expected_ids = set(case.expected_patent_ids)
    retrieved_ids = [hit.patent_id for hit in hits]
    first_rank = _first_relevant_rank(retrieved_ids, expected_ids)

    hit_at_k: dict[str, float] = {}
    recall_at_k: dict[str, float] = {}
    for top_k in active_top_k_values:
        retrieved_at_k = set(retrieved_ids[:top_k])
        relevant_at_k = expected_ids.intersection(retrieved_at_k)
        hit_at_k[f"hit@{top_k}"] = 1.0 if relevant_at_k else 0.0
        recall_at_k[f"recall@{top_k}"] = round(
            len(relevant_at_k) / len(expected_ids),
            4,
        )

    return RetrievalCaseResult(
        question=case.question,
        expected_patent_ids=case.expected_patent_ids,
        retrieved_patent_ids=retrieved_ids,
        hit_at_k=hit_at_k,
        recall_at_k=recall_at_k,
        reciprocal_rank=round(1 / first_rank, 4) if first_rank else 0.0,
        first_relevant_rank=first_rank,
    )


def _first_relevant_rank(retrieved_ids: list[str], expected_ids: set[str]) -> int | None:
    for index, patent_id in enumerate(retrieved_ids, start=1):
        if patent_id in expected_ids:
            return index
    return None


def _aggregate_results(
    case_results: list[RetrievalCaseResult],
    top_k_values: list[int],
) -> RetrievalEvaluationReport:
    case_count = len(case_results)
    if case_count == 0:
        return RetrievalEvaluationReport(
            case_count=0,
            top_k_values=top_k_values,
            mean_hit_at_k={f"hit@{top_k}": 0.0 for top_k in top_k_values},
            mean_recall_at_k={f"recall@{top_k}": 0.0 for top_k in top_k_values},
            mean_reciprocal_rank=0.0,
            cases=[],
        )

    mean_hit_at_k = {
        f"hit@{top_k}": round(
            sum(result.hit_at_k[f"hit@{top_k}"] for result in case_results) / case_count,
            4,
        )
        for top_k in top_k_values
    }
    mean_recall_at_k = {
        f"recall@{top_k}": round(
            sum(result.recall_at_k[f"recall@{top_k}"] for result in case_results) / case_count,
            4,
        )
        for top_k in top_k_values
    }
    mean_reciprocal_rank = round(
        sum(result.reciprocal_rank for result in case_results) / case_count,
        4,
    )
    return RetrievalEvaluationReport(
        case_count=case_count,
        top_k_values=top_k_values,
        mean_hit_at_k=mean_hit_at_k,
        mean_recall_at_k=mean_recall_at_k,
        mean_reciprocal_rank=mean_reciprocal_rank,
        cases=case_results,
    )
