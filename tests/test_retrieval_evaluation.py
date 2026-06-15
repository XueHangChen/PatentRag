from patent_rag.evaluation import (
    RetrievalCase,
    evaluate_retrieval_cases,
    evaluate_retrieval_hits,
)
from patent_rag.retrieval import SearchHit


def test_evaluate_retrieval_hits_computes_hit_recall_and_mrr() -> None:
    case = RetrievalCase(
        question="低温样本运输时如何避免碰撞？",
        expected_patent_ids=["CN206539886U"],
    )
    hits = [
        _hit("A", "CN000000000A"),
        _hit("B", "CN206539886U"),
        _hit("C", "CN109395173A"),
    ]

    result = evaluate_retrieval_hits(case, hits, top_k_values=[1, 3])

    assert result.hit_at_k["hit@1"] == 0.0
    assert result.hit_at_k["hit@3"] == 1.0
    assert result.recall_at_k["recall@3"] == 1.0
    assert result.first_relevant_rank == 2
    assert result.reciprocal_rank == 0.5


def test_evaluate_retrieval_cases_aggregates_metrics() -> None:
    cases = [
        RetrievalCase(question="question-1", expected_patent_ids=["P1"]),
        RetrievalCase(question="question-2", expected_patent_ids=["P2"]),
    ]

    def retrieve(question: str, top_k: int):
        if question == "question-1":
            return [_hit("A", "P1"), _hit("B", "P3")][:top_k]
        return [_hit("C", "P3"), _hit("D", "P2")][:top_k]

    report = evaluate_retrieval_cases(cases, retrieve, top_k_values=[1, 2])

    assert report.case_count == 2
    assert report.mean_hit_at_k["hit@1"] == 0.5
    assert report.mean_hit_at_k["hit@2"] == 1.0
    assert report.mean_recall_at_k["recall@2"] == 1.0
    assert report.mean_reciprocal_rank == 0.75


def _hit(chunk_id: str, patent_id: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        patent_id=patent_id,
        title=f"title-{patent_id}",
        section="摘要",
        score=1.0,
        snippet=f"snippet-{chunk_id}",
        source_file=f"{patent_id}.md",
    )
