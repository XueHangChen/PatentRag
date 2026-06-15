from patent_rag.retrieval import RetrievalPostprocessConfig, SearchHit, postprocess_hits


def test_postprocess_limits_chunks_per_patent() -> None:
    hits = [
        _hit("A-1", "P1", "背景技术", 0.9),
        _hit("A-2", "P1", "摘要", 0.88),
        _hit("A-3", "P1", "技术领域", 0.87),
        _hit("B-1", "P2", "摘要", 0.7),
    ]

    processed = postprocess_hits(
        hits,
        top_k=3,
        config=RetrievalPostprocessConfig(max_chunks_per_patent=2, section_weight=0.0),
    )

    assert [hit.chunk_id for hit in processed] == ["A-1", "A-2", "B-1"]


def test_postprocess_filters_by_min_score() -> None:
    hits = [
        _hit("A-1", "P1", "摘要", 0.9),
        _hit("B-1", "P2", "摘要", 0.2),
    ]

    processed = postprocess_hits(
        hits,
        top_k=5,
        config=RetrievalPostprocessConfig(min_score=0.5),
    )

    assert [hit.chunk_id for hit in processed] == ["A-1"]


def test_postprocess_section_priority_can_promote_better_evidence_sections() -> None:
    hits = [
        _hit("A-1", "P1", "具体实施方式", 0.90),
        _hit("B-1", "P2", "摘要", 0.88),
    ]

    processed = postprocess_hits(
        hits,
        top_k=2,
        config=RetrievalPostprocessConfig(section_weight=0.5),
    )

    assert processed[0].chunk_id == "B-1"


def _hit(chunk_id: str, patent_id: str, section: str, score: float) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        patent_id=patent_id,
        title=f"title-{patent_id}",
        section=section,
        score=score,
        snippet=f"snippet-{chunk_id}",
        source_file=f"{patent_id}.md",
    )
