"""Post-processing and lightweight reranking for retrieved chunks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from patent_rag.retrieval.keyword import SearchHit

DEFAULT_SECTION_PRIORITY = {
    "摘要": 1.0,
    "发明内容": 0.95,
    "实用新型内容": 0.95,
    "权利要求": 0.9,
    "背景技术": 0.85,
    "技术领域": 0.75,
    "具体实施方式": 0.65,
}


@dataclass(frozen=True)
class RetrievalPostprocessConfig:
    """Configuration for source filtering and lightweight reranking."""

    max_chunks_per_patent: int = 2
    min_score: float | None = None
    section_priority: dict[str, float] = field(
        default_factory=lambda: DEFAULT_SECTION_PRIORITY.copy()
    )
    section_weight: float = 0.08


def postprocess_hits(
    hits: list[SearchHit],
    *,
    top_k: int,
    config: RetrievalPostprocessConfig | None = None,
) -> list[SearchHit]:
    """Filter and rerank hits before they are used as RAG evidence."""

    if top_k <= 0:
        return []

    active_config = config or RetrievalPostprocessConfig()
    filtered_hits = _filter_by_score(hits, active_config.min_score)
    reranked_hits = _rerank_by_section_priority(filtered_hits, active_config)
    return _limit_chunks_per_patent(
        reranked_hits,
        top_k=top_k,
        max_chunks_per_patent=active_config.max_chunks_per_patent,
    )


def _filter_by_score(hits: list[SearchHit], min_score: float | None) -> list[SearchHit]:
    if min_score is None:
        return hits
    return [hit for hit in hits if hit.score >= min_score]


def _rerank_by_section_priority(
    hits: list[SearchHit],
    config: RetrievalPostprocessConfig,
) -> list[SearchHit]:
    scored_hits = [
        (_postprocess_score(hit, config), hit)
        for hit in hits
    ]
    scored_hits.sort(key=lambda item: item[0], reverse=True)
    return [
        hit.model_copy(update={"score": round(score, 4)})
        for score, hit in scored_hits
    ]


def _postprocess_score(hit: SearchHit, config: RetrievalPostprocessConfig) -> float:
    priority = config.section_priority.get(hit.section, 0.7)
    return hit.score + config.section_weight * priority


def _limit_chunks_per_patent(
    hits: list[SearchHit],
    *,
    top_k: int,
    max_chunks_per_patent: int,
) -> list[SearchHit]:
    if max_chunks_per_patent <= 0:
        return hits[:top_k]

    selected: list[SearchHit] = []
    patent_counts: defaultdict[str, int] = defaultdict(int)

    for hit in hits:
        if patent_counts[hit.patent_id] < max_chunks_per_patent:
            selected.append(hit)
            patent_counts[hit.patent_id] += 1
        if len(selected) >= top_k:
            return selected
    return selected
