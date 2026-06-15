"""A lightweight keyword/BM25 search implementation for patent chunks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from pydantic import BaseModel

from patent_rag.domain import PatentChunk


_ALNUM_PATTERN = re.compile(r"[A-Za-z0-9]+(?:/[A-Za-z0-9]+)?")
_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


class SearchHit(BaseModel):
    """A retrieved chunk with display metadata."""

    chunk_id: str
    patent_id: str
    title: str
    section: str
    claim_number: int | None = None
    score: float
    snippet: str
    source_file: str


@dataclass(frozen=True)
class _IndexedChunk:
    chunk: PatentChunk
    token_counts: Counter[str]
    token_count: int


class KeywordSearchIndex:
    """Small in-memory BM25 index over patent chunks."""

    def __init__(self, chunks: list[PatentChunk]) -> None:
        self._chunks = chunks
        self._indexed_chunks = [
            _IndexedChunk(
                chunk=chunk,
                token_counts=Counter(tokenize(chunk.text)),
                token_count=len(tokenize(chunk.text)),
            )
            for chunk in chunks
        ]
        self._document_frequency = self._build_document_frequency()
        self._avg_doc_len = self._build_average_doc_len()

    def search(self, query: str, top_k: int = 8) -> list[SearchHit]:
        """Search chunks by keyword relevance."""

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_token_counts = Counter(query_tokens)
        query_compact = _compact(query)
        scored_hits: list[tuple[float, PatentChunk]] = []

        for indexed_chunk in self._indexed_chunks:
            score = self._bm25_score(indexed_chunk, query_token_counts)
            score += self._phrase_bonus(indexed_chunk.chunk, query_compact)
            if score > 0:
                scored_hits.append((score, indexed_chunk.chunk))

        scored_hits.sort(key=lambda item: item[0], reverse=True)
        return [
            _to_search_hit(chunk, score, query)
            for score, chunk in scored_hits[: max(top_k, 0)]
        ]

    def _build_document_frequency(self) -> Counter[str]:
        document_frequency: Counter[str] = Counter()
        for indexed_chunk in self._indexed_chunks:
            document_frequency.update(indexed_chunk.token_counts.keys())
        return document_frequency

    def _build_average_doc_len(self) -> float:
        if not self._indexed_chunks:
            return 0.0
        return sum(indexed.token_count for indexed in self._indexed_chunks) / len(self._indexed_chunks)

    def _bm25_score(self, indexed_chunk: _IndexedChunk, query_token_counts: Counter[str]) -> float:
        k1 = 1.5
        b = 0.75
        total_documents = len(self._indexed_chunks)
        if total_documents == 0 or indexed_chunk.token_count == 0:
            return 0.0

        score = 0.0
        for token, query_weight in query_token_counts.items():
            term_frequency = indexed_chunk.token_counts.get(token, 0)
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequency.get(token, 0)
            idf = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = term_frequency + k1 * (
                1 - b + b * indexed_chunk.token_count / max(self._avg_doc_len, 1.0)
            )
            score += query_weight * idf * (term_frequency * (k1 + 1)) / denominator
        return score

    @staticmethod
    def _phrase_bonus(chunk: PatentChunk, query_compact: str) -> float:
        if len(query_compact) < 2:
            return 0.0
        chunk_compact = _compact(chunk.text)
        metadata_compact = _compact(" ".join(chunk.metadata.values()))
        if query_compact in chunk_compact:
            return 6.0
        if query_compact in metadata_compact:
            return 4.0
        return 0.0


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English patent text for lightweight search."""

    normalized = text.lower()
    tokens = _ALNUM_PATTERN.findall(normalized)
    cjk_chars = _CJK_PATTERN.findall(normalized)

    tokens.extend(cjk_chars)
    tokens.extend("".join(cjk_chars[index : index + 2]) for index in range(len(cjk_chars) - 1))
    tokens.extend("".join(cjk_chars[index : index + 3]) for index in range(len(cjk_chars) - 2))
    return [token for token in tokens if token.strip()]


def _to_search_hit(chunk: PatentChunk, score: float, query: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.chunk_id,
        patent_id=chunk.patent_id,
        title=chunk.metadata.get("title", ""),
        section=chunk.section,
        claim_number=chunk.claim_number,
        score=round(score, 4),
        snippet=_make_snippet(chunk.text, query),
        source_file=str(chunk.source_file),
    )


def _make_snippet(text: str, query: str, max_length: int = 180) -> str:
    compact_query = _compact(query)
    compact_text = _compact(text)
    start = compact_text.find(compact_query) if compact_query else -1

    if start == -1:
        return text[:max_length]

    start = max(start - 40, 0)
    end = min(start + max_length, len(text))
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())

