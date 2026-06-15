"""Evaluate keyword, vector, and hybrid retrieval over a small question set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from patent_rag.evaluation import RetrievalCase, evaluate_retrieval_cases  # noqa: E402
from patent_rag.ingestion.jsonl import read_jsonl  # noqa: E402
from patent_rag.retrieval import (  # noqa: E402
    RetrievalPostprocessConfig,
    SearchHit,
    postprocess_hits,
    search_chunks,
    search_hybrid_chunks,
    search_vector_chunks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate patent retrieval quality.")
    parser.add_argument("--evals", type=Path, default=Path("evals") / "rag_questions.jsonl")
    parser.add_argument("--chunks", type=Path, default=Path("data") / "processed" / "chunks.jsonl")
    parser.add_argument("--index", type=Path, default=Path("data") / "indexes" / "chroma")
    parser.add_argument("--collection", default="patent_chunks")
    parser.add_argument("--modes", default="keyword,vector,hybrid")
    parser.add_argument("--top-k-values", default="1,3,5")
    parser.add_argument("--embedding-provider", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dimension", type=int, default=None)
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--max-chunks-per-patent", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    cases = read_jsonl(args.evals, RetrievalCase)
    top_k_values = _parse_int_list(args.top_k_values)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    reports = {}
    for mode in modes:
        retrieve = _build_retrieve_fn(
            mode=mode,
            chunks_path=args.chunks,
            index_path=args.index,
            collection_name=args.collection,
            embedding_provider=args.embedding_provider,
            embedding_model_name=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
            postprocess=args.postprocess,
            candidate_multiplier=args.candidate_multiplier,
            postprocess_config=RetrievalPostprocessConfig(
                max_chunks_per_patent=args.max_chunks_per_patent,
                min_score=args.min_score,
            ),
        )
        reports[mode] = evaluate_retrieval_cases(
            cases,
            retrieve,
            top_k_values=top_k_values,
        ).model_dump(mode="json")

    print(json.dumps({"case_count": len(cases), "reports": reports}, ensure_ascii=False, indent=2))
    return 0


def _build_retrieve_fn(
    *,
    mode: str,
    chunks_path: Path,
    index_path: Path,
    collection_name: str,
    embedding_provider: str | None,
    embedding_model_name: str | None,
    embedding_dimension: int | None,
    postprocess: bool,
    candidate_multiplier: int,
    postprocess_config: RetrievalPostprocessConfig,
):
    def retrieve(question: str, top_k: int) -> list[SearchHit]:
        candidate_k = top_k * max(candidate_multiplier, 1) if postprocess else top_k
        if mode == "keyword":
            hits = search_chunks(chunks_path, question, top_k=candidate_k)
        elif mode == "vector":
            hits = search_vector_chunks(
                index_path,
                question,
                top_k=candidate_k,
                collection_name=collection_name,
                embedding_provider=embedding_provider,
                embedding_model_name=embedding_model_name,
                embedding_dimension=embedding_dimension,
            )
        elif mode == "hybrid":
            hits = search_hybrid_chunks(
                chunks_path,
                index_path,
                question,
                top_k=candidate_k,
                collection_name=collection_name,
                embedding_provider=embedding_provider,
                embedding_model_name=embedding_model_name,
                embedding_dimension=embedding_dimension,
            )
        else:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        if postprocess:
            return postprocess_hits(hits, top_k=top_k, config=postprocess_config)
        return hits[:top_k]

    return retrieve


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
