from pathlib import Path

from patent_rag.ingestion.pipeline import parse_patent_directory
from patent_rag.retrieval import KeywordSearchIndex, build_chunks


def test_build_chunks_from_current_patents() -> None:
    documents, errors = parse_patent_directory(Path("patant"))

    assert errors == {}

    chunks = build_chunks(documents)

    assert len(chunks) > len(documents)
    assert any(chunk.section == "摘要" for chunk in chunks)
    assert any(chunk.section == "权利要求" for chunk in chunks)
    assert any(chunk.section == "背景技术" for chunk in chunks)
    assert all(chunk.patent_id for chunk in chunks)
    assert all(chunk.metadata.get("title") for chunk in chunks)


def test_keyword_search_finds_liquid_nitrogen_patent() -> None:
    documents, _ = parse_patent_directory(Path("patant"))
    index = KeywordSearchIndex(build_chunks(documents))

    hits = index.search("液氮罐运输固定", top_k=5)

    assert hits
    assert hits[0].patent_id == "CN206539886U"
    assert hits[0].title == "车载液氮罐固定架"


def test_keyword_search_finds_heart_stent_patent() -> None:
    documents, _ = parse_patent_directory(Path("patant"))
    index = KeywordSearchIndex(build_chunks(documents))

    hits = index.search("心脏支架材料 生物降解", top_k=5)

    assert hits
    assert any(hit.patent_id == "CN109395173A" for hit in hits[:3])
