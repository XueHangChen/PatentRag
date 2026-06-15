from pathlib import Path

from patent_rag.ingestion.pipeline import ingest_patent_directory
from patent_rag.quality import build_quality_report, render_quality_report

PROCESSED_PATENTS_PATH = Path("data/processed/patents.jsonl")


def test_current_processed_patents_pass_quality_report() -> None:
    if not PROCESSED_PATENTS_PATH.exists():
        ingest_patent_directory(Path("patant"), PROCESSED_PATENTS_PATH)

    report = build_quality_report(PROCESSED_PATENTS_PATH)

    assert report.total_records == 20
    assert report.valid_records == 20
    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.claim_count_min > 0
    assert report.section_count_min > 0


def test_quality_report_flags_suspicious_title(tmp_path: Path) -> None:
    bad_jsonl = tmp_path / "bad_patents.jsonl"
    bad_jsonl.write_text(
        """
{"metadata":{"patent_id":"CN000000000A","title":"(19)中华人民共和国国家知识产权局","patent_type":"invention_application","source_file":"patant/bad.md"},"abstract":"摘要","claims":[{"claim_number":1,"text":"权利要求"}],"sections":[{"name":"技术领域","text":"正文"}]}
""".strip(),
        encoding="utf-8",
    )

    report = build_quality_report(bad_jsonl)

    assert report.error_count == 1
    assert report.issues[0].code == "suspicious_title"
    assert "suspicious_title" in render_quality_report(report)
