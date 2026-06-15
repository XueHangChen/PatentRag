"""Ingestion pipeline for patent markdown documents."""

from dataclasses import dataclass
from pathlib import Path

from patent_rag.domain import PatentDocument
from patent_rag.ingestion.jsonl import write_jsonl
from patent_rag.ingestion.parser import parse_patent_markdown
from patent_rag.ingestion.reader import iter_markdown_files, read_markdown


@dataclass(frozen=True)
class IngestionResult:
    """Summary of an ingestion run."""

    source_count: int
    parsed_count: int
    output_path: Path
    errors: dict[str, str]


def parse_patent_directory(raw_dir: Path, encoding: str = "utf-8") -> tuple[list[PatentDocument], dict[str, str]]:
    """Parse all markdown patent files from a directory."""

    documents: list[PatentDocument] = []
    errors: dict[str, str] = {}

    for path in iter_markdown_files(raw_dir):
        try:
            text = read_markdown(path, encoding=encoding)
            documents.append(parse_patent_markdown(path, text))
        except Exception as exc:  # noqa: BLE001 - keep batch ingestion resilient.
            errors[str(path)] = str(exc)

    return documents, errors


def ingest_patent_directory(
    raw_dir: Path,
    output_path: Path,
    encoding: str = "utf-8",
) -> IngestionResult:
    """Parse markdown patents and write structured records to JSONL."""

    source_count = len(iter_markdown_files(raw_dir))
    documents, errors = parse_patent_directory(raw_dir, encoding=encoding)
    parsed_count = write_jsonl(documents, output_path)

    return IngestionResult(
        source_count=source_count,
        parsed_count=parsed_count,
        output_path=output_path,
        errors=errors,
    )

