"""Quality report generation for processed patent JSONL files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from pydantic import BaseModel, Field

from patent_rag.domain import PatentDocument


_TITLE_BLOCKLIST = (
    "中华人民共和国国家知识产权局",
    "国家知识产权局",
    "技术领域",
    "背景技术",
    "发明内容",
    "实用新型内容",
    "附图说明",
    "具体实施方式",
)
_IPC_PATTERN = re.compile(r"^[A-Z]\d{2}[A-Z]\d+/\d+$")


class QualityIssue(BaseModel):
    """A single data quality issue."""

    severity: str
    code: str
    patent_id: str | None = None
    source_file: str | None = None
    field: str
    message: str
    value: str | None = None


class PatentQualityReport(BaseModel):
    """Summary and issue list for a processed patent JSONL file."""

    input_path: Path
    total_records: int
    valid_records: int
    parse_errors: int
    issue_count: int
    error_count: int
    warning_count: int
    claim_count_min: int = 0
    claim_count_max: int = 0
    claim_count_avg: float = 0.0
    section_count_min: int = 0
    section_count_max: int = 0
    section_count_avg: float = 0.0
    patent_type_counts: dict[str, int] = Field(default_factory=dict)
    issues: list[QualityIssue] = Field(default_factory=list)


def build_quality_report(input_path: Path) -> PatentQualityReport:
    """Build a quality report for a processed patents JSONL file."""

    documents: list[PatentDocument] = []
    issues: list[QualityIssue] = []
    patent_ids: list[str] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                document = PatentDocument.model_validate_json(stripped)
            except Exception as exc:  # noqa: BLE001 - report bad records instead of stopping.
                issues.append(
                    QualityIssue(
                        severity="error",
                        code="jsonl_parse_error",
                        field="record",
                        message=f"Line {line_number} cannot be parsed as PatentDocument.",
                        value=str(exc),
                    )
                )
                continue

            documents.append(document)
            patent_ids.append(document.metadata.patent_id)
            issues.extend(_inspect_document(document))

    duplicate_ids = {
        patent_id for patent_id, count in Counter(patent_ids).items() if patent_id and count > 1
    }
    for patent_id in sorted(duplicate_ids):
        issues.append(
            QualityIssue(
                severity="error",
                code="duplicate_patent_id",
                patent_id=patent_id,
                field="metadata.patent_id",
                message="Patent id appears more than once.",
                value=patent_id,
            )
        )

    claim_counts = [len(document.claims) for document in documents]
    section_counts = [len(document.sections) for document in documents]
    patent_type_counts = Counter(str(document.metadata.patent_type.value) for document in documents)
    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    parse_errors = sum(1 for issue in issues if issue.code == "jsonl_parse_error")

    return PatentQualityReport(
        input_path=input_path,
        total_records=len(documents) + parse_errors,
        valid_records=len(documents),
        parse_errors=parse_errors,
        issue_count=len(issues),
        error_count=error_count,
        warning_count=warning_count,
        claim_count_min=min(claim_counts, default=0),
        claim_count_max=max(claim_counts, default=0),
        claim_count_avg=_average(claim_counts),
        section_count_min=min(section_counts, default=0),
        section_count_max=max(section_counts, default=0),
        section_count_avg=_average(section_counts),
        patent_type_counts=dict(sorted(patent_type_counts.items())),
        issues=issues,
    )


def render_quality_report(report: PatentQualityReport) -> str:
    """Render a human-readable report."""

    lines = [
        "Patent Data Quality Report",
        f"input_path: {report.input_path}",
        f"total_records: {report.total_records}",
        f"valid_records: {report.valid_records}",
        f"parse_errors: {report.parse_errors}",
        f"issue_count: {report.issue_count}",
        f"error_count: {report.error_count}",
        f"warning_count: {report.warning_count}",
        (
            "claims: "
            f"min={report.claim_count_min}, "
            f"max={report.claim_count_max}, "
            f"avg={report.claim_count_avg:.2f}"
        ),
        (
            "sections: "
            f"min={report.section_count_min}, "
            f"max={report.section_count_max}, "
            f"avg={report.section_count_avg:.2f}"
        ),
        "patent_type_counts:",
    ]

    if report.patent_type_counts:
        for patent_type, count in report.patent_type_counts.items():
            lines.append(f"  - {patent_type}: {count}")
    else:
        lines.append("  - none")

    lines.append("issues:")
    if report.issues:
        for issue in report.issues:
            patent_id = issue.patent_id or "-"
            source_file = issue.source_file or "-"
            value = f" value={issue.value}" if issue.value else ""
            lines.append(
                f"  - [{issue.severity}] {issue.code} "
                f"patent_id={patent_id} field={issue.field} source={source_file}: "
                f"{issue.message}{value}"
            )
    else:
        lines.append("  - none")

    return "\n".join(lines)


def _inspect_document(document: PatentDocument) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    metadata = document.metadata
    patent_id = metadata.patent_id
    source_file = str(metadata.source_file)

    if not metadata.patent_id:
        issues.append(_issue(document, "error", "missing_patent_id", "metadata.patent_id", "Missing patent id."))

    if not metadata.title:
        issues.append(_issue(document, "error", "missing_title", "metadata.title", "Missing title."))
    elif _is_suspicious_title(metadata.title):
        issues.append(
            _issue(
                document,
                "error",
                "suspicious_title",
                "metadata.title",
                "Title looks like a document header or section heading.",
                metadata.title,
            )
        )

    if not document.abstract:
        issues.append(_issue(document, "warning", "missing_abstract", "abstract", "Missing abstract."))

    if not document.claims:
        issues.append(_issue(document, "error", "missing_claims", "claims", "Missing claims."))

    if not document.sections:
        issues.append(_issue(document, "warning", "missing_sections", "sections", "Missing sections."))

    for applicant in metadata.applicants:
        if "地址" in applicant:
            issues.append(
                _issue(
                    document,
                    "warning",
                    "applicant_contains_address",
                    "metadata.applicants",
                    "Applicant value appears to include an address.",
                    applicant,
                )
            )

    if metadata.agency and "代理人" in metadata.agency:
        issues.append(
            _issue(
                document,
                "warning",
                "agency_contains_agent",
                "metadata.agency",
                "Agency value appears to include agent text.",
                metadata.agency,
            )
        )

    for ipc_class in metadata.ipc_classes:
        if not _IPC_PATTERN.match(ipc_class):
            issues.append(
                _issue(
                    document,
                    "warning",
                    "invalid_ipc_format",
                    "metadata.ipc_classes",
                    "IPC class does not match the normalized format.",
                    ipc_class,
                )
            )

    if metadata.title and document.abstract and metadata.title == document.abstract:
        issues.append(
            _issue(
                document,
                "warning",
                "title_equals_abstract",
                "metadata.title",
                "Title is identical to abstract.",
                metadata.title,
            )
        )

    if not source_file:
        issues.append(
            QualityIssue(
                severity="warning",
                code="missing_source_file",
                patent_id=patent_id,
                field="metadata.source_file",
                message="Missing source file.",
            )
        )

    return issues


def _issue(
    document: PatentDocument,
    severity: str,
    code: str,
    field: str,
    message: str,
    value: str | None = None,
) -> QualityIssue:
    return QualityIssue(
        severity=severity,
        code=code,
        patent_id=document.metadata.patent_id,
        source_file=str(document.metadata.source_file),
        field=field,
        message=message,
        value=value,
    )


def _is_suspicious_title(title: str) -> bool:
    if title.startswith("("):
        return True
    return any(keyword in title for keyword in _TITLE_BLOCKLIST)


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

