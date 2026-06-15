"""Rule-based parser for Chinese patent markdown documents."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from patent_rag.domain import (
    EvidenceSpan,
    PatentClaim,
    PatentDocument,
    PatentMetadata,
    PatentSection,
)
from patent_rag.domain.schemas import PatentType
from patent_rag.ingestion.normalization import (
    clean_multiline_value,
    normalize_date,
    normalize_identifier,
    normalize_inline_spaces,
    normalize_newlines,
    remove_image_refs,
)


_IMAGE_REF_PATTERN = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
_PUBLICATION_PATTERN = re.compile(r"(CN\s*\d+(?:\s*\d+)?\s*[A-Z])\s+(\d{4}\s*\.\s*\d{2}\s*\.\s*\d{2})")
_APPLICATION_NUMBER_PATTERN = re.compile(r"\(\s*21\s*\)\s*申请号\s*([^\n]+)")
_APPLICATION_DATE_PATTERN = re.compile(r"\(\s*22\s*\)\s*申请日\s*([^\n]+)")
_IPC_PATTERN = re.compile(r"\b([A-Z]\d{2}[A-Z]\s*\d+/\d+)\b")
_SECTION_HEADING_PATTERN = re.compile(r"^(#{2,4})\s+(.+?)\s*$", re.MULTILINE)
_CLAIM_START_PATTERN = re.compile(r"(?m)^\s*(\d+)\s*\.\s*")


@dataclass(frozen=True)
class ParseWarning:
    """A non-fatal parser warning."""

    source_file: Path
    message: str


def parse_patent_markdown(source_file: Path, text: str) -> PatentDocument:
    """Parse a markdown patent document into a structured domain object."""

    normalized_text = normalize_newlines(text)
    image_refs = _IMAGE_REF_PATTERN.findall(normalized_text)
    extraction_text = remove_image_refs(normalized_text)

    raw_patent_type, patent_type = _extract_patent_type(extraction_text)
    publication_number, publication_date = _extract_publication(extraction_text)
    application_number = _extract_application_number(extraction_text)
    application_date = _extract_application_date(extraction_text)
    title = _extract_title(extraction_text)
    abstract = _extract_abstract(extraction_text)
    ipc_classes = _extract_ipc_classes(extraction_text)
    applicants = _extract_people_or_orgs(extraction_text, ("申请人", "专利权人"))
    inventors = _extract_people_or_orgs(extraction_text, ("发明人",))
    agency = _extract_agency(extraction_text)
    agents = _extract_agents(extraction_text)
    claims = _extract_claims(source_file, extraction_text)
    sections = _extract_sections(source_file, extraction_text, title=title)

    patent_id = publication_number or application_number or source_file.stem

    metadata = PatentMetadata(
        patent_id=patent_id,
        title=title,
        patent_type=patent_type,
        raw_patent_type=raw_patent_type,
        publication_number=publication_number,
        application_number=application_number,
        application_date=application_date,
        publication_date=publication_date,
        applicants=applicants,
        inventors=inventors,
        agency=agency,
        agents=agents,
        ipc_classes=ipc_classes,
        source_file=source_file,
    )

    return PatentDocument(
        metadata=metadata,
        abstract=abstract,
        claims=claims,
        sections=sections,
        image_refs=image_refs,
        raw_text=normalized_text,
    )


def _extract_patent_type(text: str) -> tuple[str | None, PatentType]:
    match = re.search(r"^\s*#\s*\(12\)\s*(.+?)\s*$", text, flags=re.MULTILINE)
    raw_type = normalize_inline_spaces(match.group(1)) if match else None

    if raw_type and "发明" in raw_type:
        return raw_type, PatentType.INVENTION_APPLICATION
    if raw_type and "实用新型" in raw_type:
        return raw_type, PatentType.UTILITY_MODEL
    return raw_type, PatentType.UNKNOWN


def _extract_publication(text: str) -> tuple[str | None, str | None]:
    match = _PUBLICATION_PATTERN.search(text)
    if not match:
        return None, None

    publication_number = normalize_identifier(match.group(1))
    publication_date = normalize_date(match.group(2))
    return publication_number, publication_date


def _extract_application_number(text: str) -> str | None:
    match = _APPLICATION_NUMBER_PATTERN.search(text)
    if not match:
        return None

    value = match.group(1).strip()
    value = value.split("\n", maxsplit=1)[0]
    return normalize_identifier(value)


def _extract_application_date(text: str) -> str | None:
    match = _APPLICATION_DATE_PATTERN.search(text)
    if not match:
        return None

    value = match.group(1).strip()
    value = value.split("\n", maxsplit=1)[0]
    return normalize_date(value)


def _extract_title(text: str) -> str | None:
    marker = re.search(r"^\s*(?:#{1,6}\s*)?\(54\).+?名称\s*$", text, flags=re.MULTILINE)
    if marker:
        for line in text[marker.end() :].split("\n"):
            cleaned = normalize_inline_spaces(line)
            if cleaned and not cleaned.startswith(("(", "!", "#")):
                return cleaned.lstrip("#").strip()

    for fallback in re.finditer(r"^\s*#{2,4}\s+(.+?)\s*$", text, flags=re.MULTILINE):
        candidate = normalize_inline_spaces(fallback.group(1))
        if _is_plausible_patent_title(candidate):
            return candidate
    return None


def _is_plausible_patent_title(candidate: str) -> bool:
    if not candidate:
        return False
    if candidate.startswith("("):
        return False
    invalid_keywords = (
        "中华人民共和国国家知识产权局",
        "技术领域",
        "背景技术",
        "发明内容",
        "实用新型内容",
        "附图说明",
        "具体实施方式",
    )
    return not any(keyword in candidate for keyword in invalid_keywords)


def _extract_abstract(text: str) -> str | None:
    marker = re.search(r"^\s*\(57\)\s*摘要\s*$", text, flags=re.MULTILINE)
    if not marker:
        return None

    tail = text[marker.end() :]
    end_match = re.search(r"权利要求书\s*\d*页", tail)
    if end_match:
        raw_abstract = tail[: end_match.start()]
    else:
        fallback_end = re.search(r"(?m)^\s*(?:CN\s+\d+|1\s*\.)", tail)
        raw_abstract = tail[: fallback_end.start()] if fallback_end else tail

    return clean_multiline_value(raw_abstract) or None


def _extract_ipc_classes(text: str) -> list[str]:
    values = []
    for match in _IPC_PATTERN.finditer(text):
        value = normalize_identifier(match.group(1))
        if value not in values:
            values.append(value)
    return values


def _extract_people_or_orgs(text: str, labels: tuple[str, ...]) -> list[str]:
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"(?ms)^\s*-\s*\(\s*(?:71|72|73)\s*\)\s*(?:{label_pattern})\s*(.+?)(?=^\s*-\s*\(\s*(?:72|73|74)\s*\)|^\s*\(51\)|^\s*####|\Z)"
    )
    match = pattern.search(text)
    if not match:
        return []

    value = clean_multiline_value(match.group(1))
    value = re.split(r"地址", value, maxsplit=1)[0]
    value = normalize_inline_spaces(value)
    return _split_names(value)


def _extract_agency(text: str) -> str | None:
    pattern = re.compile(
        r"(?ms)^\s*-\s*\(\s*74\s*\)\s*专利代理机构\s*(.+?)(?=^\s*\(51\)|^\s*####|\Z)"
    )
    match = pattern.search(text)
    if not match:
        return None

    value = clean_multiline_value(match.group(1))
    value = re.split(r"代理人", value, maxsplit=1)[0]
    return normalize_inline_spaces(value) or None


def _extract_agents(text: str) -> list[str]:
    matches = re.findall(r"代理人\s*([^\n]+)", text)
    agents: list[str] = []
    for match in matches:
        cleaned = normalize_inline_spaces(match)
        cleaned = re.sub(r"^\s*[:：]\s*", "", cleaned)
        if cleaned:
            agents.extend(_split_names(cleaned))
    return _dedupe(agents)


def _extract_claims(source_file: Path, text: str) -> list[PatentClaim]:
    first_claim = _CLAIM_START_PATTERN.search(text)
    if not first_claim:
        return []

    claims_tail = text[first_claim.start() :]
    section_start = _SECTION_HEADING_PATTERN.search(claims_tail)
    claims_text = claims_tail[: section_start.start()] if section_start else claims_tail
    starts = list(_CLAIM_START_PATTERN.finditer(claims_text))

    claims: list[PatentClaim] = []
    for index, start in enumerate(starts):
        next_start = starts[index + 1].start() if index + 1 < len(starts) else len(claims_text)
        raw_claim = claims_text[start.end() : next_start]
        claim_number = int(start.group(1))
        claim_text = _clean_body_text(raw_claim)
        if not claim_text:
            continue
        claims.append(
            PatentClaim(
                claim_number=claim_number,
                text=claim_text,
                evidence=EvidenceSpan(
                    source_file=source_file,
                    section="权利要求",
                    text=claim_text,
                ),
            )
        )

    return claims


def _extract_sections(source_file: Path, text: str, title: str | None) -> list[PatentSection]:
    headings = list(_SECTION_HEADING_PATTERN.finditer(text))
    if not headings:
        return []

    sections: list[PatentSection] = []
    known_section_keywords = (
        "技术领域",
        "背景技术",
        "发明内容",
        "实用新型内容",
        "附图说明",
        "具体实施方式",
    )

    for index, heading in enumerate(headings):
        name = normalize_inline_spaces(heading.group(2)).lstrip("#").strip()
        if not name or name == title:
            continue
        if name.startswith("("):
            continue
        if not any(keyword in name for keyword in known_section_keywords):
            continue

        next_heading_start = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = _clean_body_text(text[heading.end() : next_heading_start])
        if not body:
            continue
        sections.append(
            PatentSection(
                name=name,
                text=body,
                level=len(heading.group(1)),
                evidence=EvidenceSpan(
                    source_file=source_file,
                    section=name,
                    text=body,
                ),
            )
        )

    return sections


def _clean_body_text(text: str) -> str:
    text = remove_image_refs(text)
    text = re.sub(r"(?m)^\s*CN\s+\d+.*$", "", text)
    text = re.sub(r"(?m)^\s*第\s*\d+\s*页.*$", "", text)
    return clean_multiline_value(text)


def _split_names(value: str) -> list[str]:
    if not value:
        return []

    parts = re.split(r"[、,，;；]\s*", value)
    return _dedupe(normalize_inline_spaces(part) for part in parts if normalize_inline_spaces(part))


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
