"""Text normalization helpers for OCR/PDF-converted patent markdown."""

import re


_CJK_SPACE_PATTERN = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")


def normalize_newlines(text: str) -> str:
    """Normalize line endings without changing semantic content."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_inline_spaces(text: str) -> str:
    """Clean common OCR spacing artifacts inside a single field."""

    text = _CJK_SPACE_PATTERN.sub("", text)
    text = re.sub(r"\s+([,，。；;:：])", r"\1", text)
    text = re.sub(r"([（(])\s+", r"\1", text)
    text = re.sub(r"\s+([）)])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_date(text: str) -> str:
    """Normalize patent date strings such as ``2017 .03 .21``."""

    return re.sub(r"\s*\.\s*", ".", text.strip())


def normalize_identifier(text: str) -> str:
    """Normalize application/publication identifiers while preserving dots."""

    return re.sub(r"\s+", "", text.strip())


def clean_multiline_value(text: str) -> str:
    """Collapse a multiline field into a single readable value."""

    lines = []
    for raw_line in normalize_newlines(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"^\(\d+\)\s*", "", line)
        lines.append(line)
    return normalize_inline_spaces(" ".join(lines))


def remove_image_refs(text: str) -> str:
    """Remove markdown image references from text used for extraction."""

    return re.sub(r"!\[[^\]]*]\([^)]+\)", "", text)

