"""Markdown patent file reading."""

from collections.abc import Iterable
from pathlib import Path


def iter_markdown_files(raw_dir: Path) -> list[Path]:
    """Return markdown files in a stable order."""

    return sorted(path for path in raw_dir.glob("*.md") if path.is_file())


def read_markdown(path: Path, encoding: str = "utf-8") -> str:
    """Read a markdown patent file with an explicit encoding."""

    return path.read_text(encoding=encoding)


def read_markdown_batch(paths: Iterable[Path], encoding: str = "utf-8") -> dict[Path, str]:
    """Read multiple markdown files."""

    return {path: read_markdown(path, encoding=encoding) for path in paths}

