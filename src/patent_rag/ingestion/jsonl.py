"""JSONL persistence helpers."""

from collections.abc import Iterable
import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(records: Iterable[BaseModel], output_path: Path) -> int:
    """Write Pydantic records to JSONL and return the record count."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            payload = record.model_dump(mode="json", exclude_none=True)
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1

    return count


def read_jsonl(input_path: Path, model_type: type[T]) -> list[T]:
    """Read JSONL records into Pydantic models."""

    records: list[T] = []
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(model_type.model_validate_json(stripped))
    return records

