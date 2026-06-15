"""Core domain schemas.

These models are intentionally storage-agnostic. Parsers, vector stores,
graph databases, and Agent tools should exchange these objects instead of
passing unstructured dictionaries around.
"""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class PatentType(str, Enum):
    """Known patent categories in the current corpus."""

    INVENTION_APPLICATION = "invention_application"
    UTILITY_MODEL = "utility_model"
    UNKNOWN = "unknown"


class EvidenceSpan(BaseModel):
    """A source-backed span used for traceability."""

    source_file: Path
    section: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    text: str


class PatentMetadata(BaseModel):
    """Structured metadata extracted from a patent document."""

    patent_id: str
    title: str | None = None
    patent_type: PatentType = PatentType.UNKNOWN
    raw_patent_type: str | None = None
    publication_number: str | None = None
    application_number: str | None = None
    application_date: str | None = None
    publication_date: str | None = None
    applicants: list[str] = Field(default_factory=list)
    inventors: list[str] = Field(default_factory=list)
    agency: str | None = None
    agents: list[str] = Field(default_factory=list)
    ipc_classes: list[str] = Field(default_factory=list)
    source_file: Path


class PatentClaim(BaseModel):
    """A single patent claim."""

    claim_number: int
    text: str
    evidence: EvidenceSpan | None = None


class PatentSection(BaseModel):
    """A named section from the patent specification."""

    name: str
    text: str
    level: int | None = None
    evidence: EvidenceSpan | None = None


class PatentDocument(BaseModel):
    """A fully parsed patent document."""

    metadata: PatentMetadata
    abstract: str | None = None
    claims: list[PatentClaim] = Field(default_factory=list)
    sections: list[PatentSection] = Field(default_factory=list)
    image_refs: list[str] = Field(default_factory=list)
    raw_text: str | None = None


class PatentChunk(BaseModel):
    """A retrieval unit derived from a patent document."""

    chunk_id: str
    patent_id: str
    text: str
    section: str
    claim_number: int | None = None
    source_file: Path
    metadata: dict[str, str] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """A knowledge graph node."""

    node_id: str
    label: str
    name: str
    properties: dict[str, str] = Field(default_factory=dict)
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class GraphEdge(BaseModel):
    """A knowledge graph relationship."""

    edge_id: str
    source_id: str
    target_id: str
    relation: str
    properties: dict[str, str] = Field(default_factory=dict)
    evidence: list[EvidenceSpan] = Field(default_factory=list)

