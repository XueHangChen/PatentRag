"""Build retrieval chunks from structured patent documents."""

from __future__ import annotations

import hashlib

from patent_rag.domain import PatentChunk, PatentDocument


def build_chunks_for_document(document: PatentDocument) -> list[PatentChunk]:
    """Build retrieval chunks for a single patent document."""

    chunks: list[PatentChunk] = []
    metadata = document.metadata
    base_metadata = {
        "title": metadata.title or "",
        "patent_type": metadata.patent_type.value,
        "publication_number": metadata.publication_number or "",
        "application_number": metadata.application_number or "",
        "applicants": ";".join(metadata.applicants),
        "inventors": ";".join(metadata.inventors),
        "ipc_classes": ";".join(metadata.ipc_classes),
    }

    if document.abstract:
        title_prefix = f"专利名称：{metadata.title}\n" if metadata.title else ""
        chunks.append(
            _make_chunk(
                patent_id=metadata.patent_id,
                section="摘要",
                text=f"{title_prefix}摘要：{document.abstract}",
                source_file=metadata.source_file,
                metadata=base_metadata,
            )
        )

    for claim in document.claims:
        chunks.append(
            _make_chunk(
                patent_id=metadata.patent_id,
                section="权利要求",
                text=f"权利要求{claim.claim_number}：{claim.text}",
                source_file=metadata.source_file,
                metadata=base_metadata,
                claim_number=claim.claim_number,
            )
        )

    for section in document.sections:
        chunks.append(
            _make_chunk(
                patent_id=metadata.patent_id,
                section=section.name,
                text=f"{section.name}：{section.text}",
                source_file=metadata.source_file,
                metadata=base_metadata,
            )
        )

    return chunks


def build_chunks(documents: list[PatentDocument]) -> list[PatentChunk]:
    """Build retrieval chunks for multiple patent documents."""

    chunks: list[PatentChunk] = []
    for document in documents:
        chunks.extend(build_chunks_for_document(document))
    return chunks


def _make_chunk(
    patent_id: str,
    section: str,
    text: str,
    source_file,
    metadata: dict[str, str],
    claim_number: int | None = None,
) -> PatentChunk:
    chunk_id = _stable_chunk_id(patent_id, section, text, claim_number)
    return PatentChunk(
        chunk_id=chunk_id,
        patent_id=patent_id,
        text=text,
        section=section,
        claim_number=claim_number,
        source_file=source_file,
        metadata=metadata,
    )


def _stable_chunk_id(
    patent_id: str,
    section: str,
    text: str,
    claim_number: int | None,
) -> str:
    seed = f"{patent_id}|{section}|{claim_number or ''}|{text[:120]}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"{patent_id}:{section}:{claim_number or 'section'}:{digest}"

