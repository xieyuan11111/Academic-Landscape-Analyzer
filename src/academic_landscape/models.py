"""Domain contracts for the first academic-landscape vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "char_count": self.char_count,
            "text": self.text,
        }


@dataclass(frozen=True)
class DocumentRecord:
    artifact_id: str
    paper_id: str
    relative_path: str
    filename: str
    sha256: str
    size_bytes: int
    status: str
    duplicate_of: str | None = None
    parse_error: str | None = None
    page_count: int = 0
    text_char_count: int = 0
    quality_flags: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)
    pages: tuple[PageText, ...] = ()
    archive_path: str | None = None

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "paper_id": self.paper_id,
            "relative_path": self.relative_path,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "duplicate_of": self.duplicate_of,
            "parse_error": self.parse_error,
            "page_count": self.page_count,
            "text_char_count": self.text_char_count,
            "quality_flags": list(self.quality_flags),
            "metadata": dict(self.metadata),
            "pages_file": f"pages/{self.artifact_id}.json"
            if self.pages
            else None,
            "artifact_file": f"artifacts/{self.artifact_id}.pdf",
        }


@dataclass(frozen=True)
class CorpusSnapshot:
    snapshot_id: str
    created_at: str
    input_root: str
    documents: tuple[DocumentRecord, ...]
    schema_version: str = "0.1"

    @property
    def included_documents(self) -> tuple[DocumentRecord, ...]:
        return tuple(document for document in self.documents if document.status == "parsed")

    @property
    def unique_paper_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(document.paper_id for document in self.documents))

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for document in self.documents:
            counts[document.status] = counts.get(document.status, 0) + 1
        return dict(sorted(counts.items()))

    def canonical_identity(self) -> list[dict[str, Any]]:
        return [
            {
                "artifact_id": document.artifact_id,
                "paper_id": document.paper_id,
                "relative_path": document.relative_path,
                "sha256": document.sha256,
                "size_bytes": document.size_bytes,
                "status": document.status,
                "duplicate_of": document.duplicate_of,
                "parse_error": document.parse_error,
                "page_count": document.page_count,
                "text_char_count": document.text_char_count,
                "quality_flags": list(document.quality_flags),
                "metadata": dict(sorted(document.metadata.items())),
            }
            for document in self.documents
        ]

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "input_root": self.input_root,
            "corpus": {
                "document_count": len(self.documents),
                "included_document_count": len(self.included_documents),
                "unique_paper_count": len(self.unique_paper_ids),
                "status_counts": self.status_counts(),
            },
            "documents": [document.manifest_dict() for document in self.documents],
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
