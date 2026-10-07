"""Deterministic page/segment chunks bound to a corpus snapshot."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .models import CorpusSnapshot


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    snapshot_id: str
    paper_id: str
    artifact_id: str
    page_number: int
    chunk_type: str
    ordinal: int
    text: str
    text_hash: str

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "snapshot_id": self.snapshot_id,
            "paper_id": self.paper_id,
            "artifact_id": self.artifact_id,
            "page_number": self.page_number,
            "chunk_type": self.chunk_type,
            "ordinal": self.ordinal,
            "text_hash": self.text_hash,
            "text": self.text,
        }


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _chunk_words(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start
        length = 0
        while end < len(words):
            added = len(words[end]) if end == start else len(words[end]) + 1
            if end > start and length + added > max_chars:
                break
            length += added
            end += 1
        if end == start:
            end = start + 1
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break

        overlap_start = end
        overlap_length = 0
        while overlap_start > start:
            candidate = len(words[overlap_start - 1])
            if overlap_start < end:
                candidate += 1
            if overlap_length + candidate > overlap_chars:
                break
            overlap_length += candidate
            overlap_start -= 1
        start = max(start + 1, overlap_start)
    return chunks


def _chunk_id(
    snapshot_id: str,
    artifact_id: str,
    page_number: int,
    ordinal: int,
    text_hash: str,
) -> str:
    value = "\0".join(
        [snapshot_id, artifact_id, str(page_number), "page", str(ordinal), text_hash]
    )
    return f"chunk_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:32]}"


def chunk_snapshot(
    snapshot: CorpusSnapshot,
    *,
    max_chars: int = 1800,
    overlap_chars: int = 180,
) -> list[Chunk]:
    """Create stable page-bound chunks from parsed documents only."""
    chunks: list[Chunk] = []
    for document in snapshot.included_documents:
        for page in sorted(document.pages, key=lambda item: item.page_number):
            text = _normalise_text(page.text)
            if not text:
                continue
            page_chunks = _chunk_words(text, max_chars, overlap_chars)
            for ordinal, chunk_text in enumerate(page_chunks):
                text_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(
                            snapshot.snapshot_id,
                            document.artifact_id,
                            page.page_number,
                            ordinal,
                            text_hash,
                        ),
                        snapshot_id=snapshot.snapshot_id,
                        paper_id=document.paper_id,
                        artifact_id=document.artifact_id,
                        page_number=page.page_number,
                        chunk_type="page",
                        ordinal=ordinal,
                        text=chunk_text,
                        text_hash=text_hash,
                    )
                )
    return chunks
