"""Safe archival of source PDF artifacts for immutable evidence runs."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .models import CorpusSnapshot, DocumentRecord


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_path(snapshot: CorpusSnapshot, document: DocumentRecord) -> Path:
    root = Path(snapshot.input_root).expanduser().resolve()
    candidate = (root / document.relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"artifact path escapes snapshot input root: {document.relative_path}")
    if not candidate.is_file():
        if document.archive_path:
            archived = Path(document.archive_path).expanduser().resolve()
            if archived.is_file():
                return archived
        raise FileNotFoundError(f"source PDF is missing: {candidate}")
    return candidate


def archive_snapshot(snapshot: CorpusSnapshot, archive_root: str | Path) -> dict[str, Path]:
    """Copy every source PDF to deterministic artifact_id paths and verify hashes."""
    destination_root = Path(archive_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    archived: dict[str, Path] = {}
    for document in snapshot.documents:
        destination = (destination_root / f"{document.artifact_id}.pdf").resolve()
        if not destination.is_relative_to(destination_root):
            raise ValueError(f"artifact destination escapes archive root: {destination}")
        if destination.exists():
            if _sha256(destination) != document.sha256:
                raise ValueError(f"archived artifact hash mismatch: {document.artifact_id}")
            archived[document.artifact_id] = destination
            continue
        source = _source_path(snapshot, document)
        shutil.copy2(source, destination)
        if _sha256(destination) != document.sha256:
            destination.unlink(missing_ok=True)
            raise ValueError(f"archived artifact hash mismatch: {document.artifact_id}")
        archived[document.artifact_id] = destination
    return archived
