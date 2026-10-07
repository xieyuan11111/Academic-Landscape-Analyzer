from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import CorpusSnapshot, DocumentRecord, PageText


def _snapshot_from_manifest(snapshot_dir: Path, manifest: dict[str, Any]) -> CorpusSnapshot:
    documents: list[DocumentRecord] = []
    pages_root = snapshot_dir.resolve()
    for item in manifest.get("documents", []):
        pages: tuple[PageText, ...] = ()
        pages_file = item.get("pages_file")
        if pages_file:
            candidate = (snapshot_dir / pages_file).resolve()
            if pages_root not in candidate.parents and candidate != pages_root:
                raise ValueError(f"pages_file escapes snapshot directory: {pages_file}")
            if not candidate.exists():
                raise FileNotFoundError(f"Missing saved page evidence: {candidate}")
            page_payload = json.loads(candidate.read_text(encoding="utf-8"))
            pages = tuple(
                PageText(page_number=int(page["page_number"]), text=str(page.get("text", "")))
                for page in page_payload.get("pages", [])
            )
        documents.append(
            DocumentRecord(
                artifact_id=str(item["artifact_id"]),
                paper_id=str(item["paper_id"]),
                relative_path=str(item["relative_path"]),
                filename=str(item["filename"]),
                sha256=str(item["sha256"]),
                size_bytes=int(item["size_bytes"]),
                status=str(item["status"]),
                duplicate_of=item.get("duplicate_of"),
                parse_error=item.get("parse_error"),
                page_count=int(item.get("page_count", len(pages))),
                text_char_count=int(item.get("text_char_count", sum(len(page.text) for page in pages))),
                quality_flags=tuple(str(flag) for flag in item.get("quality_flags", [])),
                metadata={str(key): str(value) for key, value in item.get("metadata", {}).items()},
                pages=pages,
            )
        )
    return CorpusSnapshot(
        snapshot_id=str(manifest["snapshot_id"]),
        created_at=str(manifest["created_at"]),
        input_root=str(manifest.get("input_root", "")),
        documents=tuple(documents),
        schema_version=str(manifest.get("schema_version", "0.1")),
    )


def load_snapshot(snapshot_dir: str | Path) -> CorpusSnapshot:
    directory = Path(snapshot_dir).expanduser().resolve()
    manifest_path = directory / "snapshot.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing snapshot manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("snapshot manifest root must be an object")
    return _snapshot_from_manifest(directory, manifest)
