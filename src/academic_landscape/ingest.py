"""PDF discovery, extraction, hashing and corpus snapshot construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf

from .models import CorpusSnapshot, DocumentRecord, PageText, utc_now_iso


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


def _normalise_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _metadata(document: Any, pages: tuple[PageText, ...]) -> dict[str, str]:
    raw = document.metadata or {}
    result: dict[str, str] = {}
    for key in ("title", "author", "subject", "keywords", "creationDate"):
        value = raw.get(key)
        if value:
            result[key] = str(value).strip()

    if not result.get("title"):
        for page in pages:
            for line in page.text.splitlines():
                candidate = line.strip()
                if candidate and len(candidate) <= 240:
                    result["title"] = candidate
                    result["title_source"] = "first_nonempty_line"
                    break
            if result.get("title"):
                break
    return result


def _failed_record(
    path: Path,
    root: Path,
    digest: str,
    size_bytes: int,
    error: Exception,
) -> DocumentRecord:
    relative = path.relative_to(root).as_posix()
    artifact_id = _stable_id("artifact", f"{relative}\0{digest}")
    paper_id = _stable_id("paper", digest)
    return DocumentRecord(
        artifact_id=artifact_id,
        paper_id=paper_id,
        relative_path=relative,
        filename=path.name,
        sha256=digest,
        size_bytes=size_bytes,
        status="failed",
        parse_error=f"{type(error).__name__}: {error}",
    )


def _parse_pdf(path: Path, root: Path, digest: str) -> DocumentRecord:
    relative = path.relative_to(root).as_posix()
    artifact_id = _stable_id("artifact", f"{relative}\0{digest}")
    paper_id = _stable_id("paper", digest)
    size_bytes = path.stat().st_size

    try:
        document = pymupdf.open(path)
        pages = tuple(
            PageText(page_number=index + 1, text=_normalise_text(page.get_text("text")))
            for index, page in enumerate(document)
        )
        metadata = _metadata(document, pages)
        document.close()
    except Exception as error:  # noqa: BLE001 - a file must become a visible failed record
        return _failed_record(path, root, digest, size_bytes, error)

    text_char_count = sum(page.char_count for page in pages)
    quality_flags: list[str] = []
    if not pages:
        quality_flags.append("no_pages")
    if text_char_count == 0:
        quality_flags.append("no_text_layer")
    elif text_char_count < 100:
        quality_flags.append("sparse_text")

    status = "low_quality" if "no_text_layer" in quality_flags or "no_pages" in quality_flags else "parsed"
    return DocumentRecord(
        artifact_id=artifact_id,
        paper_id=paper_id,
        relative_path=relative,
        filename=path.name,
        sha256=digest,
        size_bytes=size_bytes,
        status=status,
        page_count=len(pages),
        text_char_count=text_char_count,
        quality_flags=tuple(quality_flags),
        metadata=metadata,
        pages=pages,
    )


def _snapshot_id(documents: tuple[DocumentRecord, ...]) -> str:
    identity = [document.manifest_dict() for document in documents]
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _stable_id("snapshot", encoded)


def build_snapshot(input_dir: str | Path) -> CorpusSnapshot:
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    pdfs = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: (
            len(path.relative_to(root).as_posix()),
            path.relative_to(root).as_posix().lower(),
        ),
    )
    documents: list[DocumentRecord] = []
    first_by_digest: dict[str, str] = {}
    for path in pdfs:
        digest = _sha256(path)
        if digest in first_by_digest:
            relative = path.relative_to(root).as_posix()
            artifact_id = _stable_id("artifact", f"{relative}\0{digest}")
            documents.append(
                DocumentRecord(
                    artifact_id=artifact_id,
                    paper_id=_stable_id("paper", digest),
                    relative_path=relative,
                    filename=path.name,
                    sha256=digest,
                    size_bytes=path.stat().st_size,
                    status="duplicate",
                    duplicate_of=first_by_digest[digest],
                    quality_flags=("duplicate_content",),
                )
            )
            continue

        record = _parse_pdf(path, root, digest)
        documents.append(record)
        first_by_digest[digest] = record.artifact_id

    frozen_documents = tuple(documents)
    return CorpusSnapshot(
        snapshot_id=_snapshot_id(frozen_documents),
        created_at=utc_now_iso(),
        input_root=str(root),
        documents=frozen_documents,
    )
