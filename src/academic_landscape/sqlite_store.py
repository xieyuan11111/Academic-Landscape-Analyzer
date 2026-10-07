"""Durable SQLite authority layer for corpus snapshots and evidence spans.

This module deliberately uses only the Python standard library. SQLite stores the
structured provenance needed to rebuild downstream vector and graph projections;
it is not a replacement for either database.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_store import archive_snapshot
from .models import CorpusSnapshot, DocumentRecord, PageText


@dataclass(frozen=True)
class EvidenceSpan:
    evidence_id: str
    snapshot_id: str
    paper_id: str
    artifact_id: str
    page_number: int
    quote: str
    locator: str
    evidence_type: str


@dataclass(frozen=True)
class PageSearchHit:
    snapshot_id: str
    paper_id: str
    artifact_id: str
    page_number: int
    text: str
    rank: float


class SQLiteEvidenceStore:
    """Persist snapshots, pages and source-bound evidence with idempotent writes."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = self.db_path.parent / "artifacts"
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                input_root TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                manifest_json TEXT NOT NULL
            );

            -- An artifact is a content-addressed file/version and may occur in
            -- more than one immutable corpus snapshot.
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                archive_path TEXT
            );

            CREATE TABLE IF NOT EXISTS pages (
                artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
                page_number INTEGER NOT NULL CHECK(page_number >= 1),
                text TEXT NOT NULL,
                PRIMARY KEY(artifact_id, page_number)
            );

            -- Snapshot-specific document metadata is kept separate from the
            -- reusable artifact and its page text.
            CREATE TABLE IF NOT EXISTS documents (
                snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE RESTRICT,
                artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
                paper_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                duplicate_of TEXT,
                parse_error TEXT,
                page_count INTEGER NOT NULL,
                text_char_count INTEGER NOT NULL,
                quality_flags_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, artifact_id),
                UNIQUE(snapshot_id, relative_path)
            );

            CREATE TABLE IF NOT EXISTS evidence (
                snapshot_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                page_number INTEGER NOT NULL CHECK(page_number >= 1),
                quote TEXT NOT NULL,
                locator TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, evidence_id),
                FOREIGN KEY(snapshot_id, artifact_id)
                    REFERENCES documents(snapshot_id, artifact_id)
                    ON DELETE RESTRICT
            );

            CREATE INDEX IF NOT EXISTS idx_documents_snapshot
                ON documents(snapshot_id);
            CREATE INDEX IF NOT EXISTS idx_pages_artifact
                ON pages(artifact_id);
            CREATE INDEX IF NOT EXISTS idx_evidence_snapshot
                ON evidence(snapshot_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS page_fts
                USING fts5(artifact_id UNINDEXED, page_number UNINDEXED, text, tokenize='unicode61');

            CREATE TABLE IF NOT EXISTS fts_indexed_pages (
                artifact_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                PRIMARY KEY(artifact_id, page_number)
            );
            """
        )
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(artifacts)").fetchall()
        }
        if "archive_path" not in columns:
            self._connection.execute("ALTER TABLE artifacts ADD COLUMN archive_path TEXT")
        self._connection.commit()
        self._backfill_fts_index()

    def _index_page(self, artifact_id: str, page_number: int, text: str) -> None:
        indexed = self._connection.execute(
            """
            SELECT 1 FROM fts_indexed_pages
            WHERE artifact_id = ? AND page_number = ?
            """,
            (artifact_id, page_number),
        ).fetchone()
        if indexed is not None:
            return
        self._connection.execute(
            """
            INSERT INTO page_fts (artifact_id, page_number, text)
            VALUES (?, ?, ?)
            """,
            (artifact_id, page_number, text),
        )
        self._connection.execute(
            """
            INSERT INTO fts_indexed_pages (artifact_id, page_number)
            VALUES (?, ?)
            """,
            (artifact_id, page_number),
        )

    def _backfill_fts_index(self) -> None:
        rows = self._connection.execute(
            """
            SELECT p.artifact_id, p.page_number, p.text
            FROM pages AS p
            LEFT JOIN fts_indexed_pages AS indexed
              ON indexed.artifact_id = p.artifact_id
             AND indexed.page_number = p.page_number
            WHERE indexed.artifact_id IS NULL
            ORDER BY p.artifact_id, p.page_number
            """
        ).fetchall()
        if not rows:
            return
        with self._connection:
            for row in rows:
                self._index_page(row["artifact_id"], row["page_number"], row["text"])

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _snapshot_fingerprint(cls, snapshot: CorpusSnapshot) -> str:
        payload = {
            "schema_version": snapshot.schema_version,
            "snapshot_id": snapshot.snapshot_id,
            "created_at": snapshot.created_at,
            "input_root": snapshot.input_root,
            "documents": [
                {
                    **document.manifest_dict(),
                    "pages": [page.to_dict() for page in document.pages],
                }
                for document in snapshot.documents
            ],
        }
        return hashlib.sha256(cls._canonical_json(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _normalise_for_match(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        normalized = normalized.translate(str.maketrans({
            "’": "'", "‘": "'", "“": '"', "”": '"',
            "‐": "-", "‑": "-", "–": "-", "—": "-",
        }))
        normalized = re.sub(r"-\s*(?:\r?\n)\s*", "", normalized)
        normalized = re.sub(r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])", "", normalized)
        return re.sub(r"\s+", " ", normalized).strip().casefold()

    @classmethod
    def _quote_exists(cls, page_text: str, quote: str) -> bool:
        return cls._normalise_for_match(quote) in cls._normalise_for_match(page_text)

    def save_snapshot(self, snapshot: CorpusSnapshot) -> None:
        archived = archive_snapshot(snapshot, self.artifact_root)
        fingerprint = self._snapshot_fingerprint(snapshot)
        manifest_json = self._canonical_json(snapshot.to_manifest_dict())
        existing = self._connection.execute(
            "SELECT fingerprint FROM snapshots WHERE snapshot_id = ?",
            (snapshot.snapshot_id,),
        ).fetchone()
        if existing is not None:
            if existing["fingerprint"] == fingerprint:
                return
            raise ValueError(f"snapshot_id already exists with different content: {snapshot.snapshot_id}")

        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO snapshots
                        (snapshot_id, schema_version, created_at, input_root, fingerprint, manifest_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.snapshot_id,
                        snapshot.schema_version,
                        snapshot.created_at,
                        snapshot.input_root,
                        fingerprint,
                        manifest_json,
                    ),
                )
                for document in snapshot.documents:
                    artifact_row = self._connection.execute(
                        """
                        SELECT paper_id, sha256, size_bytes, archive_path
                        FROM artifacts WHERE artifact_id = ?
                        """,
                        (document.artifact_id,),
                    ).fetchone()
                    artifact_identity = (document.paper_id, document.sha256, document.size_bytes)
                    if artifact_row is None:
                        self._connection.execute(
                            """
                            INSERT INTO artifacts (artifact_id, paper_id, sha256, size_bytes, archive_path)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                document.artifact_id,
                                *artifact_identity,
                                str(archived[document.artifact_id]),
                            ),
                        )
                    elif tuple(artifact_row[:3]) != artifact_identity:
                        raise ValueError(
                            f"artifact_id already exists with different content: {document.artifact_id}"
                        )
                    elif artifact_row["archive_path"] != str(archived[document.artifact_id]):
                        self._connection.execute(
                            "UPDATE artifacts SET archive_path = ? WHERE artifact_id = ?",
                            (str(archived[document.artifact_id]), document.artifact_id),
                        )

                    for page in document.pages:
                        page_row = self._connection.execute(
                            """
                            SELECT text FROM pages
                            WHERE artifact_id = ? AND page_number = ?
                            """,
                            (document.artifact_id, page.page_number),
                        ).fetchone()
                        if page_row is None:
                            self._connection.execute(
                                """
                                INSERT INTO pages (artifact_id, page_number, text)
                                VALUES (?, ?, ?)
                                """,
                                (document.artifact_id, page.page_number, page.text),
                            )
                            self._index_page(document.artifact_id, page.page_number, page.text)
                        elif page_row["text"] != page.text:
                            raise ValueError(
                                f"artifact page already exists with different text: "
                                f"{document.artifact_id}/{page.page_number}"
                            )

                    self._connection.execute(
                        """
                        INSERT INTO documents
                            (snapshot_id, artifact_id, paper_id, relative_path, filename,
                             status, duplicate_of, parse_error, page_count, text_char_count,
                             quality_flags_json, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot.snapshot_id,
                            document.artifact_id,
                            document.paper_id,
                            document.relative_path,
                            document.filename,
                            document.status,
                            document.duplicate_of,
                            document.parse_error,
                            document.page_count,
                            document.text_char_count,
                            self._canonical_json(list(document.quality_flags)),
                            self._canonical_json(document.metadata),
                        ),
                    )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"snapshot write conflict: {snapshot.snapshot_id}") from error

    def artifact_path(self, artifact_id: str) -> Path:
        row = self._connection.execute(
            "SELECT archive_path FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is not None and row["archive_path"]:
            return Path(row["archive_path"])
        return self.artifact_root / f"{artifact_id}.pdf"

    def load_snapshot(self, snapshot_id: str) -> CorpusSnapshot:
        snapshot_row = self._connection.execute(
            """
            SELECT snapshot_id, schema_version, created_at, input_root
            FROM snapshots WHERE snapshot_id = ?
            """,
            (snapshot_id,),
        ).fetchone()
        if snapshot_row is None:
            raise KeyError(f"unknown snapshot_id: {snapshot_id}")

        document_rows = self._connection.execute(
            """
            SELECT d.artifact_id, d.paper_id, d.relative_path, d.filename,
                   a.sha256, a.size_bytes, a.archive_path, d.status, d.duplicate_of, d.parse_error,
                   d.page_count, d.text_char_count, d.quality_flags_json, d.metadata_json
            FROM documents AS d
            JOIN artifacts AS a ON a.artifact_id = d.artifact_id
            WHERE d.snapshot_id = ?
            ORDER BY LENGTH(d.relative_path), LOWER(d.relative_path)
            """,
            (snapshot_id,),
        ).fetchall()
        documents: list[DocumentRecord] = []
        for row in document_rows:
            page_rows = self._connection.execute(
                """
                SELECT page_number, text FROM pages
                WHERE artifact_id = ? ORDER BY page_number
                """,
                (row["artifact_id"],),
            ).fetchall()
            documents.append(
                DocumentRecord(
                    artifact_id=row["artifact_id"],
                    paper_id=row["paper_id"],
                    relative_path=row["relative_path"],
                    filename=row["filename"],
                    sha256=row["sha256"],
                    size_bytes=row["size_bytes"],
                    status=row["status"],
                    archive_path=row["archive_path"],
                    duplicate_of=row["duplicate_of"],
                    parse_error=row["parse_error"],
                    page_count=row["page_count"],
                    text_char_count=row["text_char_count"],
                    quality_flags=tuple(json.loads(row["quality_flags_json"])),
                    metadata=dict(json.loads(row["metadata_json"])),
                    pages=tuple(
                        PageText(page_number=page["page_number"], text=page["text"])
                        for page in page_rows
                    ),
                )
            )
        return CorpusSnapshot(
            snapshot_id=snapshot_row["snapshot_id"],
            created_at=snapshot_row["created_at"],
            input_root=snapshot_row["input_root"],
            documents=tuple(documents),
            schema_version=snapshot_row["schema_version"],
        )

    def count_snapshots(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM snapshots").fetchone()
        return int(row["count"])

    def count_documents(self, snapshot_id: str) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM documents WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        return int(row["count"])

    def save_evidence(self, evidence: EvidenceSpan) -> None:
        snapshot = self._connection.execute(
            "SELECT snapshot_id FROM snapshots WHERE snapshot_id = ?",
            (evidence.snapshot_id,),
        ).fetchone()
        if snapshot is None:
            raise ValueError(f"unknown snapshot_id: {evidence.snapshot_id}")

        document = self._connection.execute(
            """
            SELECT d.paper_id, d.page_count
            FROM documents AS d
            WHERE d.artifact_id = ? AND d.snapshot_id = ?
            """,
            (evidence.artifact_id, evidence.snapshot_id),
        ).fetchone()
        if document is None:
            raise ValueError("artifact_id does not belong to snapshot_id")
        if document["paper_id"] != evidence.paper_id:
            raise ValueError("paper_id does not match artifact_id")
        if evidence.page_number < 1 or evidence.page_number > document["page_count"]:
            raise ValueError("page_out_of_range")
        page = self._connection.execute(
            """
            SELECT text FROM pages WHERE artifact_id = ? AND page_number = ?
            """,
            (evidence.artifact_id, evidence.page_number),
        ).fetchone()
        if page is None:
            raise ValueError("page_out_of_range")
        if not self._quote_exists(page["text"], evidence.quote):
            raise ValueError("quote_not_found")

        existing = self._connection.execute(
            """
            SELECT paper_id, artifact_id, page_number, quote, locator, evidence_type
            FROM evidence WHERE snapshot_id = ? AND evidence_id = ?
            """,
            (evidence.snapshot_id, evidence.evidence_id),
        ).fetchone()
        if existing is not None:
            existing_evidence = self._evidence_from_row(
                existing,
                evidence.evidence_id,
                evidence.snapshot_id,
            )
            if existing_evidence == evidence:
                return
            raise ValueError(f"evidence_id already exists with different content: {evidence.evidence_id}")

        try:
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO evidence
                        (snapshot_id, evidence_id, paper_id, artifact_id, page_number,
                         quote, locator, evidence_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        evidence.snapshot_id,
                        evidence.evidence_id,
                        evidence.paper_id,
                        evidence.artifact_id,
                        evidence.page_number,
                        evidence.quote,
                        evidence.locator,
                        evidence.evidence_type,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"evidence write conflict: {evidence.evidence_id}") from error

    @staticmethod
    def _evidence_from_row(
        row: sqlite3.Row,
        evidence_id: str,
        snapshot_id: str,
    ) -> EvidenceSpan:
        return EvidenceSpan(
            evidence_id=evidence_id,
            snapshot_id=snapshot_id,
            paper_id=row["paper_id"],
            artifact_id=row["artifact_id"],
            page_number=row["page_number"],
            quote=row["quote"],
            locator=row["locator"],
            evidence_type=row["evidence_type"],
        )

    def get_evidence(self, evidence_id: str, snapshot_id: str | None = None) -> EvidenceSpan | None:
        if snapshot_id is None:
            rows = self._connection.execute(
                """
                SELECT snapshot_id, paper_id, artifact_id, page_number, quote, locator, evidence_type
                FROM evidence WHERE evidence_id = ? ORDER BY snapshot_id
                """,
                (evidence_id,),
            ).fetchall()
            if not rows:
                return None
            if len(rows) > 1:
                raise ValueError(f"evidence_id is ambiguous across snapshots: {evidence_id}")
            row = rows[0]
            return self._evidence_from_row(row, evidence_id, row["snapshot_id"])

        row = self._connection.execute(
            """
            SELECT paper_id, artifact_id, page_number, quote, locator, evidence_type
            FROM evidence WHERE snapshot_id = ? AND evidence_id = ?
            """,
            (snapshot_id, evidence_id),
        ).fetchone()
        return None if row is None else self._evidence_from_row(row, evidence_id, snapshot_id)

    def list_evidence(self, snapshot_id: str) -> list[EvidenceSpan]:
        rows = self._connection.execute(
            """
            SELECT evidence_id, paper_id, artifact_id, page_number,
                   quote, locator, evidence_type
            FROM evidence WHERE snapshot_id = ? ORDER BY evidence_id
            """,
            (snapshot_id,),
        ).fetchall()
        return [
            self._evidence_from_row(row, row["evidence_id"], snapshot_id)
            for row in rows
        ]

    @staticmethod
    def _safe_fts_query(query: str) -> str:
        tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not tokens:
            return '""'
        # Quote every token and OR them so punctuation/hyphens cannot become
        # FTS5 operators or column selectors.
        return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)

    def search_pages(self, snapshot_id: str, query: str, *, limit: int = 10) -> list[PageSearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        try:
            rows = self._connection.execute(
                """
                SELECT d.snapshot_id, d.paper_id, d.artifact_id, page_fts.page_number,
                       page_fts.text, bm25(page_fts) AS rank
                FROM page_fts
                JOIN documents AS d ON d.artifact_id = page_fts.artifact_id
                WHERE d.snapshot_id = ?
                  AND d.status = 'parsed'
                  AND page_fts MATCH ?
                ORDER BY rank, d.artifact_id, page_fts.page_number
                LIMIT ?
                """,
                (snapshot_id, self._safe_fts_query(query), limit),
            ).fetchall()
        except sqlite3.OperationalError as error:
            raise ValueError("invalid_fts_query") from error
        return [
            PageSearchHit(
                snapshot_id=row["snapshot_id"],
                paper_id=row["paper_id"],
                artifact_id=row["artifact_id"],
                page_number=int(row["page_number"]),
                text=row["text"],
                rank=float(row["rank"]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteEvidenceStore":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
