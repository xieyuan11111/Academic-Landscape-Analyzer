"""Durable SQLite-backed engine projection manifest store.

Authority invariant: actual PDFs / page SQLite EvidenceSpan remain authority;
this store is a rebuildable projection ledger that must survive crash and be
scoped per snapshot.  One protocol adapter is exposed:

- :class:`SQLiteProvenanceManifest`     — implements ``graphiti_projection.ProvenanceManifest``

Both share one underlying :class:`EngineManifestStore` that is also usable
directly.  The store is stdlib-only (sqlite3 + json + dataclasses) so it works
in environments where Graphiti is not installed.

Schema invariant
----------------
A single ``engine_manifest`` table carries Graphiti ``ManifestEntry``
records.  ``(engine, projection_id)`` is the primary key; ``input_hash`` is
what guards idempotency.

Crash safety
------------
WAL journal, ``busy_timeout`` 5000ms, ``CREATE TABLE IF NOT EXISTS`` migration,
deterministic primary keys so reopening the file restores the ledger exactly.

Snapshot isolation
------------------
Read APIs require the caller to pass ``snapshot_id`` and ``engine``.  A bare
``projection_id`` lookup is intentionally NOT exposed so the caller cannot
accidentally cross-mix manifests from different snapshots — that was the
original failure mode this store replaces.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS engine_manifest (
    engine         TEXT NOT NULL,
    projection_id  TEXT NOT NULL,
    snapshot_id    TEXT NOT NULL,
    input_hash     TEXT NOT NULL,
    evidence_id    TEXT,
    relation_id    TEXT,
    paper_id       TEXT,
    status         TEXT NOT NULL,
    payload        TEXT NOT NULL,
    error          TEXT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    PRIMARY KEY (engine, projection_id)
);

CREATE INDEX IF NOT EXISTS ix_engine_manifest_snapshot
    ON engine_manifest(snapshot_id);

"""

_VALID_ENGINES = frozenset({"graphiti"})


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ManifestStoreError(RuntimeError):
    """Raised when the manifest store rejects an operation."""


class SnapshotMismatchError(ManifestStoreError):
    """Raised when a caller tries to read a projection under the wrong snapshot."""


class InputHashConflictError(ManifestStoreError):
    """Raised when re-recording a projection_id with a different input_hash."""


class InvalidEvidenceScopeError(ManifestStoreError):
    """Raised when an evidence id lacks the global ``::`` scope."""


# ---------------------------------------------------------------------------
# Low-level store
# ---------------------------------------------------------------------------

class EngineManifestStore:
    """Durable SQLite-backed ledger of projection manifests.

    The store is intentionally thin: it is the *projection* layer, not the
    authority layer.  PDFs and EvidenceSpan remain authority; this store
    records what was projected so a crash does not lose the ledger.

    Use one store per database file.  The file may not yet exist — it is
    created on first open.  Reopening the same file restores all rows
    exactly (deterministic PK).
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
    ) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._busy_timeout_ms = busy_timeout_ms
        self._conn = sqlite3.connect(
            str(self._path),
            isolation_level=None,  # autocommit; we use explicit BEGIN
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA_SQL)

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "EngineManifestStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _validate_engine(engine: str) -> str:
        if engine not in _VALID_ENGINES:
            raise ManifestStoreError(
                f"unknown engine '{engine}'; expected one of {sorted(_VALID_ENGINES)}"
            )
        return engine

    @staticmethod
    def _validate_evidence_id(evidence_id: str | None) -> str | None:
        if evidence_id is None:
            return None
        if not isinstance(evidence_id, str) or "::" not in evidence_id:
            raise InvalidEvidenceScopeError(
                f"evidence_id '{evidence_id}' is not globally scoped — "
                "must contain '::' separator (e.g. 'paper_id::local_id')"
            )
        return evidence_id

    @staticmethod
    def _serialize_payload(obj: Any) -> str:
        """Serialize a dataclass / dict into a JSON string."""
        if is_dataclass(obj) and not isinstance(obj, type):
            data = asdict(obj)
        elif isinstance(obj, dict):
            data = obj
        else:
            raise ManifestStoreError(
                f"payload must be a dataclass or dict, got {type(obj).__name__}"
            )
        return json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)

    @staticmethod
    def _deserialize_payload(raw: str) -> dict[str, Any]:
        return json.loads(raw)

    # -- low-level write --------------------------------------------------

    def _upsert(
        self,
        *,
        engine: str,
        projection_id: str,
        snapshot_id: str,
        input_hash: str,
        evidence_id: str | None,
        relation_id: str | None,
        paper_id: str | None,
        status: str,
        payload: str,
        error: str | None,
        now: float,
        require_input_hash_match: bool,
    ) -> bool:
        """Insert or update a row.  Returns True on insert, False on no-op
        (same projection_id+input_hash already present).  Raises
        :class:`InputHashConflictError` if a different input_hash is
        presented for the same (engine, projection_id)."""
        engine = self._validate_engine(engine)
        if not projection_id:
            raise ManifestStoreError("projection_id MUST NOT be empty")
        if not snapshot_id:
            raise ManifestStoreError("snapshot_id MUST NOT be empty")
        if not input_hash:
            raise ManifestStoreError("input_hash MUST NOT be empty")

        self._validate_evidence_id(evidence_id)

        cur = self._conn.execute(
            "SELECT input_hash, status FROM engine_manifest "
            "WHERE engine = ? AND projection_id = ?",
            (engine, projection_id),
        )
        row = cur.fetchone()
        if row is not None:
            existing_hash, existing_status = row
            if existing_hash != input_hash:
                raise InputHashConflictError(
                    f"projection_id={projection_id} already recorded for "
                    f"engine={engine} with input_hash={existing_hash[:12]}…; "
                    f"refusing to overwrite with input_hash={input_hash[:12]}…"
                )
            # Same id, same hash → idempotent no-op (or update status).
            self._conn.execute(
                "UPDATE engine_manifest SET status = ?, error = ?, "
                "updated_at = ?, payload = ?, paper_id = COALESCE(?, paper_id) "
                "WHERE engine = ? AND projection_id = ?",
                (status, error, now, payload, paper_id, engine, projection_id),
            )
            return False

        self._conn.execute(
            "INSERT INTO engine_manifest "
            "(engine, projection_id, snapshot_id, input_hash, evidence_id, "
            " relation_id, paper_id, status, payload, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                engine,
                projection_id,
                snapshot_id,
                input_hash,
                evidence_id,
                relation_id,
                paper_id,
                status,
                payload,
                error,
                now,
                now,
            ),
        )
        return True

    # -- Graphiti side ----------------------------------------------------

    def write_graphiti(
        self,
        entry: Any,  # graphiti_projection.ManifestEntry (lazy-imported)
    ) -> bool:
        """Persist a Graphiti ``ManifestEntry``.

        Returns True if a new row was inserted, False if idempotent no-op.
        Mirrors ``graphiti_projection.ProvenanceManifest.record`` semantics:
        an existing entry with the same input_hash is left untouched and
        False is returned.
        """
        from .graphiti_projection import ManifestEntry

        if not isinstance(entry, ManifestEntry):
            raise ManifestStoreError(
                f"write_graphiti expects ManifestEntry, "
                f"got {type(entry).__name__}"
            )

        payload = self._serialize_payload(entry)
        now = time.time()
        return self._upsert(
            engine="graphiti",
            projection_id=entry.projection_id,
            snapshot_id=entry.snapshot_id,
            input_hash=entry.input_hash,
            evidence_id=None,
            relation_id=entry.relation_id,
            paper_id=None,
            status="applied",
            payload=payload,
            error=None,
            now=now,
            require_input_hash_match=True,
        )

    def get_graphiti_entry(
        self,
        snapshot_id: str,
        projection_id: str,
    ) -> Any | None:
        """Return a :class:`graphiti_projection.ManifestEntry` rebuilt from
        the row.  Returns ``None`` if not present under this snapshot.

        Raises :class:`SnapshotMismatchError` if the row exists under a
        *different* snapshot (fail-closed retrieval)."""
        from .graphiti_projection import ManifestEntry

        row = self._fetch_row("graphiti", snapshot_id, projection_id)
        if row is None:
            return None
        data = self._deserialize_payload(row["payload"])
        return ManifestEntry(**data)

    # -- raw / admin ------------------------------------------------------

    def _fetch_row(
        self,
        engine: str,
        snapshot_id: str,
        projection_id: str,
    ) -> sqlite3.Row | None:
        engine = self._validate_engine(engine)
        if not snapshot_id:
            raise ManifestStoreError("snapshot_id MUST NOT be passed to get()")
        if not projection_id:
            raise ManifestStoreError("projection_id MUST NOT be empty")
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT * FROM engine_manifest "
            "WHERE engine = ? AND projection_id = ?",
            (engine, projection_id),
        )
        row = cur.fetchone()
        if row is None:
            return None
        if row["snapshot_id"] != snapshot_id:
            raise SnapshotMismatchError(
                f"projection_id={projection_id} (engine={engine}) exists under "
                f"snapshot_id={row['snapshot_id']!r}; refusing to expose it "
                f"under the requested snapshot_id={snapshot_id!r}"
            )
        return row

    def list_for_snapshot(self, snapshot_id: str) -> list[dict[str, Any]]:
        """Return all rows under a snapshot — diagnostic / rebuild helper."""
        if not snapshot_id:
            raise ManifestStoreError("snapshot_id MUST NOT be empty")
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT * FROM engine_manifest WHERE snapshot_id = ? "
            "ORDER BY engine, projection_id",
            (snapshot_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def iter_all(self) -> Iterator[dict[str, Any]]:
        """Iterate over every row — admin / rebuild helper."""
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT * FROM engine_manifest ORDER BY engine, projection_id"
        )
        return iter(dict(r) for r in cur.fetchall())

class SQLiteProvenanceManifest:
    """Adapter that satisfies :class:`graphiti_projection.ProvenanceManifest`.

    Bound to one snapshot_id and refuses to record an entry whose snapshot_id
    disagrees.  ``record`` returns True on new insertion, False on idempotent
    re-record (same projection_id + same input_hash), matching the in-memory
    class.
    """

    def __init__(
        self,
        store: EngineManifestStore,
        snapshot_id: str,
    ) -> None:
        if not snapshot_id:
            raise ManifestStoreError(
                "SQLiteProvenanceManifest requires a non-empty snapshot_id"
            )
        self._store = store
        self._snapshot_id = snapshot_id

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    def get(self, projection_id: str) -> Any | None:
        return self._store.get_graphiti_entry(self._snapshot_id, projection_id)

    def record(self, entry: Any) -> bool:
        if not hasattr(entry, "snapshot_id") or not hasattr(entry, "projection_id"):
            raise ManifestStoreError(
                "SQLiteProvenanceManifest.record expects a ManifestEntry-shaped "
                "object (projection_id, input_hash, snapshot_id, ...)"
            )
        if entry.snapshot_id != self._snapshot_id:
            raise SnapshotMismatchError(
                f"manifest is bound to snapshot_id={self._snapshot_id!r}; "
                f"entry carries snapshot_id={entry.snapshot_id!r} — "
                "refusing to cross-write"
            )
        return self._store.write_graphiti(entry)

    def has(self, projection_id: str) -> bool:
        return self.get(projection_id) is not None

    def entries(self) -> list[Any]:
        rows = self._store.list_for_snapshot(self._snapshot_id)
        return [r for r in rows if r["engine"] == "graphiti"]


__all__ = [
    "EngineManifestStore",
    "SQLiteProvenanceManifest",
    "ManifestStoreError",
    "SnapshotMismatchError",
    "InputHashConflictError",
    "InvalidEvidenceScopeError",
]