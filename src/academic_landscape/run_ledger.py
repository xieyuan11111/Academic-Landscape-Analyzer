"""SQLite run ledger for resumable, versioned batch analysis."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .models import DocumentRecord, utc_now_iso


class RunLedger:
    """Persist run metadata and per-artifact processing state."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                snapshot_id TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS run_items (
                run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                artifact_id TEXT NOT NULL,
                paper_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                input_hash TEXT,
                result_path TEXT,
                result_sha256 TEXT,
                started_at TEXT,
                finished_at TEXT,
                error_code TEXT,
                error_message TEXT,
                duration_ms REAL,
                PRIMARY KEY(run_id, artifact_id)
            );

            CREATE INDEX IF NOT EXISTS idx_run_items_status
                ON run_items(run_id, status);
            """
        )
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(run_items)").fetchall()
        }
        if "duration_ms" not in columns:
            self._connection.execute("ALTER TABLE run_items ADD COLUMN duration_ms REAL")
        self._connection.commit()

    @staticmethod
    def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return None if row is None else dict(row)

    def create_run(
        self,
        *,
        run_id: str,
        snapshot_id: str,
        provider: str,
        model: str,
        prompt_version: str,
        schema_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        existing = self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if existing is not None:
            expected = {
                "snapshot_id": snapshot_id,
                "provider": provider,
                "model": model,
                "prompt_version": prompt_version,
                "schema_version": schema_version,
            }
            if any(existing[key] != value for key, value in expected.items()):
                raise ValueError(f"run_id already exists with different metadata: {run_id}")
            return
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO runs
                    (run_id, snapshot_id, provider, model, prompt_version,
                     schema_version, status, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    run_id,
                    snapshot_id,
                    provider,
                    model,
                    prompt_version,
                    schema_version,
                    metadata_json,
                    utc_now_iso(),
                ),
            )

    def register_documents(self, run_id: str, documents: Iterable[DocumentRecord]) -> None:
        if self.get_run(run_id) is None:
            raise KeyError(f"unknown run_id: {run_id}")
        with self._connection:
            for document in documents:
                self._connection.execute(
                    """
                    INSERT OR IGNORE INTO run_items
                        (run_id, artifact_id, paper_id, relative_path, status)
                    VALUES (?, ?, ?, ?, 'pending')
                    """,
                    (
                        run_id,
                        document.artifact_id,
                        document.paper_id,
                        document.relative_path,
                    ),
                )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        result = self._row_dict(row)
        if result is not None:
            result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def get_item(self, run_id: str, artifact_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT * FROM run_items WHERE run_id = ? AND artifact_id = ?",
            (run_id, artifact_id),
        ).fetchone()
        result = self._row_dict(row)
        if result is None:
            raise KeyError(f"unknown run item: {run_id}/{artifact_id}")
        return result

    def pending_items(self, run_id: str, *, include_failed: bool = True) -> list[dict[str, Any]]:
        statuses = ("pending", "failed") if include_failed else ("pending",)
        placeholders = ",".join("?" for _ in statuses)
        rows = self._connection.execute(
            f"""
            SELECT * FROM run_items
            WHERE run_id = ? AND status IN ({placeholders})
            ORDER BY rowid
            """,
            (run_id, *statuses),
        ).fetchall()
        return [dict(row) for row in rows]

    def claim_item(self, run_id: str, artifact_id: str) -> None:
        now = utc_now_iso()
        with self._connection:
            result = self._connection.execute(
                """
                UPDATE run_items
                SET status = 'running', attempt_count = attempt_count + 1,
                    started_at = ?, finished_at = NULL,
                    error_code = NULL, error_message = NULL
                WHERE run_id = ? AND artifact_id = ?
                  AND status IN ('pending', 'failed')
                """,
                (now, run_id, artifact_id),
            )
            if result.rowcount != 1:
                raise ValueError(f"run item is not claimable: {run_id}/{artifact_id}")
            self._connection.execute(
                "UPDATE runs SET status = 'running', started_at = COALESCE(started_at, ?) WHERE run_id = ?",
                (now, run_id),
            )

    def recover_interrupted(self, run_id: str) -> int:
        with self._connection:
            result = self._connection.execute(
                """
                UPDATE run_items
                SET status = 'pending', error_code = 'interrupted',
                    error_message = 'run resumed after an interrupted attempt',
                    finished_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (utc_now_iso(), run_id),
            )
            if result.rowcount:
                self._connection.execute(
                    "UPDATE runs SET status = 'pending' WHERE run_id = ?",
                    (run_id,),
                )
            return int(result.rowcount)

    def mark_succeeded(
        self,
        run_id: str,
        artifact_id: str,
        *,
        result_path: str,
        result_sha256: str,
        input_hash: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        with self._connection:
            result = self._connection.execute(
                """
                UPDATE run_items
                SET status = 'succeeded', result_path = ?, result_sha256 = ?,
                    input_hash = COALESCE(?, input_hash), duration_ms = ?, finished_at = ?
                WHERE run_id = ? AND artifact_id = ? AND status = 'running'
                """,
                (
                    result_path,
                    result_sha256,
                    input_hash,
                    duration_ms,
                    utc_now_iso(),
                    run_id,
                    artifact_id,
                ),
            )
            if result.rowcount != 1:
                raise ValueError(f"run item is not running: {run_id}/{artifact_id}")
        self._refresh_run_status(run_id)

    def mark_failed(
        self,
        run_id: str,
        artifact_id: str,
        error_code: str,
        error_message: str,
        duration_ms: float | None = None,
    ) -> None:
        with self._connection:
            result = self._connection.execute(
                """
                UPDATE run_items
                SET status = 'failed', error_code = ?, error_message = ?, duration_ms = ?, finished_at = ?
                WHERE run_id = ? AND artifact_id = ? AND status = 'running'
                """,
                (
                    error_code,
                    error_message[:4000],
                    duration_ms,
                    utc_now_iso(),
                    run_id,
                    artifact_id,
                ),
            )
            if result.rowcount != 1:
                raise ValueError(f"run item is not running: {run_id}/{artifact_id}")
        self._refresh_run_status(run_id)

    def status_counts(self, run_id: str) -> dict[str, int]:
        rows = self._connection.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM run_items WHERE run_id = ?
            GROUP BY status ORDER BY status
            """,
            (run_id,),
        ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def _refresh_run_status(self, run_id: str) -> None:
        counts = self.status_counts(run_id)
        total = sum(counts.values())
        if total and counts.get("succeeded", 0) == total:
            status = "succeeded"
        elif counts.get("running", 0):
            status = "running"
        elif counts.get("pending", 0):
            status = "pending"
        elif counts.get("failed", 0):
            status = "failed"
        else:
            status = "pending"
        finished = utc_now_iso() if status in {"succeeded", "failed"} else None
        with self._connection:
            self._connection.execute(
                "UPDATE runs SET status = ?, finished_at = COALESCE(?, finished_at) WHERE run_id = ?",
                (status, finished, run_id),
            )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "RunLedger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
