"""Regression tests for natural-language-safe SQLite FTS queries."""

from __future__ import annotations

from academic_landscape.sqlite_store import SQLiteEvidenceStore
from academic_landscape.models import CorpusSnapshot, DocumentRecord, PageText


def _snapshot() -> CorpusSnapshot:
    document = DocumentRecord(
        artifact_id="artifact_fts",
        paper_id="paper_fts",
        relative_path="fts.pdf",
        filename="fts.pdf",
        sha256="sha_fts",
        size_bytes=1,
        status="parsed",
        pages=(PageText(1, "Cold Adapted enzyme mechanisms in wastewater"),),
    )
    return CorpusSnapshot(
        snapshot_id="snapshot_fts_safe",
        created_at="2026-09-23T00:00:00+00:00",
        input_root="E:/input",
        documents=(document,),
    )


def test_natural_language_hyphen_query_is_fts_safe() -> None:
    safe = SQLiteEvidenceStore._safe_fts_query(
        "Cold-Adapted Enzyme Flexibility Mechanisms"
    )
    assert safe == '"Cold" OR "Adapted" OR "Enzyme" OR "Flexibility" OR "Mechanisms"'
