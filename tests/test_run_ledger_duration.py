"""Tests for real-duration fields in the resumable extraction ledger."""

from __future__ import annotations

from academic_landscape.run_ledger import RunLedger
from academic_landscape.models import DocumentRecord


def _doc() -> DocumentRecord:
    return DocumentRecord(
        artifact_id="artifact_x", paper_id="paper_x", relative_path="x.pdf",
        filename="x.pdf", sha256="sha", size_bytes=1, status="parsed",
    )


def test_ledger_records_duration_ms(tmp_path) -> None:
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    ledger.create_run(
        run_id="run_x", snapshot_id="snapshot_x", provider="p", model="m",
        prompt_version="v", schema_version="v", metadata={},
    )
    ledger.register_documents("run_x", [_doc()])
    ledger.claim_item("run_x", "artifact_x")
    ledger.mark_succeeded(
        "run_x", "artifact_x", result_path="result.json", result_sha256="abc",
        input_hash="sha", duration_ms=1234.5,
    )
    item = ledger.get_item("run_x", "artifact_x")
    assert item["duration_ms"] == 1234.5
    ledger.close()
