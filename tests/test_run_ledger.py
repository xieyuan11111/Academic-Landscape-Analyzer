from pathlib import Path

import fitz

from academic_landscape.ingest import build_snapshot
from academic_landscape.run_ledger import RunLedger


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_run_ledger_persists_metadata_and_recovers_running_items(tmp_path: Path) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    make_pdf(papers / "a.pdf", "Paper A")
    make_pdf(papers / "b.pdf", "Paper B")
    snapshot = build_snapshot(papers)
    ledger = RunLedger(tmp_path / "runs.sqlite3")
    run_id = "run-test-001"

    ledger.create_run(
        run_id=run_id,
        snapshot_id=snapshot.snapshot_id,
        provider="deepseek",
        model="deepseek-v4-flash",
        prompt_version="paper-v0.2",
        schema_version="paper-schema-v0.2",
    )
    ledger.register_documents(run_id, snapshot.included_documents)
    artifact_id = snapshot.included_documents[0].artifact_id
    ledger.claim_item(run_id, artifact_id)

    assert ledger.get_run(run_id)["model"] == "deepseek-v4-flash"
    assert ledger.get_item(run_id, artifact_id)["status"] == "running"
    assert ledger.get_item(run_id, artifact_id)["attempt_count"] == 1

    recovered = ledger.recover_interrupted(run_id)

    assert recovered == 1
    item = ledger.get_item(run_id, artifact_id)
    assert item["status"] == "pending"
    assert item["error_code"] == "interrupted"
    ledger.close()


def test_run_ledger_is_idempotent_and_records_success_failure(tmp_path: Path) -> None:
    papers = tmp_path / "papers"
    papers.mkdir()
    make_pdf(papers / "a.pdf", "Paper A")
    snapshot = build_snapshot(papers)
    ledger = RunLedger(tmp_path / "runs.sqlite3")
    run_id = "run-test-002"

    ledger.create_run(
        run_id=run_id,
        snapshot_id=snapshot.snapshot_id,
        provider="deepseek",
        model="deepseek-v4-flash",
        prompt_version="paper-v0.2",
        schema_version="paper-schema-v0.2",
    )
    ledger.create_run(
        run_id=run_id,
        snapshot_id=snapshot.snapshot_id,
        provider="deepseek",
        model="deepseek-v4-flash",
        prompt_version="paper-v0.2",
        schema_version="paper-schema-v0.2",
    )
    ledger.register_documents(run_id, snapshot.included_documents)
    artifact_id = snapshot.included_documents[0].artifact_id
    ledger.claim_item(run_id, artifact_id)
    ledger.mark_failed(run_id, artifact_id, "quote_not_found", "quote rejected")

    assert ledger.status_counts(run_id) == {"failed": 1}
    assert ledger.get_item(run_id, artifact_id)["error_code"] == "quote_not_found"
    ledger.close()
