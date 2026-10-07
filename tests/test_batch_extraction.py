from pathlib import Path

import fitz

from academic_landscape.batch_extraction import BatchExtractionRunner
from academic_landscape.ingest import build_snapshot
from academic_landscape.paper_extraction import PaperExtractionResult
from academic_landscape.run_ledger import RunLedger


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class FakeExtractor:
    def __init__(self, failures: set[str] | None = None):
        self.failures = failures or set()
        self.calls: list[str] = []

    def extract(self, snapshot, document):
        self.calls.append(document.artifact_id)
        if document.artifact_id in self.failures:
            raise ValueError("quote_not_found")
        return PaperExtractionResult(paper_id=document.paper_id)


def setup_run(tmp_path: Path):
    papers = tmp_path / "papers"
    papers.mkdir()
    for name in ("a.pdf", "b.pdf", "c.pdf"):
        make_pdf(papers / name, f"Text for {name}")
    snapshot = build_snapshot(papers)
    ledger = RunLedger(tmp_path / "runs.sqlite3")
    ledger.create_run(
        run_id="run-batch-001",
        snapshot_id=snapshot.snapshot_id,
        provider="deepseek",
        model="deepseek-v4-flash",
        prompt_version="paper-v0.2",
        schema_version="paper-schema-v0.2",
    )
    ledger.register_documents("run-batch-001", snapshot.included_documents)
    return snapshot, ledger


def test_batch_runner_isolates_failure_and_resumes_without_reprocessing_success(tmp_path: Path) -> None:
    snapshot, ledger = setup_run(tmp_path)
    documents = snapshot.included_documents
    first_failure = {documents[1].artifact_id}
    extractor = FakeExtractor(first_failure)
    runner = BatchExtractionRunner(ledger, extractor, tmp_path / "results")

    runner.run(snapshot, "run-batch-001", max_items=2)

    assert ledger.status_counts("run-batch-001") == {"failed": 1, "succeeded": 1, "pending": 1}
    assert len(extractor.calls) == 2
    assert (tmp_path / "results" / f"{documents[0].artifact_id}.json").exists()

    extractor.failures.clear()
    runner.run(snapshot, "run-batch-001")

    assert ledger.status_counts("run-batch-001") == {"succeeded": 3}
    assert extractor.calls.count(documents[0].artifact_id) == 1
    assert extractor.calls.count(documents[1].artifact_id) == 2
    assert extractor.calls.count(documents[2].artifact_id) == 1
    assert ledger.get_run("run-batch-001")["status"] == "succeeded"
    ledger.close()


def test_batch_runner_records_failed_output_without_stopping_other_documents(tmp_path: Path) -> None:
    snapshot, ledger = setup_run(tmp_path)
    documents = snapshot.included_documents
    extractor = FakeExtractor({documents[0].artifact_id})
    runner = BatchExtractionRunner(ledger, extractor, tmp_path / "results")

    runner.run(snapshot, "run-batch-001")

    failed = ledger.get_item("run-batch-001", documents[0].artifact_id)
    succeeded = ledger.get_item("run-batch-001", documents[1].artifact_id)
    assert failed["status"] == "failed"
    assert failed["error_code"] == "ValueError"
    assert succeeded["status"] == "succeeded"
    ledger.close()
