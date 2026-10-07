"""D15: the "authority evidence layer" must actually hold the evidence.

``SQLiteEvidenceStore.save_evidence`` is fully implemented — it validates the
snapshot, the artifact/document binding, the page range, and that the quote is
verbatim in that page's text — yet nothing in the write path ever calls it. On
MAG-50 the `evidence` table had **0 rows** while `pages` had 2,685 and
`documents` 211: the spans lived only in JSON artifacts.

So "the authority layer is SQLite" was true of pages and false of evidence. This
wires extraction output into the store, at the seam that already has both the
document and the validated payload: `BatchExtractionRunner`.

Two properties matter and are pinned here:

- **Globalized IDs.** The table key is ``(snapshot_id, evidence_id)`` while the
  extraction numbers evidence ``E1..En`` per paper — the same collision
  ``evidence_scope`` exists to solve. The spans must be stored as
  ``paper_id::E1`` to match the source-card convention.
- **Rejection is loud.** A row the store refuses (e.g. ``quote_not_found``) is a
  real validation failure. Dropping it silently would recreate exactly the class
  of bug this project keeps finding; the runner reports the count and writes the
  details where they can be read.
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz

from academic_landscape.batch_extraction import BatchExtractionRunner, persist_evidence
from academic_landscape.ingest import build_snapshot
from academic_landscape.paper_extraction import PaperExtractionResult
from academic_landscape.run_ledger import RunLedger
from academic_landscape.sqlite_store import SQLiteEvidenceStore

PAGE_TEXT = "The paper evaluates method A on outcome B."


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def setup(tmp_path: Path, *, quote: str = "evaluates method A"):
    papers = tmp_path / "papers"
    papers.mkdir()
    make_pdf(papers / "a.pdf", PAGE_TEXT)
    make_pdf(papers / "b.pdf", PAGE_TEXT)
    snapshot = build_snapshot(papers)

    store = SQLiteEvidenceStore(tmp_path / "authority.sqlite3")
    store.save_snapshot(snapshot)

    document = snapshot.included_documents[0]
    payload = {
        "paper_id": document.paper_id,
        "evidence": [
            {
                "evidence_id": "E1",
                "artifact_id": document.artifact_id,
                "page_number": 1,
                "quote": quote,
                "locator": "page 1",
            }
        ],
        "claims": [
            {
                "claim_id": "C1",
                "text": "method A works",
                "evidence_type": "A",
                "evidence_ids": ["E1"],
            }
        ],
    }
    return snapshot, store, document, payload


def test_persist_evidence_writes_globalized_spans(tmp_path: Path) -> None:
    snapshot, store, document, payload = setup(tmp_path)

    outcome = persist_evidence(store, snapshot, document, payload)

    assert outcome["saved"] == 1
    assert outcome["rejected"] == []

    stored = store.list_evidence(snapshot.snapshot_id)
    assert [span.evidence_id for span in stored] == [f"{document.paper_id}::E1"]
    assert stored[0].page_number == 1
    assert stored[0].quote == "evaluates method A"
    assert stored[0].evidence_type == "A"


def test_persist_evidence_is_idempotent(tmp_path: Path) -> None:
    snapshot, store, document, payload = setup(tmp_path)

    persist_evidence(store, snapshot, document, payload)
    persist_evidence(store, snapshot, document, payload)

    assert len(store.list_evidence(snapshot.snapshot_id)) == 1


def test_persist_evidence_rejects_unfindable_quote_without_raising(tmp_path: Path) -> None:
    snapshot, store, document, payload = setup(tmp_path, quote="no such sentence here")

    outcome = persist_evidence(store, snapshot, document, payload)

    assert outcome["saved"] == 0
    assert outcome["rejected"][0]["reason"] == "quote_not_found"
    assert store.list_evidence(snapshot.snapshot_id) == []


def test_persist_evidence_types_spans_without_a_citing_claim(tmp_path: Path) -> None:
    snapshot, store, document, payload = setup(tmp_path)
    payload["claims"] = []

    persist_evidence(store, snapshot, document, payload)

    stored = store.list_evidence(snapshot.snapshot_id)
    assert stored[0].evidence_type == ""


class _Extractor:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def extract(self, snapshot, document):
        return PaperExtractionResult(paper_id=document.paper_id)


def _runner(tmp_path: Path, store: SQLiteEvidenceStore, snapshot):
    ledger = RunLedger(tmp_path / "runs.sqlite3")
    ledger.create_run(
        run_id="run-001",
        snapshot_id=snapshot.snapshot_id,
        provider="deepseek",
        model="deepseek-v4-flash",
        prompt_version="paper-v0.2",
        schema_version="paper-schema-v0.2",
    )
    ledger.register_documents("run-001", snapshot.included_documents)
    return ledger


def test_batch_runner_persists_evidence_and_counts_it(tmp_path: Path) -> None:
    snapshot, store, document, payload = setup(tmp_path)
    ledger = _runner(tmp_path, store, snapshot)

    def _per_document(snap, doc):
        """Real service shape: evidence binds the paper being extracted."""
        row = json.loads(json.dumps(payload))
        row["paper_id"] = doc.paper_id
        row["evidence"][0]["artifact_id"] = doc.artifact_id
        return row

    class RunnerExtractor(_Extractor):
        def extract(self, snap, doc):
            return _per_document(snap, doc)

    runner = BatchExtractionRunner(
        ledger, RunnerExtractor(payload), tmp_path / "results", evidence_store=store
    )
    counts = runner.run(snapshot, "run-001")

    assert counts["evidence_saved"] == 2
    assert counts["evidence_rejected"] == 0
    # two papers → two globally scoped ids, not one merged row
    stored_ids = sorted(span.evidence_id for span in store.list_evidence(snapshot.snapshot_id))
    assert stored_ids == sorted(
        f"{doc.paper_id}::E1" for doc in snapshot.included_documents
    )
    ledger.close()


def test_batch_runner_reports_rejections_loudly(tmp_path: Path) -> None:
    snapshot, store, document, payload = setup(tmp_path)
    ledger = _runner(tmp_path, store, snapshot)

    def _bad(snap, doc):
        return {
            "paper_id": doc.paper_id,
            "evidence": [
                {
                    "evidence_id": "E1",
                    "artifact_id": doc.artifact_id,
                    "page_number": 1,
                    "quote": "not in the page at all",
                    "locator": "",
                }
            ],
            "claims": [],
        }

    class BadExtractor:
        def extract(self, snap, doc):
            return _bad(snap, doc)

    runner = BatchExtractionRunner(
        ledger, BadExtractor(), tmp_path / "results", evidence_store=store
    )
    counts = runner.run(snapshot, "run-001")

    assert counts["evidence_rejected"] == 2
    # the loss is written where it can be read, not swallowed
    sidecar = json.loads((tmp_path / "results" / "evidence-rejections.json").read_text())
    assert len(sidecar) == 2
    assert all(entry["reason"] == "quote_not_found" for entry in sidecar)
    ledger.close()


def test_batch_runner_without_evidence_store_is_unchanged(tmp_path: Path) -> None:
    snapshot, store, document, payload = setup(tmp_path)
    ledger = _runner(tmp_path, store, snapshot)

    class RunnerExtractor(_Extractor):
        def extract(self, snap, doc):
            return payload

    runner = BatchExtractionRunner(ledger, RunnerExtractor(payload), tmp_path / "results")
    counts = runner.run(snapshot, "run-001")

    assert "evidence_saved" not in counts
    assert store.list_evidence(snapshot.snapshot_id) == []
    ledger.close()
