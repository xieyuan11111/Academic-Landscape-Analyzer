from pathlib import Path

import fitz

from academic_landscape.graph_projector import Neo4jProjector
from academic_landscape.ingest import build_snapshot
from academic_landscape.semantic import validate_analysis_payload
from academic_landscape.sqlite_store import SQLiteEvidenceStore


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_sqlite_store_archives_pdf_and_survives_source_removal(tmp_path: Path) -> None:
    source_dir = tmp_path / "papers"
    source_dir.mkdir()
    source_pdf = source_dir / "paper.pdf"
    make_pdf(source_pdf, "Archived source material.")
    original_bytes = source_pdf.read_bytes()
    snapshot = build_snapshot(source_dir)
    document = snapshot.included_documents[0]

    store = SQLiteEvidenceStore(tmp_path / "evidence.sqlite3")
    store.save_snapshot(snapshot)
    archive_path = store.artifact_path(document.artifact_id)

    assert archive_path.exists()
    assert archive_path.read_bytes() == original_bytes
    source_pdf.unlink()
    store.close()

    reopened = SQLiteEvidenceStore(tmp_path / "evidence.sqlite3")
    assert reopened.artifact_path(document.artifact_id).exists()
    assert reopened.artifact_path(document.artifact_id).read_bytes() == original_bytes
    reopened.close()


def test_author_claim_cannot_reference_another_papers_evidence(tmp_path: Path) -> None:
    make_pdf(tmp_path / "a.pdf", "Paper A has its own evidence.")
    make_pdf(tmp_path / "b.pdf", "Paper B has its own evidence.")
    snapshot = build_snapshot(tmp_path)
    first, second = snapshot.included_documents
    payload = {
        "schema_version": "0.1",
        "snapshot_id": snapshot.snapshot_id,
        "papers": [
            {
                "paper_id": first.paper_id,
                "claims": [
                    {
                        "claim_id": "C-A",
                        "text": "A uses evidence from B.",
                        "evidence_type": "B",
                        "evidence_ids": ["E-B"],
                    }
                ],
                "evidence": [],
            },
            {
                "paper_id": second.paper_id,
                "claims": [],
                "evidence": [
                    {
                        "evidence_id": "E-B",
                        "artifact_id": second.artifact_id,
                        "page_number": 1,
                        "quote": "Paper B has its own evidence",
                    }
                ],
            },
        ],
        "directions": [],
        "relations": [],
        "controversies": [],
        "frontier_candidates": [],
    }

    result = validate_analysis_payload(snapshot, payload)

    assert result.valid is False
    assert any(error["code"] == "evidence_owner_mismatch" for error in result.errors)


class RecordingTx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        return None


class RecordingSession:
    def __init__(self, tx: RecordingTx) -> None:
        self.tx = tx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute_write(self, callback, *args):
        return callback(self.tx, *args)


class RecordingDriver:
    def __init__(self) -> None:
        self.tx = RecordingTx()

    def session(self, *, database: str):
        return RecordingSession(self.tx)

    def close(self):
        return None


def test_graph_projects_all_artifacts_for_one_paper(tmp_path: Path) -> None:
    source_dir = tmp_path / "papers"
    source_dir.mkdir()
    first_pdf = source_dir / "first.pdf"
    second_pdf = source_dir / "second.pdf"
    make_pdf(first_pdf, "Same work content.")
    second_pdf.write_bytes(first_pdf.read_bytes())
    snapshot = build_snapshot(source_dir)
    first, second = snapshot.documents
    payload = {
        "schema_version": "0.1",
        "snapshot_id": snapshot.snapshot_id,
        "papers": [
            {
                "paper_id": first.paper_id,
                "claims": [],
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "artifact_id": first.artifact_id,
                        "page_number": 1,
                        "quote": "Same work content",
                    }
                ],
            }
        ],
        "directions": [],
        "relations": [],
        "controversies": [],
        "frontier_candidates": [],
    }
    driver = RecordingDriver()
    projector = Neo4jProjector(
        uri="bolt://127.0.0.1:7687",
        auth=("neo4j", "test-only"),
        namespace="academic_landscape_test",
        driver=driver,
    )

    projector.project(snapshot, payload)

    artifact_ids = {
        params.get("properties", {}).get("artifact_id")
        for query, params in driver.tx.calls
        if params.get("properties", {}).get("artifact_id")
    }
    assert {first.artifact_id, second.artifact_id} <= artifact_ids
    projector.close()
