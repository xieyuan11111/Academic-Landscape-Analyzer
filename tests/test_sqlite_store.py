from pathlib import Path

import fitz
import pytest

from academic_landscape.ingest import build_snapshot
from academic_landscape.sqlite_store import EvidenceSpan, SQLiteEvidenceStore


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_snapshot_round_trip_is_durable_and_idempotent(tmp_path: Path) -> None:
    input_dir = tmp_path / "papers"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "A durable source sentence.")
    snapshot = build_snapshot(input_dir)
    db_path = tmp_path / "evidence.sqlite3"

    store = SQLiteEvidenceStore(db_path)
    store.save_snapshot(snapshot)
    store.save_snapshot(snapshot)
    store.close()

    reopened = SQLiteEvidenceStore(db_path)
    loaded = reopened.load_snapshot(snapshot.snapshot_id)

    assert loaded.snapshot_id == snapshot.snapshot_id
    assert loaded.documents[0].artifact_id == snapshot.documents[0].artifact_id
    assert loaded.documents[0].pages[0].text == "A durable source sentence."
    assert reopened.count_snapshots() == 1
    assert reopened.count_documents(snapshot.snapshot_id) == 1
    reopened.close()


def test_same_artifact_can_be_reused_by_a_new_snapshot(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    make_pdf(first_dir / "paper.pdf", "Shared source.")
    (second_dir / "paper.pdf").write_bytes((first_dir / "paper.pdf").read_bytes())
    make_pdf(second_dir / "new-paper.pdf", "New source.")
    first = build_snapshot(first_dir)
    second = build_snapshot(second_dir)

    store = SQLiteEvidenceStore(tmp_path / "evidence.sqlite3")
    store.save_snapshot(first)
    store.save_snapshot(second)

    assert store.count_snapshots() == 2
    assert store.count_documents(first.snapshot_id) == 1
    assert store.count_documents(second.snapshot_id) == 2
    assert store.load_snapshot(second.snapshot_id).documents[0].pages[0].text == "Shared source."
    store.close()


def test_conflicting_snapshot_reuse_is_rejected(tmp_path: Path) -> None:
    input_a = tmp_path / "a"
    input_b = tmp_path / "b"
    input_a.mkdir()
    input_b.mkdir()
    make_pdf(input_a / "paper.pdf", "First source.")
    make_pdf(input_b / "paper.pdf", "Second source.")
    first = build_snapshot(input_a)
    second = build_snapshot(input_b)
    same_id = first.__class__(
        snapshot_id=first.snapshot_id,
        created_at=first.created_at,
        input_root=second.input_root,
        documents=second.documents,
    )

    store = SQLiteEvidenceStore(tmp_path / "evidence.sqlite3")
    store.save_snapshot(first)

    with pytest.raises(ValueError, match="snapshot_id already exists"):
        store.save_snapshot(same_id)

    assert store.load_snapshot(first.snapshot_id).documents[0].pages[0].text == "First source."
    store.close()


def test_evidence_span_requires_real_page_and_quote_and_round_trips(tmp_path: Path) -> None:
    input_dir = tmp_path / "papers"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "A quoted sentence for evidence.")
    snapshot = build_snapshot(input_dir)
    document = snapshot.included_documents[0]
    store = SQLiteEvidenceStore(tmp_path / "evidence.sqlite3")
    store.save_snapshot(snapshot)

    evidence = EvidenceSpan(
        evidence_id="E1",
        snapshot_id=snapshot.snapshot_id,
        paper_id=document.paper_id,
        artifact_id=document.artifact_id,
        page_number=1,
        quote="A quoted sentence for evidence",
        locator="page 1",
        evidence_type="A",
    )
    store.save_evidence(evidence)
    store.save_evidence(evidence)

    assert store.get_evidence("E1") == evidence
    assert store.list_evidence(snapshot.snapshot_id) == [evidence]

    with pytest.raises(ValueError, match="quote_not_found"):
        store.save_evidence(
            EvidenceSpan(
                evidence_id="E2",
                snapshot_id=snapshot.snapshot_id,
                paper_id=document.paper_id,
                artifact_id=document.artifact_id,
                page_number=1,
                quote="This sentence is fabricated",
                locator="page 1",
                evidence_type="A",
            )
        )

    with pytest.raises(ValueError, match="page_out_of_range"):
        store.save_evidence(
            EvidenceSpan(
                evidence_id="E3",
                snapshot_id=snapshot.snapshot_id,
                paper_id=document.paper_id,
                artifact_id=document.artifact_id,
                page_number=2,
                quote="A quoted sentence for evidence",
                locator="page 2",
                evidence_type="A",
            )
        )
    store.close()
