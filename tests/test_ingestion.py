from pathlib import Path

import fitz

from academic_landscape.ingest import build_snapshot


def make_pdf(path: Path, text: str | None, title: str | None = None) -> None:
    document = fitz.open()
    page = document.new_page()
    if text is not None:
        page.insert_text((72, 72), text)
    metadata = {"title": title or "", "author": "Test Author"}
    document.set_metadata(metadata)
    document.save(path)
    document.close()


def test_build_snapshot_tracks_parsed_duplicate_and_low_quality_files(tmp_path: Path) -> None:
    original = tmp_path / "paper-a.pdf"
    duplicate = tmp_path / "paper-a-copy.pdf"
    scan = tmp_path / "scan.pdf"
    make_pdf(original, "Research question: how does method A affect outcome B?", "Paper A")
    duplicate.write_bytes(original.read_bytes())
    make_pdf(scan, None, "Scanned paper")

    snapshot = build_snapshot(tmp_path)

    assert len(snapshot.documents) == 3
    statuses = {document.filename: document.status for document in snapshot.documents}
    assert statuses["paper-a.pdf"] == "parsed"
    assert statuses["paper-a-copy.pdf"] == "duplicate"
    assert statuses["scan.pdf"] == "low_quality"

    parsed = next(document for document in snapshot.documents if document.filename == "paper-a.pdf")
    duplicate_record = next(
        document for document in snapshot.documents if document.filename == "paper-a-copy.pdf"
    )
    assert duplicate_record.duplicate_of == parsed.artifact_id
    assert parsed.pages[0].page_number == 1
    assert "Research question" in parsed.pages[0].text
    assert parsed.metadata["title"] == "Paper A"
    assert "no_text_layer" in next(
        document for document in snapshot.documents if document.filename == "scan.pdf"
    ).quality_flags


def test_same_input_produces_same_snapshot_and_artifact_ids(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "A stable text body", "Stable paper")

    first = build_snapshot(tmp_path)
    second = build_snapshot(tmp_path)

    assert first.snapshot_id == second.snapshot_id
    assert [item.artifact_id for item in first.documents] == [
        item.artifact_id for item in second.documents
    ]
