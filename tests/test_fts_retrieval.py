from pathlib import Path

import fitz

from academic_landscape.ingest import build_snapshot
from academic_landscape.sqlite_store import SQLiteEvidenceStore


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_fts_search_returns_source_bound_pages_and_is_snapshot_scoped(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    make_pdf(first_dir / "paper.pdf", "Method alpha appears only in the first snapshot.")
    make_pdf(second_dir / "paper.pdf", "Method beta appears only in the second snapshot.")
    first = build_snapshot(first_dir)
    second = build_snapshot(second_dir)

    store = SQLiteEvidenceStore(tmp_path / "evidence.sqlite3")
    store.save_snapshot(first)
    store.save_snapshot(second)

    first_hits = store.search_pages(first.snapshot_id, "alpha", limit=10)
    second_hits = store.search_pages(second.snapshot_id, "alpha", limit=10)

    assert len(first_hits) == 1
    assert first_hits[0].snapshot_id == first.snapshot_id
    assert first_hits[0].artifact_id == first.included_documents[0].artifact_id
    assert first_hits[0].page_number == 1
    assert "alpha" in first_hits[0].text
    assert second_hits == []
    store.close()
