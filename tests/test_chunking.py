from pathlib import Path

import fitz

from academic_landscape.chunking import chunk_snapshot
from academic_landscape.ingest import build_snapshot


def make_pdf(path: Path, pages: list[str]) -> None:
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_textbox(
            fitz.Rect(72, 72, 520, 760),
            text,
            fontsize=8,
        )
    document.save(path)
    document.close()


def test_page_chunks_are_source_bound_and_deterministic(tmp_path: Path) -> None:
    make_pdf(
        tmp_path / "paper.pdf",
        ["First page of the paper.", "Second page of the paper."],
    )
    snapshot = build_snapshot(tmp_path)

    first = chunk_snapshot(snapshot)
    second = chunk_snapshot(snapshot)

    assert [(chunk.chunk_id, chunk.text_hash) for chunk in first] == [
        (chunk.chunk_id, chunk.text_hash) for chunk in second
    ]
    assert len(first) == 2
    assert first[0].chunk_type == "page"
    assert first[0].snapshot_id == snapshot.snapshot_id
    assert first[0].artifact_id == snapshot.documents[0].artifact_id
    assert first[0].page_number == 1
    assert first[0].text == "First page of the paper."
    assert first[0].payload()["text_hash"] == first[0].text_hash


def test_long_page_is_split_without_losing_source_coordinates(tmp_path: Path) -> None:
    text = " ".join(f"token-{index}" for index in range(80))
    make_pdf(tmp_path / "paper.pdf", [text])
    snapshot = build_snapshot(tmp_path)

    chunks = chunk_snapshot(snapshot, max_chars=120, overlap_chars=20)

    assert len(chunks) > 1
    assert {chunk.page_number for chunk in chunks} == {1}
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.payload()["chunk_type"] == "page" for chunk in chunks)
