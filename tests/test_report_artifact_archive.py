from pathlib import Path

import fitz

from academic_landscape.ingest import build_snapshot
from academic_landscape.report import write_report


def test_report_archives_source_pdf_under_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Archived report source.")
    document.save(source)
    document.close()
    original = source.read_bytes()
    snapshot = build_snapshot(tmp_path)

    paths = write_report(snapshot, tmp_path / "report")
    artifact = snapshot.documents[0].artifact_id
    archived = paths.output_dir / "artifacts" / f"{artifact}.pdf"

    assert archived.exists()
    assert archived.read_bytes() == original
