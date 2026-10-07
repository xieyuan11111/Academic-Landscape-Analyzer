from pathlib import Path

import fitz

from academic_landscape.ingest import build_snapshot
from academic_landscape.report import write_report


def test_report_includes_deterministic_corpus_overview(tmp_path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "A reportable English source.")
    document.save(tmp_path / "paper.pdf")
    document.close()
    snapshot = build_snapshot(tmp_path)

    paths = write_report(snapshot, tmp_path / "report")
    payload = __import__("json").loads(paths.json_path.read_text(encoding="utf-8"))
    markdown = paths.markdown_path.read_text(encoding="utf-8")

    assert payload["corpus_overview"]["snapshot_id"] == snapshot.snapshot_id
    assert payload["corpus_overview"]["parsed_document_count"] == 1
    assert payload["analysis"]["semantic_status"] == "not_run"
    assert "语料概览" in markdown
    assert snapshot.snapshot_id in markdown
    assert "研究方向、争议和前沿" in markdown
