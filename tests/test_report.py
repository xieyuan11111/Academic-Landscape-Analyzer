import json
from pathlib import Path

import fitz

from academic_landscape.ingest import build_snapshot
from academic_landscape.report import write_report


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_write_report_emits_json_and_honest_markdown(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "A paper about evidence and methods.")

    snapshot = build_snapshot(input_dir)
    paths = write_report(snapshot, output_dir)

    assert paths.json_path.exists()
    assert paths.markdown_path.exists()
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert payload["snapshot_id"] == snapshot.snapshot_id
    assert payload["corpus"]["document_count"] == 1
    assert payload["analysis"]["semantic_status"] == "not_run"
    assert payload["analysis"]["research_directions"] == []

    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "在当前语料库内" in markdown
    assert "尚未运行语义分析" in markdown
    assert "不会伪装成研究方向或前沿结论" in markdown
