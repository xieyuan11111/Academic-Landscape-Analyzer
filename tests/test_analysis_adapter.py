import json
from pathlib import Path

import fitz
import pytest

from academic_landscape.analysis import apply_analysis
from academic_landscape.ingest import build_snapshot


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_apply_analysis_reads_json_and_writes_validated_report(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "A source sentence for the analysis.")
    snapshot = build_snapshot(input_dir)
    baseline_dir = tmp_path / "baseline"
    from academic_landscape.report import write_report

    write_report(snapshot, baseline_dir)
    document = snapshot.included_documents[0]
    analysis = {
        "schema_version": "0.1",
        "snapshot_id": snapshot.snapshot_id,
        "papers": [
            {
                "paper_id": document.paper_id,
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "artifact_id": document.artifact_id,
                        "page_number": 1,
                        "quote": "A source sentence for the analysis",
                    }
                ],
            }
        ],
        "directions": [],
        "relations": [],
        "controversies": [],
        "frontier_candidates": [],
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")

    paths, validation = apply_analysis(baseline_dir, analysis_path, tmp_path / "final")

    assert validation.valid is True
    assert paths.markdown_path.exists()


def test_apply_analysis_refuses_bad_json(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "A source sentence.")
    snapshot = build_snapshot(input_dir)
    baseline_dir = tmp_path / "baseline"
    from academic_landscape.report import write_report

    write_report(snapshot, baseline_dir)
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="analysis validation failed"):
        apply_analysis(baseline_dir, analysis_path, tmp_path / "final")
