import json
from pathlib import Path

import fitz

from academic_landscape.cli import main
from academic_landscape.ingest import build_snapshot


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_cli_ingest_and_apply_analysis_commands(tmp_path: Path, capsys) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "A sentence that can be cited.")
    baseline_dir = tmp_path / "baseline"

    assert main(["ingest", str(input_dir), "--output", str(baseline_dir)]) == 0
    ingest_output = json.loads(capsys.readouterr().out)
    assert ingest_output["included_document_count"] == 1

    snapshot = build_snapshot(input_dir)
    document = snapshot.included_documents[0]
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
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
                                "quote": "A sentence that can be cited",
                            }
                        ],
                    }
                ],
                "directions": [],
                "relations": [],
                "controversies": [],
                "frontier_candidates": [],
            }
        ),
        encoding="utf-8",
    )
    final_dir = tmp_path / "final"

    assert main(["apply-analysis", str(baseline_dir), str(analysis_path), "--output", str(final_dir)]) == 0
    apply_output = json.loads(capsys.readouterr().out)
    assert apply_output["semantic_status"] == "validated"
    assert (final_dir / "report.md").exists()
