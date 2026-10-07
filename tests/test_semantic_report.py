import json
from pathlib import Path

import fitz
import pytest

from academic_landscape.ingest import build_snapshot
from academic_landscape.report import write_report
from academic_landscape.snapshot import load_snapshot


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def analysis_for(snapshot):
    document = snapshot.included_documents[0]
    return {
        "schema_version": "0.1",
        "snapshot_id": snapshot.snapshot_id,
        "papers": [
            {
                "paper_id": document.paper_id,
                "research_question": {
                    "text": "What does this paper examine?",
                    "evidence_ids": ["E1"],
                },
                "claims": [
                    {
                        "claim_id": "C1",
                        "text": "The paper reports an evidence-based result.",
                        "evidence_type": "B",
                        "evidence_ids": ["E1"],
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "artifact_id": document.artifact_id,
                        "page_number": 1,
                        "quote": "This paper reports an evidence-based result",
                        "locator": "page 1",
                    }
                ],
            }
        ],
        "directions": [
            {
                "direction_id": "D1",
                "label": "evidence methods",
                "paper_ids": [document.paper_id],
                "evidence_ids": ["E1"],
                "status": "candidate",
            }
        ],
        "relations": [],
        "controversies": [],
        "frontier_candidates": [],
    }


def test_snapshot_round_trip_keeps_page_evidence(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "This paper reports an evidence-based result.")
    snapshot = build_snapshot(tmp_path)
    output = tmp_path / "baseline"
    write_report(snapshot, output)

    loaded = load_snapshot(output)

    assert loaded.snapshot_id == snapshot.snapshot_id
    assert loaded.documents[0].artifact_id == snapshot.documents[0].artifact_id
    assert loaded.documents[0].pages[0].text == snapshot.documents[0].pages[0].text


def test_semantic_report_requires_validation_and_renders_evidence(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "This paper reports an evidence-based result.")
    snapshot = build_snapshot(input_dir)

    paths = write_report(snapshot, tmp_path / "semantic", analysis=analysis_for(snapshot))
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    markdown = paths.markdown_path.read_text(encoding="utf-8")

    assert payload["analysis"]["semantic_status"] == "validated"
    assert payload["analysis"]["directions"][0]["label"] == "evidence methods"
    assert "研究方向候选" in markdown
    assert "evidence methods" in markdown
    assert "This paper reports an evidence-based result" in markdown


def test_semantic_report_refuses_invalid_analysis(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    make_pdf(input_dir / "paper.pdf", "This paper reports an evidence-based result.")
    snapshot = build_snapshot(input_dir)
    invalid = analysis_for(snapshot)
    invalid["papers"][0]["evidence"][0]["quote"] = "Not present in source"

    with pytest.raises(ValueError, match="quote_not_found"):
        write_report(snapshot, tmp_path / "semantic", analysis=invalid)
