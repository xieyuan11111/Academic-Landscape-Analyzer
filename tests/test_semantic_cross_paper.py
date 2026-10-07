from pathlib import Path

import fitz

from academic_landscape.ingest import build_snapshot
from academic_landscape.semantic import validate_analysis_payload


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_cross_paper_direction_must_reference_known_papers_and_evidence(tmp_path: Path) -> None:
    make_pdf(tmp_path / "a.pdf", "A method is introduced in this study.")
    make_pdf(tmp_path / "b.pdf", "A method is extended in a second study.")
    snapshot = build_snapshot(tmp_path)
    first, second = snapshot.included_documents
    payload = {
        "schema_version": "0.1",
        "snapshot_id": snapshot.snapshot_id,
        "papers": [
            {
                "paper_id": first.paper_id,
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "artifact_id": first.artifact_id,
                        "page_number": 1,
                        "quote": "A method is introduced",
                    }
                ],
            },
            {
                "paper_id": second.paper_id,
                "evidence": [
                    {
                        "evidence_id": "E2",
                        "artifact_id": second.artifact_id,
                        "page_number": 1,
                        "quote": "A method is extended",
                    }
                ],
            },
        ],
        "directions": [
            {
                "direction_id": "D1",
                "label": "method development",
                "paper_ids": [first.paper_id, second.paper_id],
                "evidence_ids": ["E1", "E2"],
                "status": "candidate",
            }
        ],
        "relations": [],
        "controversies": [],
        "frontier_candidates": [],
    }

    result = validate_analysis_payload(snapshot, payload)

    assert result.valid is True
    assert result.payload["directions"][0]["paper_ids"] == [first.paper_id, second.paper_id]
