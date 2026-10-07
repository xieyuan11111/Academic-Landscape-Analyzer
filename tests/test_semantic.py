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


def test_valid_ai_payload_is_accepted_when_quotes_match_pdf(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The study compares two methods on a shared dataset.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]

    payload = {
        "schema_version": "0.1",
        "snapshot_id": snapshot.snapshot_id,
        "papers": [
            {
                "paper_id": document.paper_id,
                "research_question": {
                    "text": "How do the two methods compare?",
                    "evidence_ids": ["E1"],
                },
                "claims": [
                    {
                        "claim_id": "C1",
                        "text": "The paper compares two methods.",
                        "evidence_type": "A",
                        "evidence_ids": ["E1"],
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "artifact_id": document.artifact_id,
                        "page_number": 1,
                        "quote": "The study compares two methods",
                        "locator": "page 1",
                    }
                ],
            }
        ],
        "directions": [],
        "relations": [],
        "controversies": [],
        "frontier_candidates": [],
    }

    result = validate_analysis_payload(snapshot, payload)

    assert result.valid is True
    assert result.errors == []
    assert result.payload["papers"][0]["evidence"][0]["evidence_id"] == "E1"


def test_fabricated_quote_and_unknown_page_are_rejected(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "Only this sentence exists in the source.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]
    payload = {
        "schema_version": "0.1",
        "snapshot_id": snapshot.snapshot_id,
        "papers": [
            {
                "paper_id": document.paper_id,
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "artifact_id": document.artifact_id,
                        "page_number": 2,
                        "quote": "A fabricated sentence from nowhere.",
                    },
                    {
                        "evidence_id": "E2",
                        "artifact_id": document.artifact_id,
                        "page_number": 1,
                        "quote": "A fabricated sentence from nowhere.",
                    }
                ],
            }
        ],
        "directions": [],
        "relations": [],
        "controversies": [],
        "frontier_candidates": [],
    }

    result = validate_analysis_payload(snapshot, payload)

    assert result.valid is False
    assert {error["code"] for error in result.errors} >= {
        "page_out_of_range",
        "quote_not_found",
    }
