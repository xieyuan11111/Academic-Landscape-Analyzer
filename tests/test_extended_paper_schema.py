from pathlib import Path

import fitz

from academic_landscape.ingest import build_snapshot
from academic_landscape.paper_extraction import PaperExtractionService


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_extended_single_paper_schema_preserves_research_dimensions_and_evidence(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The study uses interviews to examine rural caregivers.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]

    class FakeClient:
        def complete_json(self, system_prompt: str, user_prompt: str):
            if "evidence anchoring stage" in system_prompt:
                return {
                    "paper_id": document.paper_id,
                    "anchors": [
                        {
                            "evidence_id": "E1",
                            "page_number": 1,
                            "hint": "study uses interviews to examine rural caregivers",
                        }
                    ],
                }
            return {
                "paper_id": document.paper_id,
                "research_question": {"text": "How are caregivers studied?", "evidence_ids": ["E1"]},
                "concepts": [
                    {
                        "name": "caregiver support",
                        "definition": "Support available to caregivers.",
                        "evidence_ids": ["E1"],
                    }
                ],
                "theoretical_frameworks": [
                    {"name": "care work lens", "description": "A lens for care labor.", "evidence_ids": ["E1"]}
                ],
                "methods": [
                    {"name": "semi-structured interviews", "description": "Interview method.", "evidence_ids": ["E1"]}
                ],
                "data_samples": [
                    {"description": "Rural caregiver interviews.", "evidence_ids": ["E1"]}
                ],
                "contexts": [
                    {"description": "Rural caregiving context.", "evidence_ids": ["E1"]}
                ],
                "results": [{"text": "Support varies by context.", "evidence_ids": ["E1"]}],
                "limitations": [{"text": "Small interview sample.", "evidence_ids": ["E1"]}],
                "claims": [],
            }

    result = PaperExtractionService(FakeClient(), tmp_path / "cache.json").extract(snapshot, document)

    assert result.concepts[0].name == "caregiver support"
    assert result.theoretical_frameworks[0].name == "care work lens"
    assert result.methods[0].name == "semi-structured interviews"
    assert result.data_samples[0].evidence_ids == ["E1"]
    assert result.contexts[0].description.startswith("Rural")
    assert result.results[0].text.startswith("Support")
    assert result.limitations[0].text.startswith("Small")


def test_extended_field_cannot_reference_unknown_evidence(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The study uses interviews.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]

    class FakeClient:
        def complete_json(self, system_prompt: str, user_prompt: str):
            if "evidence anchoring stage" in system_prompt:
                return {
                    "paper_id": document.paper_id,
                    "anchors": [
                        {"evidence_id": "E1", "page_number": 1, "hint": "study uses interviews"}
                    ],
                }
            return {
                "paper_id": document.paper_id,
                "methods": [{"name": "interviews", "description": "", "evidence_ids": ["NOPE"]}],
                "claims": [],
            }

    try:
        PaperExtractionService(FakeClient(), tmp_path / "cache.json").extract(snapshot, document)
    except ValueError as error:
        assert "unknown_evidence_id" in str(error)
    else:
        raise AssertionError("fabricated evidence reference was accepted")