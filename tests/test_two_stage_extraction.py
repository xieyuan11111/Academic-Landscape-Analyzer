from pathlib import Path

import fitz
import pytest

from academic_landscape.ingest import build_snapshot
from academic_landscape.paper_extraction import PaperExtractionService


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_two_stage_extraction_anchors_evidence_before_research_card(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The study uses interviews with rural caregivers.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]

    class StagedClient:
        def __init__(self):
            self.calls: list[str] = []

        def complete_json(self, system_prompt: str, user_prompt: str):
            self.calls.append(system_prompt)
            if "evidence anchoring stage" in system_prompt:
                return {
                    "paper_id": document.paper_id,
                    "anchors": [
                        {
                            "evidence_id": "E1",
                            "page_number": 1,
                            "hint": "study uses interviews with rural caregivers",
                        }
                    ],
                }
            if "research card stage" in system_prompt:
                return {
                    "paper_id": document.paper_id,
                    "research_question": {
                        "text": "How are rural caregivers studied?",
                        "evidence_ids": ["E1"],
                    },
                    "methods": [
                        {
                            "name": "interviews",
                            "description": "Interviews with caregivers.",
                            "evidence_ids": ["E1"],
                        }
                    ],
                    "concepts": [],
                    "theoretical_frameworks": [],
                    "data_samples": [],
                    "contexts": [],
                    "results": [],
                    "limitations": [],
                    "claims": [],
                }
            raise AssertionError("unexpected prompt")

    client = StagedClient()
    result = PaperExtractionService(client, tmp_path / "cache.json").extract(snapshot, document)

    assert len(client.calls) == 2
    assert "evidence anchoring stage" in client.calls[0]
    assert "research card stage" in client.calls[1]
    assert result.evidence[0].evidence_id == "E1"
    assert result.evidence[0].quote in document.pages[0].text
    assert result.methods[0].evidence_ids == ["E1"]


def test_research_card_cannot_reference_evidence_not_in_anchor_set(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The study uses interviews.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]

    class StagedClient:
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
                "research_question": {"text": "What is studied?", "evidence_ids": ["E2"]},
                "concepts": [],
                "theoretical_frameworks": [],
                "methods": [],
                "data_samples": [],
                "contexts": [],
                "results": [],
                "limitations": [],
                "claims": [],
            }

    with pytest.raises(ValueError, match="unknown_evidence_id"):
        PaperExtractionService(StagedClient(), tmp_path / "cache.json").extract(snapshot, document)


def test_research_card_cannot_submit_a_new_evidence_array(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The study uses interviews.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]

    class StagedClient:
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
                "research_question": {"text": "What is studied?", "evidence_ids": ["E1"]},
                "evidence": [
                    {
                        "evidence_id": "E2",
                        "artifact_id": document.artifact_id,
                        "page_number": 1,
                        "quote": "The study uses interviews",
                    }
                ],
                "concepts": [],
                "theoretical_frameworks": [],
                "methods": [],
                "data_samples": [],
                "contexts": [],
                "results": [],
                "limitations": [],
                "claims": [],
            }

    with pytest.raises(ValueError, match="research_card_new_evidence"):
        PaperExtractionService(StagedClient(), tmp_path / "cache.json").extract(snapshot, document)