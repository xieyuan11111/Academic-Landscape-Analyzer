from pathlib import Path

import fitz

from academic_landscape.engine import AnalysisEngine
from academic_landscape.ingest import build_snapshot


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class FakeClient:
    def __init__(self, paper_response, corpus_response, anchor_hint=None):
        self.paper_response = paper_response
        self.corpus_response = corpus_response
        self.anchor_hint = anchor_hint
        self.calls = []

    def complete_json(self, system_prompt: str, user_prompt: str):
        self.calls.append((system_prompt, user_prompt))
        if "evidence anchoring stage" in system_prompt:
            anchors = []
            if self.anchor_hint:
                anchors.append(
                    {"evidence_id": "E1", "page_number": 1, "hint": self.anchor_hint}
                )
            return {"paper_id": self.paper_response.get("paper_id"), "anchors": anchors}
        if "research card stage" in system_prompt:
            return self.paper_response
        return self.corpus_response


def test_engine_runs_paper_then_corpus_analysis(tmp_path: Path):
    root = tmp_path
    make_pdf(root / "paper.pdf", "The paper evaluates method A on outcome B.")
    snapshot = build_snapshot(root)
    document = snapshot.included_documents[0]
    client = FakeClient(
        {
            "paper_id": document.paper_id,
            "research_question": {"text": "What is evaluated?", "evidence_ids": ["E1"]},
            "claims": [
                {
                    "claim_id": "C1",
                    "text": "Method A is evaluated.",
                    "evidence_type": "B",
                    "evidence_ids": ["E1"],
                }
            ],
        },
        {
            "directions": [
                {
                    "direction_id": "D1",
                    "label": "method evaluation",
                    "paper_ids": [document.paper_id],
                    "evidence_ids": ["E1"],
                    "status": "candidate",
                }
            ],
            "relations": [],
            "controversies": [],
            "frontier_candidates": [],
        },
        anchor_hint="paper evaluates method A",
    )

    result = AnalysisEngine(client).run(snapshot)

    assert result["snapshot_id"] == snapshot.snapshot_id
    assert result["papers"][0]["paper_id"] == document.paper_id
    assert result["papers"][0]["evidence"][0]["quote"] in document.pages[0].text
    assert result["directions"][0]["label"] == "method evaluation"
    assert len(client.calls) == 3
    assert "evidence anchoring stage" in client.calls[0][0]
    assert "research card stage" in client.calls[1][0]


def test_engine_does_not_fabricate_evidence_from_unlocatable_hint(tmp_path: Path):
    make_pdf(tmp_path / "paper.pdf", "Only a real sentence is present.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]
    client = FakeClient(
        {
            "paper_id": document.paper_id,
            "research_question": {"text": "", "evidence_ids": []},
            "claims": [],
        },
        {"directions": [], "relations": [], "controversies": [], "frontier_candidates": []},
        anchor_hint="fabricated evidence",
    )

    result = AnalysisEngine(client).run(snapshot)

    # The hint does not locate a source span, so no fabricated evidence is produced.
    assert result["papers"][0]["evidence"] == []
    assert result["papers"][0]["claims"] == []