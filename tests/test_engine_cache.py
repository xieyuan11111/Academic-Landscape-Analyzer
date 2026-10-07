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


def test_engine_paper_cache_skips_only_the_repeated_paper_call(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The paper evaluates method A.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]

    class FakeClient:
        def __init__(self):
            self.calls = []

        def complete_json(self, system_prompt: str, user_prompt: str):
            self.calls.append(system_prompt)
            if "evidence anchoring stage" in system_prompt:
                return {"paper_id": document.paper_id, "anchors": []}
            if "research card stage" in system_prompt:
                return {
                    "paper_id": document.paper_id,
                    "research_question": {"text": "", "evidence_ids": []},
                    "concepts": [],
                    "theoretical_frameworks": [],
                    "methods": [],
                    "data_samples": [],
                    "contexts": [],
                    "results": [],
                    "limitations": [],
                    "claims": [],
                }
            return {
                "directions": [],
                "relations": [],
                "controversies": [],
                "frontier_candidates": [],
            }

    client = FakeClient()
    cache_path = tmp_path / "paper-cache.json"
    first = AnalysisEngine(client, cache_path=cache_path).run(snapshot)
    second = AnalysisEngine(client, cache_path=cache_path).run(snapshot)

    assert first == second
    assert len(client.calls) == 4
    assert sum("evidence anchoring stage" in prompt for prompt in client.calls) == 1
    assert sum("research card stage" in prompt for prompt in client.calls) == 1
