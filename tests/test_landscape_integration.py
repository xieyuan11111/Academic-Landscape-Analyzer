"""Integration tests: versioning, debate cards and source cards wired into product (TDD)."""

from __future__ import annotations

import importlib.util
from pathlib import Path as _Path
from typing import Any

from academic_landscape.landscape_report import (
    LandscapeResult,
    render_landscape_report,
)
from academic_landscape.pipeline import LandscapePipeline
from academic_landscape.versioning import AnalysisVersion

_spec = importlib.util.spec_from_file_location(
    "tp_shared", _Path(__file__).parent / "test_pipeline.py"
)
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


class _ControversyLLM(tp.ScriptedLLM):
    """Shared ScriptedLLM plus a relation branch and a controversy-rich comparison."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if "structured relations" in system_prompt:
            return {
                "relations": [
                    {
                        "source_paper_id": "paper_a",
                        "target_paper_id": "paper_b",
                        "relation_type": "challenges",
                        "statement": "paper_b challenges paper_a mechanism",
                        "source_evidence_ids": ["paper_a::E1"],
                        "target_evidence_ids": ["paper_b::E1"],
                        "difference_dimension": "conclusion",
                        "comparability_note": "same reactor system",
                        "uncertainty": "single run",
                    }
                ]
            }
        if "compare papers" in system_prompt:
            return {
                "commonalities": [
                    {"statement": "shared mechanism", "evidence": ["paper_a::E1", "paper_b::E1"]}
                ],
                "differences": [
                    {"dimension": "conclusion", "statement": "conclusion split",
                     "evidence": ["paper_a::E2"]}
                ],
                "controversies": [
                    {"statement": "which nitrogen pathway dominates?",
                     "positions": [
                         {"paper_id": "paper_a", "stance": "pathway X dominates",
                          "evidence": ["paper_a::E3"]},
                         {"paper_id": "paper_b", "stance": "pathway Y dominates",
                          "evidence": ["paper_b::E3"]},
                     ]}
                ],
            }
        return super().complete_json(system_prompt, user_prompt)


def _run(llm: Any = None) -> LandscapeResult:
    snapshot = tp._snapshot()
    extractions = {
        "paper_a": tp._extraction("paper_a"),
        "paper_b": tp._extraction("paper_b"),
    }
    pipeline = LandscapePipeline(
        client=llm or _ControversyLLM(),
        embedding_backend=tp.FakeEmbeddingBackend(),
        current_year=2026,
        n_clusters=1,
    )
    return pipeline.run(snapshot, extractions)


def test_pipeline_creates_analysis_version() -> None:
    result = _run()
    assert isinstance(result.analysis_version, AnalysisVersion)
    assert result.analysis_version.snapshot_id == "snapshot_int"
    assert result.analysis_version.model
    assert result.analysis_version.created_at


def test_pipeline_builds_debate_card_from_controversy() -> None:
    result = _run()
    assert len(result.debate_cards) == 1
    card = result.debate_cards[0]
    assert "pathway" in card.question
    assert len(card.positions) == 2
    assert {position.paper_id for position in card.positions} == {"paper_a", "paper_b"}


def test_pipeline_builds_source_card_for_relation_judgment() -> None:
    result = _run()
    assert result.source_cards
    card = result.source_cards[0]
    assert card.judgment_type in {"A", "B", "C", "D"}
    assert card.supporting, "judgment must have supporting sources"
    assert card.opposing, "a challenge relation gives opposing sources"


def test_report_renders_debate_card_fields() -> None:
    text = render_landscape_report(_run())
    assert "辩论候选卡" in text
    assert "which nitrogen pathway dominates?" in text
    assert "pathway X dominates" in text


def test_report_renders_version_and_source_card() -> None:
    result = _run()
    text = render_landscape_report(result)
    assert "ver_" in text
    assert "来源卡" in text
    assert "paper_b challenges paper_a mechanism" in text
