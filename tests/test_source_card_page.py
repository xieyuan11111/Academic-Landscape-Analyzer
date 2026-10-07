"""D14: a source card must carry a structured page number.

PRD §13.2 #5 asks that a sampled source card can be traced back to the PDF page
and quote. Measured on MAG-50 r5: **all 454 SourceRefs had an empty `locator`
and no page field at all**, while the upstream extraction cache carried
`page_number` and `locator: "page 1"` for every evidence row. `pipeline._ref()`
dropped both.

So the honest reading of the earlier gate result was: "quote ↔ page" was
verified by re-reading the extraction cache, because the card itself contained
no page information — the product promise ("point a judgment back at the
original page") was never actually met by the artifact we ship.

These tests pin the page from the extraction all the way to the rendered report.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path as _Path
from typing import Any

from academic_landscape.landscape_report import render_landscape_report
from academic_landscape.pipeline import LandscapePipeline
from academic_landscape.source_cards import SourceRef, source_card_to_dict

_spec = importlib.util.spec_from_file_location(
    "tp_shared", _Path(__file__).parent / "test_pipeline.py"
)
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


class _RelationLLM(tp.ScriptedLLM):
    """Shared ScriptedLLM plus a relation branch, so source cards get built."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if "structured relations" in system_prompt:
            return {"relations": [{
                "source_paper_id": "paper_a",
                "target_paper_id": "paper_b",
                "relation_type": "supports",
                "statement": "paper_b supports paper_a mechanism",
                "source_evidence_ids": ["paper_a::E1"],
                "target_evidence_ids": [],
                "difference_dimension": None,
                "comparability_note": "",
                "uncertainty": "",
            }]}
        return super().complete_json(system_prompt, user_prompt)


def _run():
    snapshot = tp._snapshot()
    extractions = {
        "paper_a": tp._extraction("paper_a"),
        "paper_b": tp._extraction("paper_b"),
    }
    pipeline = LandscapePipeline(
        client=_RelationLLM(),
        embedding_backend=tp.FakeEmbeddingBackend(),
        current_year=2026,
        n_clusters=1,
    )
    return pipeline.run(snapshot, extractions)


def _all_refs(result: Any) -> list[Any]:
    return [
        ref
        for card in result.source_cards
        for bucket in (card.supporting, card.revising, card.opposing)
        for ref in bucket
    ]


def test_sourceref_exposes_a_structured_page_number() -> None:
    ref = SourceRef(paper_id="paper_a", evidence_id="paper_a::E1", page_number=7)
    assert ref.page_number == 7


def test_page_number_is_optional_for_backward_compatibility() -> None:
    ref = SourceRef(paper_id="paper_a", evidence_id="paper_a::E1")
    assert ref.page_number is None


def test_source_card_serialization_includes_page_number() -> None:
    from academic_landscape.source_cards import SourceCard

    card = SourceCard(
        card_id="card_x",
        judgment="j",
        judgment_type="A",
        supporting=(SourceRef(paper_id="paper_a", evidence_id="paper_a::E1", page_number=3),),
        revising=(),
        opposing=(),
        time_attribute="",
        snapshot_id="snapshot_x",
    )
    payload = source_card_to_dict(card)
    assert payload["supporting"][0]["page_number"] == 3


def test_pipeline_populates_page_number_from_the_extraction() -> None:
    result = _run()
    refs = _all_refs(result)
    assert refs, "fixture must produce source refs"
    missing = [ref.evidence_id for ref in refs if ref.page_number is None]
    assert not missing, f"SourceRefs without a page number: {missing}"


def test_pipeline_populates_a_readable_locator_too() -> None:
    """The card should not ship an empty locator when the page is known."""
    result = _run()
    refs = _all_refs(result)
    assert refs
    blank = [ref.evidence_id for ref in refs if not ref.locator]
    assert not blank, f"SourceRefs with blank locator: {blank}"


def test_rendered_report_shows_the_page_for_a_source_ref() -> None:
    text = render_landscape_report(_run())
    assert "来源卡" in text
    # the fixture's evidence is on page 1, so the rendered card must mention it
    assert "p1" in text or "page 1" in text
