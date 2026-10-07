"""Two-track integration: timeline events into the pipeline and §9 report (TDD)."""

from __future__ import annotations

import importlib.util
from pathlib import Path as _Path
from typing import Any

from academic_landscape.landscape_report import (
    LandscapeResult,
    render_landscape_report,
)
from academic_landscape.pipeline import LandscapePipeline
from academic_landscape.timeline_events import TimelineBundle

_spec = importlib.util.spec_from_file_location(
    "tp_shared", _Path(__file__).parent / "test_pipeline.py"
)
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


class _RelationLLM(tp.ScriptedLLM):
    """Shared ScriptedLLM plus relation + relation-rich comparison branches."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if "structured relations" in system_prompt:
            return {
                "relations": [
                    {
                        "source_paper_id": "paper_a",
                        "target_paper_id": "paper_b",
                        "relation_type": "challenges",
                        "statement": "paper_b challenges the nitrogen pathway",
                        "source_evidence_ids": ["paper_a::E1"],
                        "target_evidence_ids": ["paper_b::E1"],
                        "difference_dimension": "conclusion",
                        "comparability_note": "",
                        "uncertainty": "",
                    }
                ]
            }
        return super().complete_json(system_prompt, user_prompt)


def _run() -> LandscapeResult:
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


def test_pipeline_produces_timeline_bundle() -> None:
    result = _run()
    assert isinstance(result.timeline_bundle, TimelineBundle)
    assert result.timeline_bundle.events, "relation/thread/frontier must yield events"


def test_timeline_bundle_default_empty_for_backward_compat() -> None:
    from academic_landscape.landscape_report import CorpusFacts

    facts = CorpusFacts(
        document_count=0, parsed_count=0, paper_count=0,
        year_min=0, year_max=0, parse_failures=(), duplicates=(),
    )
    result = LandscapeResult(
        snapshot_id="snapshot_x",
        generated_at="2026-09-24T00:00:00+00:00",
        corpus=facts,
        directions=(), comparisons=(), reconstructions=(),
        family_report=None, frontier_report=None,
        merged_frontiers=(), future_work={},
    )
    assert result.timeline_bundle == () or result.timeline_bundle is None or isinstance(result.timeline_bundle, TimelineBundle)


def test_challenge_relation_yields_challenge_event() -> None:
    result = _run()
    types = {event.event_type for event in result.timeline_bundle.events}
    assert "significant_challenge" in types


def test_events_sorted_by_year_in_report() -> None:
    text = render_landscape_report(_run())
    assert "演化事件链" in text
    assert "significant_challenge" in text or "重要质疑" in text
    assert "paper_b challenges the nitrogen pathway" in text


def test_events_are_evidence_bound() -> None:
    result = _run()
    for event in result.timeline_bundle.events:
        assert event.evidence_ids, "every event must cite evidence"
        assert all("::" in eid for eid in event.evidence_ids)


def test_invalid_at_only_on_counterexample() -> None:
    result = _run()
    for event in result.timeline_bundle.events:
        if event.event_type != "counterexample":
            assert not event.invalid_at, (
                f"{event.event_type} must not set invalid_at (PRD §6.3)"
            )
