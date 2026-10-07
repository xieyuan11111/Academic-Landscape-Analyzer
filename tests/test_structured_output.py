"""The structured artifact must be pure JSON (audit finding D9).

``landscape_result_to_dict`` used ``dataclasses.asdict``, which recurses only
into dataclasses. Pydantic models (``PaperRelation``, ``EvolutionEvent``) passed
through untouched and were then stringified by ``json.dumps(..., default=str)``
into Python reprs, so the first real MAG-50 run produced a ``landscape.json``
where the two newest features — cross-paper relations and evolution events —
were not readable data at all.

The contract pinned here: the returned dict must survive ``json.dumps`` **with
no ``default=`` fallback**. A repr string that merely *looks* like data is worse
than an error, because downstream consumers cannot tell the difference.
"""

from __future__ import annotations

import json
from typing import Any

from academic_landscape.landscape_report import (
    CorpusFacts,
    LandscapeResult,
    landscape_result_to_dict,
)
from academic_landscape.relation_extraction import (
    PaperRelation,
    RelationExtractionResult,
)
from academic_landscape.timeline_events import EvolutionEvent, TimelineBundle


def _relation() -> PaperRelation:
    return PaperRelation(
        relation_id="rel_test",
        source_paper_id="paper_a",
        target_paper_id="paper_b",
        relation_type="surface_conflict",
        statement="apparent disagreement about the same measurement",
        source_evidence_ids=["paper_a::E1"],
        target_evidence_ids=["paper_b::E1"],
        difference_dimension="method",
        comparability_note="different instruments",
        uncertainty="small sample",
        snapshot_id="snapshot_x",
    )


def _event() -> EvolutionEvent:
    return EvolutionEvent(
        event_id="evt_test",
        event_type="boundary_revision",
        subject="thermal optimum",
        paper_id="paper_b",
        source_paper_id="paper_a",
        year=2024,
        evidence_ids=("paper_b::E1",),
        statement="boundary revised under cold conditions",
        difference_dimension="conditions",
        valid_at="2024-01-01",
        invalid_at="",
        snapshot_id="snapshot_x",
        analysis_version="analysis-v0.1",
        uncertainty="",
    )


def _result() -> LandscapeResult:
    facts = CorpusFacts(
        document_count=2, parsed_count=2, paper_count=2,
        year_min=2020, year_max=2024, parse_failures=(), duplicates=(),
    )
    return LandscapeResult(
        snapshot_id="snapshot_x",
        generated_at="2026-09-24T00:00:00+00:00",
        corpus=facts,
        directions=(), comparisons=(), reconstructions=(),
        family_report=None, frontier_report=None,
        merged_frontiers=(), future_work={},
        relation_results=(
            RelationExtractionResult(
                direction_id="dir_1",
                relations=(_relation(),),
                ignored_relations=(),
                validation_errors=({"code": "duplicate_relation"},),
            ),
        ),
        timeline_bundle=TimelineBundle(
            events=(_event(),),
            ignored_events=({"reason": "duplicate_event"},),
            validation_errors=(),
        ),
    )


def test_relations_serialize_as_json_objects() -> None:
    payload = landscape_result_to_dict(_result())
    relations = payload["relation_results"][0]["relations"]
    assert isinstance(relations[0], dict), type(relations[0]).__name__
    assert relations[0]["relation_type"] == "surface_conflict"
    assert relations[0]["source_evidence_ids"] == ["paper_a::E1"]


def test_events_serialize_as_json_objects() -> None:
    payload = landscape_result_to_dict(_result())
    events = payload["timeline_bundle"]["events"]
    assert isinstance(events[0], dict), type(events[0]).__name__
    assert events[0]["event_type"] == "boundary_revision"
    assert events[0]["year"] == 2024


def test_dict_is_pure_json_without_default_str() -> None:
    """No `default=` fallback allowed: anything else is silent degradation."""
    payload = landscape_result_to_dict(_result())
    encoded = json.dumps(payload, ensure_ascii=False)  # must not raise
    assert '"surface_conflict"' in encoded
    assert '"boundary_revision"' in encoded


def test_plain_dataclass_fields_still_serialize() -> None:
    payload = landscape_result_to_dict(_result())
    assert isinstance(payload["corpus"], dict)
    assert payload["corpus"]["paper_count"] == 2
    assert payload["snapshot_id"] == "snapshot_x"


def test_no_value_leaks_a_python_repr() -> None:
    payload = landscape_result_to_dict(_result())

    def walk(node: Any) -> list[str]:
        if isinstance(node, str):
            return [node] if "='" in node else []
        if isinstance(node, dict):
            return [s for v in node.values() for s in walk(v)]
        if isinstance(node, (list, tuple)):
            return [s for v in node for s in walk(v)]
        return []

    leaks = walk(payload)
    assert not leaks, f"repr-shaped strings leaked into the artifact: {leaks[:2]}"
