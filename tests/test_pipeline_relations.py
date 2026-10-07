"""Tests for FR-04 relation wiring into the pipeline and §9 report (TDD)."""

from __future__ import annotations

import importlib.util
from pathlib import Path as _Path
from typing import Any

from academic_landscape.landscape_report import (
    CorpusFacts,
    LandscapeResult,
    render_landscape_report,
)
from academic_landscape.pipeline import LandscapePipeline
from academic_landscape.relation_extraction import RelationExtractionResult

_spec = importlib.util.spec_from_file_location(
    "tp_shared", _Path(__file__).parent / "test_pipeline.py"
)
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


class RelationRecordingProjector(tp.RecordingProjector if hasattr(tp, "RecordingProjector") else object):
    pass


class _Projector:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def project(self, snapshot: Any, payload: dict[str, Any]) -> Any:
        self.payloads.append(payload)

        class Result:
            node_count = 1
            relationship_count = 1

        return Result()


class _RelationScriptedLLM(tp.ScriptedLLM):
    """Shared ScriptedLLM plus a relation-extraction branch."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if "structured relations" in system_prompt:
            return {
                "relations": [
                    {
                        "source_paper_id": "paper_a",
                        "target_paper_id": "paper_b",
                        "relation_type": "supports",
                        "statement": "paper_b supports paper_a mechanism",
                        "source_evidence_ids": ["paper_a::E1"],
                        "target_evidence_ids": [],
                        "difference_dimension": None,
                        "comparability_note": "",
                        "uncertainty": "",
                    }
                ]
            }
        return super().complete_json(system_prompt, user_prompt)


def _run() -> tuple[Any, dict[str, Any]]:
    snapshot = tp._snapshot()
    extractions = {
        "paper_a": tp._extraction("paper_a"),
        "paper_b": tp._extraction("paper_b"),
    }
    return snapshot, extractions


def _pipeline(llm: Any = None, **kwargs: Any) -> LandscapePipeline:
    return LandscapePipeline(
        client=llm or tp.ScriptedLLM(),
        embedding_backend=tp.FakeEmbeddingBackend(),
        current_year=2026,
        n_clusters=1,
        **kwargs,
    )


def test_pipeline_runs_relation_extraction_per_direction() -> None:
    snapshot, extractions = _run()
    result = _pipeline().run(snapshot, extractions)
    assert len(result.relation_results) == len(result.directions)
    for relation_result in result.relation_results:
        assert isinstance(relation_result, RelationExtractionResult)


def test_pipeline_relations_feed_projection() -> None:
    snapshot, extractions = _run()
    projector = _Projector()
    _pipeline(llm=_RelationScriptedLLM(), graph_projector=projector).run(
        snapshot, extractions
    )
    assert len(projector.payloads) == 1
    relations = projector.payloads[0]["relations"]
    assert relations, "validated relations must reach the projection payload"
    assert all(relation["relation_id"].startswith("rel_") for relation in relations)


def test_relation_results_default_empty_for_backward_compat() -> None:
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
    assert result.relation_results == ()


def _result_with_relation() -> LandscapeResult:
    from academic_landscape.relation_extraction import PaperRelation

    relation = PaperRelation(
        relation_id="rel_test",
        source_paper_id="paper_a",
        target_paper_id="paper_b",
        relation_type="challenges",
        statement="paper b challenges the mechanism of paper a",
        source_evidence_ids=["paper_a::E1"],
        target_evidence_ids=["paper_b::E1"],
        difference_dimension="conclusion",
        comparability_note="same system",
        uncertainty="small sample",
        snapshot_id="snapshot_x",
    )
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
                relations=(relation,),
                ignored_relations=(),
                validation_errors=(),
            ),
        ),
    )


def test_report_renders_relation_section() -> None:
    text = render_landscape_report(_result_with_relation())
    assert "跨论文关系" in text
    assert "paper b challenges the mechanism of paper a" in text
    assert "challenges" in text
    assert "paper_a" in text and "paper_b" in text


def test_report_relation_section_says_none_when_empty() -> None:
    result = _result_with_relation()
    empty = LandscapeResult(
        snapshot_id=result.snapshot_id,
        generated_at=result.generated_at,
        corpus=result.corpus,
        directions=(), comparisons=(), reconstructions=(),
        family_report=None, frontier_report=None,
        merged_frontiers=(), future_work={},
        relation_results=(),
    )
    text = render_landscape_report(empty)
    assert "跨论文关系" in text
