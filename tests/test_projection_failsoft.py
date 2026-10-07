"""A projection failure must never discard a completed analysis (audit D2).

The graph projector writes a *rebuildable* projection; the authority layer is
SQLite. Before this change ``LandscapePipeline.run`` let a projection error
propagate, so the first real MAG-50 run threw away every already-paid-for model
call and produced no report at all.

These tests pin the fail-soft contract: the analysis survives, the report still
renders, and the failure is recorded on the result instead of vanishing.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path as _Path
from typing import Any

from academic_landscape.graph_projector import GraphProjectionError
from academic_landscape.landscape_report import (
    landscape_result_to_dict,
    render_landscape_report,
)
from academic_landscape.pipeline import LandscapePipeline

_spec = importlib.util.spec_from_file_location(
    "tp_shared",
    _Path(__file__).parent / "test_pipeline.py",
)
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


class ExplodingProjector:
    """Fails the way a dead Neo4j does."""

    def __init__(self) -> None:
        self.calls = 0

    def project(self, snapshot: Any, payload: dict[str, Any]) -> Any:
        self.calls += 1
        raise GraphProjectionError("analysis validation failed: invalid_relation_type")


def _pipeline(projector: Any) -> tuple[Any, dict[str, Any], LandscapePipeline]:
    snapshot = tp._snapshot()
    extractions = {
        "paper_a": tp._extraction("paper_a"),
        "paper_b": tp._extraction("paper_b"),
    }
    pipeline = LandscapePipeline(
        client=tp.ScriptedLLM(),
        embedding_backend=tp.FakeEmbeddingBackend(),
        current_year=2026,
        n_clusters=1,
        graph_projector=projector,
    )
    return snapshot, extractions, pipeline


def test_projection_failure_does_not_abort_the_run() -> None:
    snapshot, extractions, pipeline = _pipeline(ExplodingProjector())

    result = pipeline.run(snapshot, extractions)

    assert result.graph_projection is None
    assert result.graph_projection_error is not None
    assert "invalid_relation_type" in result.graph_projection_error


def test_results_rest_is_populated_when_projection_fails() -> None:
    snapshot, extractions, pipeline = _pipeline(ExplodingProjector())

    result = pipeline.run(snapshot, extractions)

    assert result.directions
    assert result.analysis_version is not None


def test_report_renders_and_surfaces_the_projection_failure() -> None:
    snapshot, extractions, pipeline = _pipeline(ExplodingProjector())
    result = pipeline.run(snapshot, extractions)

    report = render_landscape_report(result)

    assert "9.1 执行摘要" in report
    assert "9.11 证据附录" in report
    assert "图谱投影" in report
    assert "invalid_relation_type" in report


def test_projection_error_is_preserved_in_structured_output() -> None:
    snapshot, extractions, pipeline = _pipeline(ExplodingProjector())
    result = pipeline.run(snapshot, extractions)

    payload = landscape_result_to_dict(result)

    assert payload["graph_projection_error"] is not None
    assert "invalid_relation_type" in payload["graph_projection_error"]


def test_successful_projection_records_no_error() -> None:
    class OkProjector:
        def project(self, snapshot: Any, payload: dict[str, Any]) -> Any:
            class Result:
                node_count = 3
                relationship_count = 2

            return Result()

    snapshot, extractions, pipeline = _pipeline(OkProjector())
    result = pipeline.run(snapshot, extractions)

    assert result.graph_projection is not None
    assert result.graph_projection_error is None


def test_default_result_has_no_projection_error_field_value() -> None:
    """The field is optional so older callers and fixtures keep working."""
    snapshot, extractions, pipeline = _pipeline(None)
    result = pipeline.run(snapshot, extractions)

    assert result.graph_projection is None
    assert result.graph_projection_error is None
    assert asdict(result)["graph_projection_error"] is None
