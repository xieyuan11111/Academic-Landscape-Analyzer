"""Tests for M4 Neo4j projection wired into the pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path as _Path
from typing import Any

from academic_landscape.models import CorpusSnapshot
from academic_landscape.pipeline import LandscapePipeline

_spec = importlib.util.spec_from_file_location(
    "tp_shared",
    _Path(__file__).parent / "test_pipeline.py",
)
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


class RecordingProjector:
    def __init__(self) -> None:
        self.project_calls: list[tuple[str, dict[str, Any]]] = []

    def project(self, snapshot: Any, payload: dict[str, Any]) -> Any:
        self.project_calls.append((snapshot.snapshot_id, payload))

        class Result:
            node_count = 7
            relationship_count = 5

        return Result()


def _run_extractions() -> tuple[CorpusSnapshot, dict[str, Any]]:
    snapshot = tp._snapshot()
    extractions = {
        "paper_a": tp._extraction("paper_a"),
        "paper_b": tp._extraction("paper_b"),
    }
    return snapshot, extractions


def test_projector_not_called_when_absent() -> None:
    snapshot, extractions = _run_extractions()
    pipeline = LandscapePipeline(
        client=tp.ScriptedLLM(),
        embedding_backend=tp.FakeEmbeddingBackend(),
        current_year=2026,
        n_clusters=1,
    )
    result = pipeline.run(snapshot, extractions)
    assert result.graph_projection is None


def test_projector_called_once_with_adapted_payload() -> None:
    snapshot, extractions = _run_extractions()
    projector = RecordingProjector()
    pipeline = LandscapePipeline(
        client=tp.ScriptedLLM(),
        embedding_backend=tp.FakeEmbeddingBackend(),
        current_year=2026,
        n_clusters=1,
        graph_projector=projector,
    )
    result = pipeline.run(snapshot, extractions)

    assert len(projector.project_calls) == 1
    snapshot_id, payload = projector.project_calls[0]
    assert snapshot_id == "snapshot_int"
    assert payload["snapshot_id"] == "snapshot_int"
    assert len(payload["papers"]) == 2
    assert len(payload["directions"]) == 1
    assert result.graph_projection.node_count == 7
