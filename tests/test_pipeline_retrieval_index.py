"""Tests for M5 retrieval index populated inside the pipeline run."""

from __future__ import annotations

import importlib.util
from pathlib import Path as _Path
from typing import Any

from academic_landscape.pipeline import LandscapePipeline

_spec = importlib.util.spec_from_file_location(
    "tp_shared", _Path(__file__).parent / "test_pipeline.py"
)
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


class FakeRetrievalIndex:
    def __init__(self) -> None:
        self.populate_calls: list[str] = []

    def populate(self, snapshot: Any) -> dict[str, Any]:
        self.populate_calls.append(snapshot.snapshot_id)
        return {"snapshot_id": snapshot.snapshot_id, "chunks": 3}

    def search(self, snapshot: Any, query: str, limit: int = 10) -> Any:
        return None


def _run() -> Any:
    snapshot = tp._snapshot()
    extractions = {
        "paper_a": tp._extraction("paper_a"),
        "paper_b": tp._extraction("paper_b"),
    }
    return snapshot, extractions


def test_index_not_populated_when_absent() -> None:
    snapshot, extractions = _run()
    pipeline = LandscapePipeline(
        client=tp.ScriptedLLM(),
        embedding_backend=tp.FakeEmbeddingBackend(),
        current_year=2026,
        n_clusters=1,
    )
    result = pipeline.run(snapshot, extractions)
    assert result is not None


def test_index_populated_once_when_present() -> None:
    snapshot, extractions = _run()
    index = FakeRetrievalIndex()
    pipeline = LandscapePipeline(
        client=tp.ScriptedLLM(),
        embedding_backend=tp.FakeEmbeddingBackend(),
        current_year=2026,
        n_clusters=1,
        retrieval_index=index,
    )
    pipeline.run(snapshot, extractions)
    assert index.populate_calls == ["snapshot_int"]
