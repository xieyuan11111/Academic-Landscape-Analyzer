"""Acceptance tests for the zero-LLM Graphiti direct-bulk runtime."""
from __future__ import annotations

import asyncio
import inspect

import numpy as np
import pytest

pytest.importorskip("graphiti_core")

from academic_landscape.graphiti_bulk_runtime import (
    GraphitiBulkEmbeddingError,
    GraphitiBulkRuntimeConfig,
    build_graphiti_bulk_runtime,
)


class _Backend:
    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.ones((len(texts), self.dim), dtype=float)


def _config(**changes: object) -> GraphitiBulkRuntimeConfig:
    fields: dict[str, object] = {
        "uri": "bolt://127.0.0.1:17687",
        "user": "neo4j",
        "password": "unit-test-only",
        "snapshot_id": "snapshot-bulk-runtime-test",
        "embedding_dim": 8,
    }
    fields.update(changes)
    return GraphitiBulkRuntimeConfig(**fields)  # type: ignore[arg-type]


def test_builds_direct_bulk_runtime_without_public_graphiti_client() -> None:
    runtime = build_graphiti_bulk_runtime(_config(), _Backend())
    try:
        assert runtime.group_id.startswith("grp_")
        assert callable(runtime.bulk_writer)
        assert not hasattr(runtime, "client")
        assert "add_triplet" not in inspect.getsource(build_graphiti_bulk_runtime)
    finally:
        asyncio.run(runtime.aclose())


def test_factories_supply_finite_project_dimension_embeddings() -> None:
    backend = _Backend()
    runtime = build_graphiti_bulk_runtime(_config(), backend)
    try:
        node = runtime.node_factory(
            uuid="paper-a", name="paper:paper-a", group_id=runtime.group_id,
            labels=["Paper"], attributes={"snapshot_id": "snapshot-bulk-runtime-test"},
        )
        edge = runtime.edge_factory(
            uuid="edge-a", group_id=runtime.group_id, source_node_uuid="paper-a",
            target_node_uuid="paper-b", created_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            name="SUPPORTS", fact="A supports B under validated conditions.", valid_at=None,
            invalid_at=None, attributes={"evidence_ids": ["paper-a::E1"]},
        )
        assert len(node.name_embedding) == 8
        assert len(edge.fact_embedding) == 8
        assert backend.calls == [["paper:paper-a"], ["A supports B under validated conditions."]]
    finally:
        asyncio.run(runtime.aclose())


def test_factory_fails_closed_on_wrong_embedding_shape() -> None:
    runtime = build_graphiti_bulk_runtime(_config(), _Backend(dim=7))
    try:
        with pytest.raises(GraphitiBulkEmbeddingError):
            runtime.node_factory(uuid="x", name="paper:x", group_id=runtime.group_id, labels=[], attributes={})
    finally:
        asyncio.run(runtime.aclose())
