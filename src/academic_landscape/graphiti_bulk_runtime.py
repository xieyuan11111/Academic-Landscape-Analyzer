"""Dependency-lazy, zero-LLM Graphiti direct-bulk runtime.

Academic evidence relations are already validated upstream, so this runtime
supplies only the concrete objects required by
``graphiti_projection.project_event`` and Graphiti 0.30.2's pinned internal
bulk writer.  It never constructs a public ``Graphiti`` client, because
``Graphiti.add_triplet`` may resolve or reinterpret a prevalidated relation
through LLM-backed logic and the evidence contract must not be bypassed.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .graphiti_projection import compute_group_id


class GraphitiBulkRuntimeError(RuntimeError):
    """Direct-bulk Graphiti runtime cannot be constructed safely."""


class GraphitiBulkEmbeddingError(GraphitiBulkRuntimeError):
    """The embedding backend did not return a single finite expected vector."""


@dataclass(frozen=True)
class GraphitiBulkRuntimeConfig:
    uri: str
    user: str
    password: str
    snapshot_id: str
    group_salt: str = "academic-landscape-graphiti-v0302"
    database: str = "neo4j"
    embedding_dim: int = 1024


@dataclass
class GraphitiBulkRuntimeHandle:
    """Concrete Graphiti direct-bulk dependencies for one snapshot group.

    It owns a project Neo4j driver but never creates a public ``Graphiti``
    client.  Callers pass ``bulk_writer``, ``driver``, ``embedder`` and the
    two factories into :func:`academic_landscape.graphiti_projection.project_event`.
    """

    driver: Any
    bulk_writer: Callable[..., Any]
    embedder: Any
    group_id: str
    embedding_backend: Any
    embedding_dim: int
    _entity_node: Any
    _entity_edge: Any
    _closed: bool = False

    def _vector(self, text: str, *, field: str) -> list[float]:
        if self._closed:
            raise GraphitiBulkRuntimeError("Graphiti bulk runtime handle is closed")
        values = np.asarray(self.embedding_backend.encode([text]), dtype=float)
        if values.ndim != 2 or values.shape != (1, self.embedding_dim):
            raise GraphitiBulkEmbeddingError(
                f"{field} embedding must have shape (1, {self.embedding_dim}), got {values.shape!r}"
            )
        if not np.isfinite(values).all():
            raise GraphitiBulkEmbeddingError(f"{field} embedding contains non-finite values")
        return values[0].tolist()

    def node_factory(self, **kwargs: Any) -> Any:
        name = kwargs.get("name")
        if not isinstance(name, str) or not name:
            raise GraphitiBulkRuntimeError("Graphiti EntityNode name must be non-empty")
        kwargs["name_embedding"] = self._vector(name, field="node name")
        return self._entity_node(**kwargs)

    def edge_factory(self, **kwargs: Any) -> Any:
        fact = kwargs.get("fact")
        if not isinstance(fact, str) or not fact:
            raise GraphitiBulkRuntimeError("Graphiti EntityEdge fact must be non-empty")
        kwargs["fact_embedding"] = self._vector(fact, field="edge fact")
        return self._entity_edge(**kwargs)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.driver, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GraphitiBulkRuntimeError(f"{name} must be non-empty")


def _load_graphiti_components() -> tuple[Any, Any, Any, Any, Any]:
    """Load exact Graphiti 0.30.2 pieces after telemetry is disabled."""
    os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
    try:
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        from graphiti_core.edges import EntityEdge
        from graphiti_core.embedder.client import EmbedderClient
        from graphiti_core.nodes import EntityNode
        from graphiti_core.telemetry import telemetry
        from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
    except ImportError as exc:
        raise GraphitiBulkRuntimeError(
            "graphiti-core is not installed; install the project graphiti extra"
        ) from exc
    if telemetry.is_telemetry_enabled():
        raise GraphitiBulkRuntimeError("Graphiti telemetry remained enabled after import")
    return Neo4jDriver, EntityNode, EntityEdge, EmbedderClient, add_nodes_and_edges_bulk


def build_graphiti_bulk_runtime(
    config: GraphitiBulkRuntimeConfig,
    embedding_backend: Any,
) -> GraphitiBulkRuntimeHandle:
    """Build a snapshot-isolated direct-bulk runtime without Graphiti LLM APIs."""
    for name in ("uri", "user", "password", "snapshot_id"):
        _require_nonempty(name, getattr(config, name))
    if config.embedding_dim <= 0:
        raise GraphitiBulkRuntimeError("embedding_dim must be positive")
    if not callable(getattr(embedding_backend, "encode", None)):
        raise GraphitiBulkRuntimeError("embedding_backend must expose encode(texts)")

    Neo4jDriver, EntityNode, EntityEdge, EmbedderClient, bulk_writer = _load_graphiti_components()

    class BackendEmbedder(EmbedderClient):
        async def create(self, input_data: str | Sequence[str]) -> list[float]:
            text = input_data[0] if isinstance(input_data, Sequence) and not isinstance(input_data, str) else input_data
            rows = await asyncio.to_thread(embedding_backend.encode, [str(text)])
            values = np.asarray(rows, dtype=float)
            if values.ndim != 2 or values.shape != (1, config.embedding_dim):
                raise GraphitiBulkEmbeddingError(
                    f"Graphiti embedder expected (1, {config.embedding_dim}), got {values.shape!r}"
                )
            if not np.isfinite(values).all():
                raise GraphitiBulkEmbeddingError("Graphiti embedder received non-finite values")
            return values[0].tolist()

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            rows = await asyncio.to_thread(embedding_backend.encode, list(input_data_list))
            values = np.asarray(rows, dtype=float)
            if values.ndim != 2 or values.shape != (len(input_data_list), config.embedding_dim):
                raise GraphitiBulkEmbeddingError(
                    f"Graphiti batch embedder expected ({len(input_data_list)}, {config.embedding_dim}), got {values.shape!r}"
                )
            if not np.isfinite(values).all():
                raise GraphitiBulkEmbeddingError("Graphiti batch embedder received non-finite values")
            return values.tolist()

    driver = Neo4jDriver(
        uri=config.uri,
        user=config.user,
        password=config.password,
        database=config.database,
    )
    return GraphitiBulkRuntimeHandle(
        driver=driver,
        bulk_writer=bulk_writer,
        embedder=BackendEmbedder(),
        group_id=compute_group_id(config.snapshot_id, config.group_salt),
        embedding_backend=embedding_backend,
        embedding_dim=config.embedding_dim,
        _entity_node=EntityNode,
        _entity_edge=EntityEdge,
    )


__all__ = [
    "GraphitiBulkRuntimeConfig",
    "GraphitiBulkRuntimeError",
    "GraphitiBulkEmbeddingError",
    "GraphitiBulkRuntimeHandle",
    "build_graphiti_bulk_runtime",
]
