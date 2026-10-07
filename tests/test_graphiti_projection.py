"""Tests for the zero-LLM Graphiti direct-bulk projection adapter.

The adapter remains optional-dependency safe: Graphiti types and its pinned internal
bulk writer are injected, so this suite never imports graphiti-core.
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest

from academic_landscape.graphiti_projection import (
    GraphitiProjectionError,
    ProjectionConfig,
    ProvenanceManifest,
    ValidatedRelation,
    compute_group_id,
    deterministic_projection_id,
    project_event,
)


@dataclass
class FakeNode:
    uuid: str
    name: str
    group_id: str
    labels: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeEdge:
    uuid: str
    group_id: str
    source_node_uuid: str
    target_node_uuid: str
    created_at: datetime
    name: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    attributes: dict[str, Any] = field(default_factory=dict)


class BulkRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, list[Any], list[Any], list[Any], list[Any], Any]] = []

    async def __call__(
        self,
        driver: Any,
        episodic_nodes: list[Any],
        episodic_edges: list[Any],
        entity_nodes: list[Any],
        entity_edges: list[Any],
        embedder: Any,
    ) -> None:
        self.calls.append((driver, episodic_nodes, episodic_edges, entity_nodes, entity_edges, embedder))


def _node_factory(**kwargs: Any) -> FakeNode:
    return FakeNode(**kwargs)


def _edge_factory(**kwargs: Any) -> FakeEdge:
    return FakeEdge(**kwargs)


def _relation(**changes: Any) -> ValidatedRelation:
    fields: dict[str, Any] = {
        "relation_id": "rel_1",
        "source_paper_id": "paper_a",
        "target_paper_id": "paper_b",
        "relation_type": "SUPPORTS",
        "evidence_ids": ["paper_a::E1", "paper_b::E2"],
        "page_numbers": [3, 5],
        "valid_at": "2025-01-02T03:04:05+00:00",
        "invalid_at": None,
        "snapshot_id": "snapshot_1",
        "authority_status": "validated",
        "statement": "Paper A supports paper B under stated conditions.",
        "difference_dimension": "method",
        "comparability_note": "same operational outcome",
        "uncertainty": "limited corpus",
    }
    fields.update(changes)
    return ValidatedRelation(**fields)


def _config(**changes: Any) -> ProjectionConfig:
    values: dict[str, Any] = {"group_id": "grp_test"}
    values.update(changes)
    return ProjectionConfig(**values)


def _run(**kwargs: Any) -> tuple[Any, BulkRecorder]:
    bulk = BulkRecorder()
    result = asyncio.run(
        project_event(
            relation=kwargs.pop("relation", _relation()),
            config=kwargs.pop("config", _config()),
            bulk_writer=bulk,
            driver=kwargs.pop("driver", "driver"),
            embedder=kwargs.pop("embedder", "embedder"),
            node_factory=kwargs.pop("node_factory", _node_factory),
            edge_factory=kwargs.pop("edge_factory", _edge_factory),
            **kwargs,
        )
    )
    return result, bulk


def test_project_event_is_async() -> None:
    assert inspect.iscoroutinefunction(project_event)


def test_direct_bulk_writer_called_once_with_actual_graphiti_shape() -> None:
    result, bulk = _run()
    assert result.authority_reinterpretation_risk is False
    assert len(bulk.calls) == 1
    driver, episodic_nodes, episodic_edges, nodes, edges, embedder = bulk.calls[0]
    assert driver == "driver"
    assert embedder == "embedder"
    assert episodic_nodes == []
    assert episodic_edges == []
    assert len(nodes) == 2 and len(edges) == 1
    source, target = nodes
    edge = edges[0]
    assert source.uuid == "paper_a" and target.uuid == "paper_b"
    assert source.group_id == target.group_id == "grp_test"
    assert source.labels == ["Paper"]
    assert edge.uuid == result.projection_id
    assert edge.source_node_uuid == source.uuid
    assert edge.target_node_uuid == target.uuid
    assert edge.name == "SUPPORTS"
    assert edge.fact == "Paper A supports paper B under stated conditions."


def test_edge_roundtrips_authority_provenance_and_time() -> None:
    _, bulk = _run()
    edge = bulk.calls[0][4][0]
    assert edge.attributes["authority_status"] == "validated"
    assert edge.attributes["snapshot_id"] == "snapshot_1"
    assert edge.attributes["evidence_ids"] == ["paper_a::E1", "paper_b::E2"]
    assert edge.attributes["page_numbers"] == [3, 5]
    assert edge.valid_at is not None and edge.valid_at.tzinfo is not None
    assert edge.invalid_at is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("snapshot_id", "", "snapshot_id"),
        ("source_paper_id", "", "source_paper_id"),
        ("target_paper_id", "", "target_paper_id"),
        ("evidence_ids", [], "evidence_ids"),
        ("evidence_ids", ["E1"], "globally scoped"),
        ("page_numbers", [], "page_numbers"),
        ("authority_status", "draft", "authority_status"),
        ("valid_at", "", "valid_at"),
        ("valid_at", "2025-01-02", "naive datetime"),
        ("valid_at", "not-a-date", "invalid temporal"),
    ],
)
def test_invalid_authority_inputs_fail_before_bulk_write(field: str, value: Any, message: str) -> None:
    bulk = BulkRecorder()
    with pytest.raises(GraphitiProjectionError, match=message):
        asyncio.run(
            project_event(
                _relation(**{field: value}), _config(), bulk, "driver", "embedder", _node_factory, _edge_factory
            )
        )
    assert bulk.calls == []


@pytest.mark.parametrize(
    "config",
    [_config(no_authority_reinterpretation=False), _config(telemetry_enabled=True)],
)
def test_unsafe_config_fails_before_bulk_write(config: ProjectionConfig) -> None:
    bulk = BulkRecorder()
    with pytest.raises(GraphitiProjectionError):
        asyncio.run(project_event(_relation(), config, bulk, "driver", "embedder", _node_factory, _edge_factory))
    assert bulk.calls == []


def test_manifest_makes_identical_input_idempotent() -> None:
    manifest = ProvenanceManifest()
    first, bulk = _run(manifest=manifest)
    second, _ = _run(manifest=manifest)
    assert first.skipped_idempotent is False
    assert second.skipped_idempotent is True
    assert len(bulk.calls) == 1
    assert len(manifest.entries()) == 1
    assert manifest.entries()[0].authority_reinterpretation_risk is False


def test_deterministic_ids_and_groups() -> None:
    relation = _relation()
    assert deterministic_projection_id(relation) == deterministic_projection_id(relation)
    assert deterministic_projection_id(relation).startswith("proj_")
    assert compute_group_id("snapshot_1", "graphiti-v0302") == compute_group_id("snapshot_1", "graphiti-v0302")
    assert compute_group_id("snapshot_1", "graphiti-v0302").startswith("grp_")


def test_readback_failure_is_visible_after_write() -> None:
    bulk = BulkRecorder()
    with pytest.raises(GraphitiProjectionError, match="readback"):
        asyncio.run(
            project_event(
                _relation(), _config(), bulk, "driver", "embedder", _node_factory, _edge_factory,
                readback=lambda _: False,
            )
        )
    assert len(bulk.calls) == 1


def test_no_graphiti_client_level_fallback_exists() -> None:
    """The adapter accepts only direct bulk writer, never Graphiti client methods."""
    params = set(inspect.signature(project_event).parameters)
    assert "bulk_writer" in params
    assert "client" not in params
    source = inspect.getsource(project_event)
    assert "add_triplet" not in source
    assert "add_episode" not in source
    assert "add_text" not in source
