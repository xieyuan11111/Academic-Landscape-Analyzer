"""Contract tests for authority-safe EvolutionEvent → Graphiti temporal projection."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from academic_landscape.engine_dual_run import (
    AuthorityProjectionError,
    temporal_event_from_authority,
)
from academic_landscape.graphiti_projection import (
    GraphitiProjectionError,
    ProjectionConfig,
    ValidatedTemporalEvent,
    project_temporal_event,
)
from academic_landscape.sqlite_store import EvidenceSpan


@dataclass
class _Event:
    event_id: str = "evo_001"
    event_type: str = "method_innovation"
    subject: str = "TetR/AcrR family"
    paper_id: str = "paper_a"
    source_paper_id: str = ""
    year: int = 2005
    evidence_ids: tuple[str, ...] = ("paper_a::E1",)
    statement: str = "Paper A introduced method M for subject S."
    difference_dimension: str | None = "method"
    valid_at: str = "2005-01-01"
    invalid_at: str = ""
    snapshot_id: str = "snapshot_1"
    analysis_version: str = ""
    uncertainty: str = ""


class _Authority:
    def __init__(self, rows: dict[str, EvidenceSpan]) -> None:
        self.rows = rows

    def get_evidence(self, evidence_id: str, snapshot_id: str) -> EvidenceSpan | None:
        row = self.rows.get(evidence_id)
        return row if row is not None and row.snapshot_id == snapshot_id else None


def _span(evidence_id: str = "paper_a::E1", *, paper_id: str = "paper_a", page: int = 3) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=evidence_id,
        snapshot_id="snapshot_1",
        paper_id=paper_id,
        artifact_id=f"artifact_{paper_id}",
        page_number=page,
        quote=f"Validated quote for {evidence_id}.",
        locator="Methods",
        evidence_type="A",
    )


# ---------------------------------------------------------------------------
# Authority seam
# ---------------------------------------------------------------------------

def test_temporal_event_resolves_pages_from_authority() -> None:
    authority = _Authority({
        "paper_a::E1": _span("paper_a::E1", paper_id="paper_a", page=3),
        "paper_a::E2": _span("paper_a::E2", paper_id="paper_a", page=9),
    })
    event = _Event(evidence_ids=("paper_a::E1", "paper_a::E2"))
    resolved = temporal_event_from_authority(event, authority)
    assert isinstance(resolved, ValidatedTemporalEvent)
    assert resolved.event_id == "evo_001"
    assert resolved.evidence_ids == ["paper_a::E1", "paper_a::E2"]
    assert resolved.page_numbers == [3, 9]
    assert resolved.snapshot_id == "snapshot_1"


def test_temporal_event_rejects_missing_authority_evidence() -> None:
    authority = _Authority({"paper_a::E1": _span()})
    with pytest.raises(AuthorityProjectionError, match="not found"):
        temporal_event_from_authority(_Event(evidence_ids=("paper_a::E2",)), authority)


def test_temporal_event_rejects_evidence_owned_by_other_paper() -> None:
    authority = _Authority({"paper_b::E1": _span("paper_b::E1", paper_id="paper_b")})
    with pytest.raises(AuthorityProjectionError, match="belongs to paper"):
        temporal_event_from_authority(_Event(evidence_ids=("paper_b::E1",)), authority)


def test_temporal_event_rejects_empty_evidence_set() -> None:
    with pytest.raises(AuthorityProjectionError, match="at least one"):
        temporal_event_from_authority(_Event(evidence_ids=()), _Authority({}))


# ---------------------------------------------------------------------------
# Graphiti direct-bulk temporal event projection
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self) -> None:
        self.calls = []

    async def __call__(self, driver, episodic_nodes, episodic_edges, entity_nodes, entity_edges, embedder):
        self.calls.append((entity_nodes, entity_edges))


def _node_factory(**kwargs):
    return kwargs


def _edge_factory(**kwargs):
    return kwargs


def _event(**changes) -> ValidatedTemporalEvent:
    fields = dict(
        event_id="evo_001",
        event_type="method_innovation",
        subject="TetR/AcrR family",
        paper_id="paper_a",
        year=2005,
        evidence_ids=["paper_a::E1"],
        page_numbers=[3],
        valid_at="2005-01-01T00:00:00+00:00",
        invalid_at=None,
        snapshot_id="snapshot_1",
        statement="Paper A introduced method M.",
    )
    fields.update(changes)
    return ValidatedTemporalEvent(**fields)


def test_temporal_event_bulk_writes_subject_paper_edge_with_cutoff_fields() -> None:
    recorder = _Recorder()
    result = asyncio.run(project_temporal_event(
        relation=_event(),
        config=ProjectionConfig(group_id="grp_t"),
        bulk_writer=recorder,
        driver="driver",
        embedder="embedder",
        node_factory=_node_factory,
        edge_factory=_edge_factory,
    ))
    nodes, edges = recorder.calls[0]
    assert len(nodes) == 2
    assert {node["uuid"] for node in nodes} == {"paper_a", result.subject_node_uuid}
    assert len(edges) == 1
    edge = edges[0]
    assert edge["uuid"] == result.projection_id
    assert edge["name"] == "method_innovation"
    assert edge["valid_at"] is not None
    assert edge["invalid_at"] is None
    assert edge["attributes"]["evidence_ids"] == ["paper_a::E1"]
    assert edge["attributes"]["page_numbers"] == [3]
    assert edge["attributes"]["event_id"] == "evo_001"


def test_temporal_event_preserves_subject_semantics_in_edge_name_and_fact() -> None:
    recorder = _Recorder()
    asyncio.run(project_temporal_event(
        relation=_event(subject="Nitrite oxidation", event_type="concept_origin",
                        statement="Nitrite oxidation was proposed.", invalid_at=None),
        config=ProjectionConfig(group_id="grp_t"),
        bulk_writer=recorder, driver="d", embedder="e",
        node_factory=_node_factory, edge_factory=_edge_factory,
    ))
    nodes, edges = recorder.calls[0]
    subject_nodes = [node for node in nodes if node["name"].startswith("subject:")]
    assert len(subject_nodes) == 1
    assert subject_nodes[0]["name"] == "subject:Nitrite oxidation"
    assert edges[0]["fact"] == "Nitrite oxidation was proposed."


def test_temporal_event_rejects_missing_evidence_or_pages() -> None:
    for changes in (
        {"evidence_ids": []},
        {"page_numbers": []},
        {"snapshot_id": ""},
        {"valid_at": "2005-01-01"},
    ):
        with pytest.raises((GraphitiProjectionError, Exception)):
            asyncio.run(project_temporal_event(
                relation=_event(**changes),
                config=ProjectionConfig(group_id="grp_t"),
                bulk_writer=_Recorder(), driver="d", embedder="e",
                node_factory=_node_factory, edge_factory=_edge_factory,
            ))


def test_temporal_event_idempotent_manifest_skips_second_write() -> None:
    class _Manifest:
        def __init__(self):
            self.rows = {}

        def get(self, projection_id):
            return self.rows.get(projection_id)

        def record(self, entry):
            self.rows[entry.projection_id] = entry
            return True

    manifest = _Manifest()
    recorder = _Recorder()
    event = _event()
    first = asyncio.run(project_temporal_event(
        relation=event, config=ProjectionConfig(group_id="grp_t"),
        bulk_writer=recorder, driver="d", embedder="e",
        node_factory=_node_factory, edge_factory=_edge_factory, manifest=manifest,
    ))
    second = asyncio.run(project_temporal_event(
        relation=event, config=ProjectionConfig(group_id="grp_t"),
        bulk_writer=recorder, driver="d", embedder="e",
        node_factory=_node_factory, edge_factory=_edge_factory, manifest=manifest,
    ))
    assert first.skipped_idempotent is False
    assert second.skipped_idempotent is True
    assert len(recorder.calls) == 1
