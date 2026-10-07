"""Dependency-lazy Graphiti 0.30.2 direct-bulk projection.

This adapter deliberately bypasses Graphiti.add_triplet: that public method
runs LLM resolution/invalidation, which must not reinterpret prevalidated
academic evidence.  The caller injects graphiti_core.utils.bulk_utils.
add_nodes_and_edges_bulk, a version-pinned internal API.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, runtime_checkable


class GraphitiProjectionError(RuntimeError):
    """A relation cannot cross the authority projection boundary."""


@dataclass(frozen=True)
class ValidatedRelation:
    relation_id: str
    source_paper_id: str
    target_paper_id: str
    relation_type: str
    evidence_ids: list[str]
    page_numbers: list[int]
    valid_at: str
    invalid_at: str | None = None
    snapshot_id: str = ""
    authority_status: str = "validated"
    statement: str = ""
    difference_dimension: str | None = None
    comparability_note: str = ""
    uncertainty: str = ""


@dataclass(frozen=True)
class ValidatedTemporalEvent:
    """Evidence-bound EvolutionEvent projection-ready for Graphiti direct-bulk.

    This mirrors ``timeline_events.EvolutionEvent`` but with page numbers
    resolved from the authority SQLite EvidenceSpan layer rather than the
    event's own metadata.  Graphiti never re-derives the event semantics.
    """
    event_id: str
    event_type: str
    subject: str
    paper_id: str
    year: int
    evidence_ids: list[str]
    page_numbers: list[int]
    valid_at: str
    invalid_at: str | None
    snapshot_id: str
    statement: str = ""
    source_paper_id: str = ""
    difference_dimension: str | None = None
    uncertainty: str = ""


@dataclass(frozen=True)
class ProjectionConfig:
    no_authority_reinterpretation: bool = True
    telemetry_enabled: bool = False
    pre_resolved_entities: bool = True
    group_id: str = ""


@dataclass(frozen=True)
class ProjectionResult:
    projection_id: str
    authority_reinterpretation_risk: bool = False
    manifest_recorded: bool = False
    skipped_idempotent: bool = False
    subject_node_uuid: str = ""


@dataclass(frozen=True)
class ManifestEntry:
    projection_id: str
    input_hash: str
    relation_id: str
    snapshot_id: str
    projected_at: str
    authority_reinterpretation_risk: bool = False


class ProvenanceManifest:
    """Caller-side idempotency ledger; production must persist its contents."""

    def __init__(self) -> None:
        self._entries: dict[str, ManifestEntry] = {}

    def get(self, projection_id: str) -> ManifestEntry | None:
        return self._entries.get(projection_id)

    def record(self, entry: ManifestEntry) -> bool:
        if entry.projection_id in self._entries:
            return False
        self._entries[entry.projection_id] = entry
        return True

    def entries(self) -> list[ManifestEntry]:
        return list(self._entries.values())


@runtime_checkable
class NodeFactory(Protocol):
    def __call__(
        self, *, uuid: str, name: str, group_id: str, labels: list[str], attributes: dict[str, Any]
    ) -> Any: ...


@runtime_checkable
class EdgeFactory(Protocol):
    def __call__(
        self,
        *,
        uuid: str,
        group_id: str,
        source_node_uuid: str,
        target_node_uuid: str,
        created_at: datetime,
        name: str,
        fact: str,
        valid_at: datetime | None,
        invalid_at: datetime | None,
        attributes: dict[str, Any],
    ) -> Any: ...


@runtime_checkable
class BulkWriter(Protocol):
    async def __call__(
        self,
        driver: Any,
        episodic_nodes: list[Any],
        episodic_edges: list[Any],
        entity_nodes: list[Any],
        entity_edges: list[Any],
        embedder: Any,
    ) -> None: ...


_GLOBAL_ID = re.compile(r"^[A-Za-z0-9_-]+::\S+$")


def _hash(payload: str, length: int = 16) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def deterministic_projection_id(relation: ValidatedRelation) -> str:
    fields = [
        relation.relation_id,
        relation.snapshot_id,
        relation.source_paper_id,
        relation.target_paper_id,
        *sorted(relation.evidence_ids),
        relation.valid_at,
    ]
    return f"proj_{_hash(chr(0).join(fields))}"


def compute_group_id(snapshot_id: str, group_salt: str) -> str:
    return f"grp_{_hash(snapshot_id + chr(0) + group_salt)}"


def _input_hash(relation: ValidatedRelation) -> str:
    return _hash(
        "\0".join(
            [
                relation.relation_id,
                relation.snapshot_id,
                relation.source_paper_id,
                relation.target_paper_id,
                relation.relation_type,
                *sorted(relation.evidence_ids),
                *map(str, sorted(relation.page_numbers)),
                relation.valid_at,
                relation.invalid_at or "",
                relation.statement,
            ]
        ),
        64,
    )


def _parse_time(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    try:
        value = datetime.fromisoformat(raw)
    except (TypeError, ValueError) as exc:
        raise GraphitiProjectionError(f"invalid temporal value: {raw!r}") from exc
    if value.tzinfo is None:
        raise GraphitiProjectionError(f"naive datetime: {raw!r}")
    return value


def _validate(relation: ValidatedRelation, config: ProjectionConfig) -> None:
    for name, value in (
        ("relation_id", relation.relation_id),
        ("source_paper_id", relation.source_paper_id),
        ("target_paper_id", relation.target_paper_id),
        ("snapshot_id", relation.snapshot_id),
        ("valid_at", relation.valid_at),
    ):
        if not value:
            raise GraphitiProjectionError(f"{name} MUST NOT be empty")
    if not relation.evidence_ids:
        raise GraphitiProjectionError("evidence_ids MUST NOT be empty")
    if any(not _GLOBAL_ID.match(item) for item in relation.evidence_ids):
        raise GraphitiProjectionError("evidence_ids must be globally scoped")
    if not relation.page_numbers:
        raise GraphitiProjectionError("page_numbers MUST NOT be empty")
    if relation.authority_status != "validated":
        raise GraphitiProjectionError("authority_status MUST be 'validated'")
    if not config.no_authority_reinterpretation:
        raise GraphitiProjectionError("no_authority_reinterpretation MUST be True")
    if config.telemetry_enabled:
        raise GraphitiProjectionError("telemetry_enabled MUST be False")
    if not config.pre_resolved_entities:
        raise GraphitiProjectionError("pre_resolved_entities MUST be True")


async def project_event(
    relation: ValidatedRelation,
    config: ProjectionConfig,
    bulk_writer: BulkWriter,
    driver: Any,
    embedder: Any,
    node_factory: NodeFactory,
    edge_factory: EdgeFactory,
    readback: Callable[[str], bool] | None = None,
    manifest: ProvenanceManifest | None = None,
) -> ProjectionResult:
    """Write prevalidated graph objects through Graphiti's internal bulk writer.

    No Graphiti client ingestion API is called, so no LLM resolution or fact
    invalidation runs.  The caller must pin graphiti-core==0.30.2 and provide
    its bulk writer, driver and embedding implementation.
    """
    _validate(relation, config)
    valid_at = _parse_time(relation.valid_at)
    invalid_at = _parse_time(relation.invalid_at)
    projection_id = deterministic_projection_id(relation)
    input_hash = _input_hash(relation)
    if manifest is not None:
        old = manifest.get(projection_id)
        if old is not None and old.input_hash == input_hash:
            return ProjectionResult(projection_id=projection_id, manifest_recorded=True, skipped_idempotent=True)
    shared = {
        "snapshot_id": relation.snapshot_id,
        "evidence_ids": list(relation.evidence_ids),
        "page_numbers": list(relation.page_numbers),
    }
    source = node_factory(
        uuid=relation.source_paper_id,
        name=f"paper:{relation.source_paper_id}",
        group_id=config.group_id,
        labels=["Paper"],
        attributes=shared,
    )
    target = node_factory(
        uuid=relation.target_paper_id,
        name=f"paper:{relation.target_paper_id}",
        group_id=config.group_id,
        labels=["Paper"],
        attributes=shared,
    )
    edge = edge_factory(
        uuid=projection_id,
        group_id=config.group_id,
        source_node_uuid=relation.source_paper_id,
        target_node_uuid=relation.target_paper_id,
        created_at=datetime.now(timezone.utc),
        name=relation.relation_type,
        fact=relation.statement or relation.relation_type,
        valid_at=valid_at,
        invalid_at=invalid_at,
        attributes={
            **shared,
            "authority_status": "validated",
            "source_paper_id": relation.source_paper_id,
            "target_paper_id": relation.target_paper_id,
            "statement": relation.statement,
            "difference_dimension": relation.difference_dimension,
            "comparability_note": relation.comparability_note,
            "uncertainty": relation.uncertainty,
        },
    )
    await bulk_writer(driver, [], [], [source, target], [edge], embedder)
    if readback is not None and not readback(projection_id):
        raise GraphitiProjectionError(f"readback verification failed for projection_id={projection_id}")
    recorded = False
    if manifest is not None:
        manifest.record(
            ManifestEntry(
                projection_id=projection_id,
                input_hash=input_hash,
                relation_id=relation.relation_id,
                snapshot_id=relation.snapshot_id,
                projected_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        recorded = True
    return ProjectionResult(projection_id=projection_id, manifest_recorded=recorded)


# ---------------------------------------------------------------------------
# Temporal EvolutionEvent projection (direct-bulk, zero LLM)
# ---------------------------------------------------------------------------

def _temporal_input_hash(event: ValidatedTemporalEvent) -> str:
    return _hash(
        "\0".join(
            [
                event.event_id,
                event.snapshot_id,
                event.event_type,
                event.subject,
                event.paper_id,
                event.source_paper_id,
                str(event.year),
                *sorted(event.evidence_ids),
                *map(str, sorted(event.page_numbers)),
                event.valid_at,
                event.invalid_at or "",
                event.statement,
            ]
        ),
        64,
    )


def _validate_temporal(event: ValidatedTemporalEvent, config: ProjectionConfig) -> None:
    for name, value in (
        ("event_id", event.event_id),
        ("event_type", event.event_type),
        ("subject", event.subject),
        ("paper_id", event.paper_id),
        ("snapshot_id", event.snapshot_id),
        ("valid_at", event.valid_at),
    ):
        if not value:
            raise GraphitiProjectionError(f"{name} MUST NOT be empty")
    if not event.evidence_ids:
        raise GraphitiProjectionError("evidence_ids MUST NOT be empty")
    if any(not _GLOBAL_ID.match(item) for item in event.evidence_ids):
        raise GraphitiProjectionError("evidence_ids must be globally scoped")
    if not event.page_numbers:
        raise GraphitiProjectionError("page_numbers MUST NOT be empty")
    if not config.no_authority_reinterpretation:
        raise GraphitiProjectionError("no_authority_reinterpretation MUST be True")
    if config.telemetry_enabled:
        raise GraphitiProjectionError("telemetry_enabled MUST be False")
    if not config.pre_resolved_entities:
        raise GraphitiProjectionError("pre_resolved_entities MUST be True")


def deterministic_temporal_projection_id(event: ValidatedTemporalEvent) -> str:
    fields = [
        event.event_id,
        event.snapshot_id,
        event.paper_id,
        event.event_type,
        event.subject,
        *sorted(event.evidence_ids),
        event.valid_at,
    ]
    return f"proj_{_hash(chr(0).join(fields))}"


async def project_temporal_event(
    relation: ValidatedTemporalEvent,
    config: ProjectionConfig,
    bulk_writer: BulkWriter,
    driver: Any,
    embedder: Any,
    node_factory: NodeFactory,
    edge_factory: EdgeFactory,
    readback: Callable[[str], bool] | None = None,
    manifest: ProvenanceManifest | None = None,
) -> ProjectionResult:
    """Write one prevalidated EvolutionEvent through Graphiti's bulk writer.

    The event semantics (event_id, event_type, subject, year, statement) come
    from the deterministic normalization layer and are never re-adjudicated by
    Graphiti.  The edge carries the same bi-temporal bounds as the original
    ``temporal_projection`` node so a time-cutoff comparison can be done
    against the same cutoff semantics.
    """
    _validate_temporal(relation, config)
    valid_at = _parse_time(relation.valid_at)
    invalid_at = _parse_time(relation.invalid_at) if relation.invalid_at else None
    projection_id = deterministic_temporal_projection_id(relation)
    input_hash = _temporal_input_hash(relation)
    if manifest is not None:
        old = manifest.get(projection_id)
        if old is not None and old.input_hash == input_hash:
            return ProjectionResult(
                projection_id=projection_id,
                manifest_recorded=True,
                skipped_idempotent=True,
            )
    subject_node_uuid = f"subj_{_hash(relation.subject + chr(0) + relation.snapshot_id)}"
    shared = {
        "snapshot_id": relation.snapshot_id,
        "evidence_ids": list(relation.evidence_ids),
        "page_numbers": list(relation.page_numbers),
    }
    paper_node = node_factory(
        uuid=relation.paper_id,
        name=f"paper:{relation.paper_id}",
        group_id=config.group_id,
        labels=["Paper"],
        attributes=shared,
    )
    subject_node = node_factory(
        uuid=subject_node_uuid,
        name=f"subject:{relation.subject}",
        group_id=config.group_id,
        labels=["EvolutionSubject"],
        attributes=shared,
    )
    edge = edge_factory(
        uuid=projection_id,
        group_id=config.group_id,
        source_node_uuid=relation.paper_id,
        target_node_uuid=subject_node_uuid,
        created_at=datetime.now(timezone.utc),
        name=relation.event_type,
        fact=relation.statement or relation.event_type,
        valid_at=valid_at,
        invalid_at=invalid_at,
        attributes={
            **shared,
            "authority_status": "validated",
            "event_id": relation.event_id,
            "event_type": relation.event_type,
            "subject": relation.subject,
            "paper_id": relation.paper_id,
            "source_paper_id": relation.source_paper_id,
            "year": relation.year,
            "statement": relation.statement,
            "difference_dimension": relation.difference_dimension,
            "uncertainty": relation.uncertainty,
        },
    )
    await bulk_writer(driver, [], [], [paper_node, subject_node], [edge], embedder)
    if readback is not None and not readback(projection_id):
        raise GraphitiProjectionError(f"readback verification failed for projection_id={projection_id}")
    recorded = False
    if manifest is not None:
        manifest.record(
            ManifestEntry(
                projection_id=projection_id,
                input_hash=input_hash,
                relation_id=relation.event_id,
                snapshot_id=relation.snapshot_id,
                projected_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        recorded = True
    return ProjectionResult(
        projection_id=projection_id,
        manifest_recorded=recorded,
        subject_node_uuid=subject_node_uuid,
    )
