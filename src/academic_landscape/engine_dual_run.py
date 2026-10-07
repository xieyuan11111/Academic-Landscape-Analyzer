"""Authority-safe assembly helpers for Graphiti projection runs.

This module deliberately *does not* make the projection authoritative.  It
adapts existing SQLite ``EvidenceSpan`` rows into the narrow structured payloads
accepted by the optional engines, and derives Graphiti page provenance directly
from those rows rather than model output or a landscape JSON cache.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, Sequence

from .graphiti_projection import ValidatedRelation, ValidatedTemporalEvent
from .sqlite_store import EvidenceSpan


class AuthorityProjectionError(ValueError):
    """A record cannot cross from the authority layer into a projection."""


class AuthorityEvidenceStore(Protocol):
    def get_evidence(self, evidence_id: str, snapshot_id: str) -> EvidenceSpan | None: ...


def _require_global_evidence_id(evidence_id: str) -> None:
    if not isinstance(evidence_id, str) or "::" not in evidence_id:
        raise AuthorityProjectionError(
            f"evidence_id {evidence_id!r} must be globally scoped with '::'"
        )
    paper_id, local_id = evidence_id.split("::", 1)
    if not paper_id or not local_id:
        raise AuthorityProjectionError(
            f"evidence_id {evidence_id!r} must contain non-empty paper and local parts"
        )


def evidence_record_from_span(
    span: EvidenceSpan,
    *,
    namespace: str = "academic_landscape",
) -> dict[str, Any]:
    """Make a structured evidence record from an already-validated span.

    No document path or surrounding full text is carried across this boundary.
    The SQLite evidence write gate has already established quote/page validity;
    this function only carries the exact quote and its locator forward.
    """
    _require_global_evidence_id(span.evidence_id)
    if span.page_number < 1:
        raise AuthorityProjectionError("page_number must be positive")
    if not span.quote:
        raise AuthorityProjectionError("authority EvidenceSpan quote must not be empty")
    if not namespace:
        raise AuthorityProjectionError("namespace must not be empty")
    return {
        "snapshot_id": span.snapshot_id,
        "paper_id": span.paper_id,
        "evidence_id": span.evidence_id,
        "page_number": span.page_number,
        "quote": span.quote,
        "locator": span.locator,
        "evidence_type": span.evidence_type,
        "artifact_id": span.artifact_id,
        "namespace": namespace,
    }


def evidence_records_from_authority(
    authority: AuthorityEvidenceStore,
    *,
    snapshot_id: str,
    evidence_ids: Sequence[str],
    namespace: str = "academic_landscape",
) -> list[dict[str, Any]]:
    """Resolve a deterministic evidence set from SQLite, failing closed."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for evidence_id in evidence_ids:
        _require_global_evidence_id(evidence_id)
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        span = authority.get_evidence(evidence_id, snapshot_id)
        if span is None:
            raise AuthorityProjectionError(
                f"authority EvidenceSpan not found: {evidence_id!r} under {snapshot_id!r}"
            )
        if span.snapshot_id != snapshot_id:
            raise AuthorityProjectionError(
                f"authority EvidenceSpan snapshot mismatch for {evidence_id!r}"
            )
        records.append(evidence_record_from_span(span, namespace=namespace))
    return records


def _require_aware_time(field: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise AuthorityProjectionError(f"{field} must be a non-empty ISO-8601 time")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise AuthorityProjectionError(f"{field} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise AuthorityProjectionError(f"{field} must be timezone-aware: {value!r}")
    return value


def _relation_value(relation: Any, field: str, default: Any = None) -> Any:
    if isinstance(relation, dict):
        return relation.get(field, default)
    return getattr(relation, field, default)


def relation_from_authority(
    relation: Any,
    authority: AuthorityEvidenceStore,
    *,
    valid_at: str,
    invalid_at: str | None = None,
) -> ValidatedRelation:
    """Build a Graphiti projection relation with pages resolved from SQLite.

    Relation statement/type may originate from the validated FR-04 output, but
    all evidence IDs are looked up again in the authority store.  This closes
    the gap where a cached landscape JSON could otherwise cite a span that was
    rejected or belongs to another snapshot.
    """
    relation_id = _relation_value(relation, "relation_id", "")
    snapshot_id = _relation_value(relation, "snapshot_id", "")
    source_paper_id = _relation_value(relation, "source_paper_id", "")
    target_paper_id = _relation_value(relation, "target_paper_id", "")
    relation_type = _relation_value(relation, "relation_type", "")
    if not all(isinstance(value, str) and value for value in (
        relation_id, snapshot_id, source_paper_id, target_paper_id, relation_type
    )):
        raise AuthorityProjectionError("relation requires non-empty identity fields")

    source_ids = list(_relation_value(relation, "source_evidence_ids", []) or [])
    target_ids = list(_relation_value(relation, "target_evidence_ids", []) or [])
    all_ids = [*source_ids, *target_ids]
    if not all_ids:
        raise AuthorityProjectionError("relation must carry at least one evidence ID")

    records = evidence_records_from_authority(
        authority,
        snapshot_id=snapshot_id,
        evidence_ids=all_ids,
    )
    by_id = {record["evidence_id"]: record for record in records}
    for expected_paper, ids, side in (
        (source_paper_id, source_ids, "source"),
        (target_paper_id, target_ids, "target"),
    ):
        for evidence_id in ids:
            _require_global_evidence_id(evidence_id)
            record = by_id[evidence_id]
            if record["paper_id"] != expected_paper:
                raise AuthorityProjectionError(
                    f"{side} evidence {evidence_id!r} belongs to paper "
                    f"{record['paper_id']!r}, not expected {expected_paper!r}"
                )

    normalized_ids = list(dict.fromkeys(all_ids))
    normalized_pages = list(dict.fromkeys(
        int(by_id[evidence_id]["page_number"]) for evidence_id in normalized_ids
    ))
    return ValidatedRelation(
        relation_id=relation_id,
        source_paper_id=source_paper_id,
        target_paper_id=target_paper_id,
        relation_type=relation_type,
        evidence_ids=normalized_ids,
        page_numbers=normalized_pages,
        valid_at=_require_aware_time("valid_at", valid_at) or "",
        invalid_at=_require_aware_time("invalid_at", invalid_at),
        snapshot_id=snapshot_id,
        authority_status="validated",
        statement=str(_relation_value(relation, "statement", "") or ""),
        difference_dimension=_relation_value(relation, "difference_dimension"),
        comparability_note=str(_relation_value(relation, "comparability_note", "") or ""),
        uncertainty=str(_relation_value(relation, "uncertainty", "") or ""),
    )


def temporal_event_from_authority(
    event: Any,
    authority: AuthorityEvidenceStore,
) -> ValidatedTemporalEvent:
    """Build a Graphiti temporal event with pages resolved from SQLite.

    Mirrors :func:`relation_from_authority`: event_id/event_type/subject/year
    are accepted from the deterministic normalization layer (``timeline_events``),
    but every evidence ID is re-checked against the authority EvidenceSpan
    store before it is allowed to reach Graphiti.
    """
    event_id = _relation_value(event, "event_id", "")
    snapshot_id = _relation_value(event, "snapshot_id", "")
    paper_id = _relation_value(event, "paper_id", "")
    event_type = _relation_value(event, "event_type", "")
    subject = _relation_value(event, "subject", "")
    year = _relation_value(event, "year")
    if not all(isinstance(value, str) and value for value in (
        event_id, snapshot_id, paper_id, event_type, subject
    )):
        raise AuthorityProjectionError("temporal event requires non-empty identity fields")
    if not isinstance(year, int) or isinstance(year, bool):
        raise AuthorityProjectionError("temporal event year must be an int")

    evidence_ids = list(_relation_value(event, "evidence_ids", []) or [])
    if not evidence_ids:
        raise AuthorityProjectionError("temporal event must carry at least one evidence ID")

    records = evidence_records_from_authority(
        authority,
        snapshot_id=snapshot_id,
        evidence_ids=evidence_ids,
    )
    by_id = {record["evidence_id"]: record for record in records}
    for evidence_id in evidence_ids:
        _require_global_evidence_id(evidence_id)
        record = by_id[evidence_id]
        if record["paper_id"] != paper_id:
            raise AuthorityProjectionError(
                f"evidence {evidence_id!r} belongs to paper {record['paper_id']!r}, "
                f"not expected {paper_id!r}"
            )

    normalized_ids = list(dict.fromkeys(evidence_ids))
    normalized_pages = list(dict.fromkeys(
        int(by_id[evidence_id]["page_number"]) for evidence_id in normalized_ids
    ))
    raw_valid = _relation_value(event, "valid_at", "")
    if raw_valid and "T" not in raw_valid:
        raw_valid = f"{raw_valid}T00:00:00+00:00"
    raw_invalid = _relation_value(event, "invalid_at") or None
    if raw_invalid and "T" not in raw_invalid:
        raw_invalid = f"{raw_invalid}T00:00:00+00:00"

    return ValidatedTemporalEvent(
        event_id=event_id,
        event_type=event_type,
        subject=subject,
        paper_id=paper_id,
        year=year,
        evidence_ids=normalized_ids,
        page_numbers=normalized_pages,
        valid_at=_require_aware_time("valid_at", raw_valid) or "",
        invalid_at=_require_aware_time("invalid_at", raw_invalid),
        snapshot_id=snapshot_id,
        statement=str(_relation_value(event, "statement", "") or ""),
        source_paper_id=str(_relation_value(event, "source_paper_id", "") or ""),
        difference_dimension=_relation_value(event, "difference_dimension"),
        uncertainty=str(_relation_value(event, "uncertainty", "") or ""),
    )


__all__ = [
    "AuthorityProjectionError",
    "AuthorityEvidenceStore",
    "evidence_record_from_span",
    "evidence_records_from_authority",
    "relation_from_authority",
    "temporal_event_from_authority",
]
