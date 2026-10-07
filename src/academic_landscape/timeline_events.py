"""EvolutionEvent temporal-graph normalisation (PRD §6.3, contract v0.1).

Two-track split: the temporal-graph track is responsible for time-ordered
evolution (this module); the direct-extraction track handles snapshot
semantics (direction_naming / cross_paper). Core principle — **sorting
is code, comprehension is LLM, both anchored to EvidenceSpan.**

This module does deterministic mapping only (thread -> events,
relation -> events, frontier -> events); the optional :func:`classify_event`
LLM call is reserved for edge cases and is mock-tested.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, Sequence

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Constants / vocabulary
# ---------------------------------------------------------------------------

EVENT_TYPES: frozenset[str] = frozenset({
    "concept_origin",
    "explicit_proposal",
    "first_operationalization",
    "initial_validation",
    "replication",
    "object_extension",
    "method_innovation",
    "mechanism_extension",
    "boundary_revision",
    "significant_challenge",
    "counterexample",
    "synthesis_or_alternative",
})

CURRENT_YEAR: int = datetime.now().year

# Static entry in FR-04 -> event mapping. ``extends`` is a dispatch key:
# its event type depends on ``difference_dimension`` (see
# :func:`relation_to_event_type`). ``supports`` has a static default of
# ``initial_validation``; the ``normalize_events`` runner may demote later
# supports to ``replication``.
_RELATION_TO_EVENT_SENTINEL = "_DIMENSION_DISPATCH"
_RELATION_TO_EVENT_DEFAULT: dict[str, str] = {
    "extends": _RELATION_TO_EVENT_SENTINEL,
    "supports": "initial_validation",
    "revises": "boundary_revision",
    "qualifies": "boundary_revision",
    "challenges": "significant_challenge",
    "contradicts": "counterexample",
}

RELATION_TO_EVENT: dict[str, str] = dict(_RELATION_TO_EVENT_DEFAULT)

# Extends dispatch table: dimension -> event_type. An extends without
# (or with an out-of-vocabulary) dimension produces no event.
_EXTENDS_DIMENSION_TO_EVENT: dict[str, str] = {
    "object": "object_extension",
    "conclusion": "mechanism_extension",
    "method": "method_innovation",
}

# Relation types that do NOT produce evolution events.
_NON_EVOLUTION_RELATIONS: frozenset[str] = frozenset({
    "parallel",
    "not_comparable",
    "unknown",
    "surface_conflict",
    "conditional_difference",
    "same_name_different_meaning",
})

# Event types that must NOT have source_paper_id == paper_id (self-reference).
_SELF_REF_FORBIDDEN: frozenset[str] = frozenset({
    "object_extension",
    "mechanism_extension",
    "method_innovation",
    "boundary_revision",
    "significant_challenge",
    "counterexample",
})

# Event types permitted to set invalid_at. Per PRD §6.3 hard constraint,
# invalid_at is ONLY a defensible signal for an explicit counterexample.
_INVALID_AT_ALLOWED: frozenset[str] = frozenset({"counterexample"})


# ---------------------------------------------------------------------------
# Pydantic contract
# ---------------------------------------------------------------------------


class EvolutionEvent(BaseModel):
    """Evidence-bound evolution event (contract §4, ``extra=\"forbid\"``)."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: str
    subject: str
    paper_id: str
    source_paper_id: str = ""
    year: int
    evidence_ids: tuple[str, ...]
    statement: str = ""
    difference_dimension: str | None = None
    valid_at: str = ""
    invalid_at: str = ""
    snapshot_id: str
    analysis_version: str = ""
    uncertainty: str = ""


# ---------------------------------------------------------------------------
# Bundle / client
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineBundle:
    """Normalisation result: accepted events + rejected audit traces."""

    events: tuple[EvolutionEvent, ...]
    ignored_events: tuple[dict, ...]
    validation_errors: tuple[dict, ...]


class _EventClient(Protocol):
    """Protocol for an LLM client capable of JSON completion."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Deterministic event id
# ---------------------------------------------------------------------------


def deterministic_event_id(
    snapshot_id: str,
    paper_id: str,
    event_type: str,
    subject: str,
    evidence_ids: Sequence[str],
) -> str:
    """SHA256-derived event id; evidence order-invariant, snapshot-scoped.

    Args:
        snapshot_id: Snapshot identifier (isolation across runs).
        paper_id: Carrying paper id.
        event_type: Event type literal (must be in :data:`EVENT_TYPES`).
        subject: Subject being evolved.
        evidence_ids: Any iterable of evidence IDs (deduplicated + sorted).

    Returns:
        ``evo_<16-hex>`` deterministic identifier.
    """
    deduped_sorted = sorted(set(evidence_ids))
    payload = "|".join((
        snapshot_id,
        paper_id,
        event_type,
        subject,
        ",".join(deduped_sorted),
    ))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"evo_{digest}"


# ---------------------------------------------------------------------------
# FR-04 relation -> event mapping (contract §3)
# ---------------------------------------------------------------------------


def relation_to_event_type(
    relation_type: str,
    difference_dimension: str | None,
) -> str | None:
    """Map an FR-04 relation to a v0.1 event_type, or ``None`` if no event.

    A ``None`` return means the relation does not produce an evolution
    event (e.g. ``parallel``, ``not_comparable``, ``unknown``,
    ``surface_conflict``, ``conditional_difference``,
    ``same_name_different_meaning`` — see contract §3 bottom paragraph).

    For ``extends`` the result depends on ``difference_dimension``:
    ``object`` -> ``object_extension``, ``conclusion`` ->
    ``mechanism_extension``, ``method`` -> ``method_innovation``.
    ``extends`` without a recognised dimension returns ``None``.

    For ``supports`` we always return ``initial_validation`` as the
    static default; ``normalize_events`` may demote subsequent
    supports to ``replication`` based on input order.
    """
    if relation_type in _NON_EVOLUTION_RELATIONS:
        return None
    if relation_type == "extends":
        if difference_dimension is None:
            return None
        return _EXTENDS_DIMENSION_TO_EVENT.get(difference_dimension)
    return _RELATION_TO_EVENT_DEFAULT.get(relation_type)


def relation_drop_reason(relation_type: str, difference_dimension: str | None) -> str | None:
    """Say *why* a relation produced no event, or ``None`` if it produced one.

    ``relation_to_event_type`` collapses two very different situations into
    ``None``: "this relation type produces no event by design" and "this one
    should have produced one, but a field is missing or unrecognised". Keeping
    them under one audit reason made the MAG-50 r5 trace read
    ``not_an_evolution_event`` for three ``extends`` relations whose real problem
    was a missing ``difference_dimension`` — a data-quality loss filed as a
    policy decision. Callers must record this reason instead.
    """
    if relation_to_event_type(relation_type, difference_dimension) is not None:
        return None
    if relation_type == "extends":
        if difference_dimension is None:
            return "missing_difference_dimension"
        return "unrecognised_difference_dimension"
    return "not_an_evolution_event"


# ---------------------------------------------------------------------------
# LLM boundary (optional, mock-tested)
# ---------------------------------------------------------------------------


_EVENT_PROMPT_SYSTEM = """You adjudicate ambiguous evolution-event boundaries. Given
one FR-04 relation (subject=temporal-relation), assign the most precise
event_type from the v0.1 twelve-class vocabulary, OR return a single string
"none" if the relation is not an evolution event.
Rules:
- Use ONLY the supplied evidence ids. Do not invent ids.
- For extends you MUST include difference_dimension when relevant.
- Return JSON: {\"event_type\": \"...\", \"reason\": \"...\"}."""


def build_event_prompt(subject: str, event_type: str, context: str) -> tuple[str, str]:
    """Build (system, user) prompts for the optional classifier."""
    user_prompt = (
        f"SUBJECT: {subject}\n"
        f"HINT EVENT_TYPE: {event_type}\n"
        f"CONTEXT: {context}\n\n"
        "Return JSON only."
    )
    return _EVENT_PROMPT_SYSTEM, user_prompt


def classify_event(
    client: _EventClient,
    relation: dict,
    years: dict[str, int],
) -> str:
    """Optional LLM adjudication for ambiguous boundaries.

    Args:
        client: LLM client implementing ``complete_json``.
        relation: FR-04 relation dict (must carry ``relation_type``).
        years: Mapping paper_id -> year (used for chronology context).

    Returns:
        A string event_type from :data:`EVENT_TYPES` (or a sentinel like
        ``"none"`` if the client chose to drop the event).
    """
    subject = relation.get("target_paper_id", relation.get("statement", ""))
    hint = relation.get("relation_type", "")
    context = (
        f"source_paper={relation.get('source_paper_id')} "
        f"target_paper={relation.get('target_paper_id')} "
        f"year_source={years.get(relation.get('source_paper_id', ''))} "
        f"year_target={years.get(relation.get('target_paper_id', ''))}"
    )
    system_prompt, user_prompt = build_event_prompt(subject, hint, context)
    payload = client.complete_json(system_prompt, user_prompt)
    if not isinstance(payload, dict):
        return "none"
    return str(payload.get("event_type", "none"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _globalize_evidence_id(paper_id: str, evidence_id: str) -> str:
    """Return ``paper_id::local_id`` form; pass-through if already global."""
    return evidence_id if "::" in evidence_id else f"{paper_id}::{evidence_id}"


def _build_allowed_and_owners(
    extraction_by_id: dict[str, dict],
) -> tuple[set[str], dict[str, str]]:
    """Build (allowed_global_ids, owner_map) from already-globalized extractions.

    Each paper's ``evidence`` list contains entries with global form
    ``paper_id::E1``; that is the contract's input posture.
    """
    allowed: set[str] = set()
    owners: dict[str, str] = {}
    for paper_id, extraction in extraction_by_id.items():
        if not isinstance(extraction, dict):
            continue
        evidence_list = extraction.get("evidence", [])
        if not isinstance(evidence_list, list):
            continue
        for evidence_item in evidence_list:
            if not isinstance(evidence_item, dict):
                continue
            eid = evidence_item.get("evidence_id")
            if isinstance(eid, str) and eid:
                # Accept both local ("E1") and global ("paper_a::E1") forms;
                # normalize to global so relation-cited ids resolve (contract §5.2).
                global_eid = eid if "::" in eid else f"{paper_id}::{eid}"
                allowed.add(global_eid)
                owners.setdefault(global_eid, paper_id)
    return allowed, owners


def _thread_paper_id_and_evidence(
    thread: Any,
    paper_id: str,
    evidence_owners: dict[str, str],
) -> tuple[str, tuple[str, ...]]:
    """Find the canonical earliest paper for ``thread`` and filter
    evidence ids to those owned by that paper.
    """
    # Pick earliest paper from thread.paper_ids (by year from `years`).
    paper_ids = list(getattr(thread, "paper_ids", ()) or ())
    if paper_id:
        canonical_paper = paper_id
    else:
        canonical_paper = paper_ids[0] if paper_ids else ""
    thread_evidence = tuple(getattr(thread, "evidence_ids", ()) or ())
    filtered = tuple(eid for eid in thread_evidence
                     if evidence_owners.get(eid) == canonical_paper)
    return canonical_paper, filtered


def _normalise_year(year: Any) -> int | None:
    if isinstance(year, bool):
        return None
    if isinstance(year, int):
        return year
    if isinstance(year, str):
        try:
            return int(year)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Event producers
# ---------------------------------------------------------------------------


def _thread_events(
    reconstructions: Sequence[Any],
    evidence_owners: dict[str, str],
    years: dict[str, int],
    snapshot_id: str,
    analysis_version: str,
    current_year: int | None = None,
) -> tuple[list[EvolutionEvent], list[dict]]:
    """Emit concept_origin / first_operationalization / method_innovation.

    Heuristic:
      * kind=concept (or unknown) thread -> concept_origin at earliest paper
      * kind=method thread -> first_operationalization at earliest paper
      * A method thread whose earliest_year is in the last 3 years
        also produces method_innovation (the thread is newly born and
        therefore itself a method innovation). Threads older produce
        only first_operationalization to avoid double counting.

    ``current_year`` bounds the "recently born" window.  It defaults to the
    latest paper year in the corpus, which is the meaningful horizon for a
    closed corpus, and only falls back to the import-time constant when the
    corpus carries no years at all.
    """
    out: list[EvolutionEvent] = []
    ignored: list[dict] = []

    horizon = (
        int(current_year) if current_year is not None
        else (max(years.values()) if years else CURRENT_YEAR)
    )
    method_window = 3  # treat births in last N years as innovation
    for recon in reconstructions:
        threads = getattr(recon, "threads", ()) or ()
        for thread in threads:
            kind = getattr(thread, "kind", "concept")
            paper_ids = list(getattr(thread, "paper_ids", ()) or ())
            years_list = list(getattr(thread, "years", ()) or ())
            if not paper_ids or not years_list:
                continue
            earliest_paper = paper_ids[0]
            earliest_year = years_list[0]
            paper_id, evidence_ids = _thread_paper_id_and_evidence(
                thread, earliest_paper, evidence_owners
            )
            if not evidence_ids:
                # No evidence from this paper -> thread event not produced
                ignored.append({
                    "thread": getattr(thread, "canonical_name", ""),
                    "kind": kind,
                    "reason": "missing_evidence_for_thread_origin",
                })
                continue
            canonical = getattr(thread, "canonical_name", "")

            if kind == "method":
                out.append(EvolutionEvent(
                    event_id=deterministic_event_id(
                        snapshot_id, paper_id, "first_operationalization",
                        canonical, evidence_ids,
                    ),
                    event_type="first_operationalization",
                    subject=canonical,
                    paper_id=paper_id,
                    year=earliest_year,
                    evidence_ids=evidence_ids,
                    valid_at=f"{earliest_year}-01-01",
                    snapshot_id=snapshot_id,
                    analysis_version=analysis_version,
                ))
                current_year = horizon
                if earliest_year >= current_year - method_window + 1:
                    out.append(EvolutionEvent(
                        event_id=deterministic_event_id(
                            snapshot_id, paper_id, "method_innovation",
                            canonical, evidence_ids,
                        ),
                        event_type="method_innovation",
                        subject=canonical,
                        paper_id=paper_id,
                        year=earliest_year,
                        evidence_ids=evidence_ids,
                        valid_at=f"{earliest_year}-01-01",
                        snapshot_id=snapshot_id,
                        analysis_version=analysis_version,
                    ))
            else:
                out.append(EvolutionEvent(
                    event_id=deterministic_event_id(
                        snapshot_id, paper_id, "concept_origin",
                        canonical, evidence_ids,
                    ),
                    event_type="concept_origin",
                    subject=canonical,
                    paper_id=paper_id,
                    year=earliest_year,
                    evidence_ids=evidence_ids,
                    valid_at=f"{earliest_year}-01-01",
                    snapshot_id=snapshot_id,
                    analysis_version=analysis_version,
                ))
    return out, ignored


def _relation_events(
    relation_results: Sequence[Any],
    evidence_owners: dict[str, str],
    years: dict[str, int],
    snapshot_id: str,
    analysis_version: str,
) -> tuple[list[EvolutionEvent], list[dict], list[dict]]:
    """Emit relation-derived events per contract §3.

    Reordering: relations are sorted by source-paper year asc so that
    the first supports to a given subject gets ``initial_validation``
    and subsequent supports become ``replication``. When multiple
    relations of the same (source_paper, target_paper, relation_type)
    appear, the FIRST one wins.
    """
    raw_relations: list[dict] = []
    for rr in relation_results:
        for rel in getattr(rr, "relations", ()) or ():
            if isinstance(rel, dict):
                raw_relations.append(rel)
            elif hasattr(rel, "model_dump"):
                # PaperRelation (pydantic) from the real pipeline
                raw_relations.append(rel.model_dump(mode="json"))
            elif hasattr(rel, "__dict__"):
                raw_relations.append(dict(rel.__dict__))
            else:
                raise TypeError(
                    "relation must be a dict or PaperRelation-like object"
                )

    # Sort by source-paper year asc; ties broken by source paper id asc
    # for deterministic behaviour across runs.
    def _sort_key(rel: dict) -> tuple[int, str]:
        year = years.get(rel.get("source_paper_id", ""), 10**9)
        return (year, rel.get("source_paper_id", ""))

    raw_relations.sort(key=_sort_key)

    events: list[EvolutionEvent] = []
    ignored: list[dict] = []
    errors: list[dict] = []
    supports_seen_subjects: set[tuple[str, str]] = set()  # (relation_type, subject)

    for raw in raw_relations:
        rel_type = raw.get("relation_type", "")
        diff_dim = raw.get("difference_dimension")
        event_type = relation_to_event_type(rel_type, diff_dim)
        if event_type is None:
            # Distinguish "no event by design" from "should have been an event,
            # but the data is incomplete" — one audit reason for both hides the
            # difference exactly where a later fix would look for it.
            ignored.append({
                "reason": relation_drop_reason(rel_type, diff_dim),
                "relation_type": rel_type,
                "difference_dimension": diff_dim,
                "source_paper_id": raw.get("source_paper_id", ""),
                "target_paper_id": raw.get("target_paper_id", ""),
            })
            continue

        source_paper = raw.get("source_paper_id", "")
        target_paper = raw.get("target_paper_id", "")
        source_ev = list(raw.get("source_evidence_ids", []) or [])
        # target_evidence_ids is NOT bound to the event: FR-04 dual-evidence
        # gate already validated it at relation level (contract §3/§4 — the
        # event carries only its carrier paper's evidence).

        # Pick carrier paper & evidence set:
        # - source-as-carrier events (extends, challenges, contradicts,
        #   revises, qualifies) bind source evidence; subject = target.
        # - supports (initial_validation / replication) bind only source
        #   evidence because the source paper is the carrier; target
        #   evidence is recorded under source_paper_id.
        if rel_type == "supports":
            subject = target_paper
            paper_id = source_paper
            evidence_ids = source_ev
            if (rel_type, subject) not in supports_seen_subjects:
                event_type = "initial_validation"
                supports_seen_subjects.add((rel_type, subject))
            else:
                event_type = "replication"
        else:
            subject = target_paper
            paper_id = source_paper
            evidence_ids = source_ev

        if not evidence_ids:
            ignored.append({
                "reason": "missing_evidence",
                "relation_type": rel_type,
                "source_paper_id": source_paper,
                "target_paper_id": target_paper,
            })
            continue

        statement = str(raw.get("statement", ""))
        year_raw = years.get(source_paper)
        if not isinstance(year_raw, int):
            year_raw = years.get(target_paper, 0)
        out_event = EvolutionEvent(
            event_id=deterministic_event_id(
                snapshot_id, paper_id, event_type, subject, evidence_ids,
            ),
            event_type=event_type,
            subject=subject,
            paper_id=paper_id,
            source_paper_id=target_paper,
            year=int(year_raw),
            evidence_ids=tuple(evidence_ids),
            statement=statement,
            difference_dimension=diff_dim,
            valid_at=f"{int(year_raw)}-01-01",
            snapshot_id=snapshot_id,
            analysis_version=analysis_version,
            uncertainty=str(raw.get("uncertainty", "")),
        )
        events.append(out_event)

    return events, ignored, errors


def _frontier_events(
    merged_frontiers: Sequence[Any],
    evidence_owners: dict[str, str],
    years: dict[str, int],
    snapshot_id: str,
    analysis_version: str,
) -> tuple[list[EvolutionEvent], list[dict]]:
    """Emit synthesis_or_alternative only for tier=emerging_frontier.

    Frontiers are direction-level signals. To respect the
    paper-scoped event model we anchor the event to the owner paper of
    the first evidence id; if no evidence we drop and audit.
    """
    events: list[EvolutionEvent] = []
    ignored: list[dict] = []
    for frontier in merged_frontiers:
        tier = getattr(frontier, "tier", "")
        if tier != "emerging_frontier":
            continue
        statement = str(getattr(frontier, "statement", ""))
        evidence = list(getattr(frontier, "evidence", ()) or ())
        if not evidence:
            ignored.append({
                "reason": "frontier_missing_evidence",
                "statement": statement,
                "tier": tier,
            })
            continue
        owner_counter = Counter()
        for eid in evidence:
            owner = evidence_owners.get(eid)
            if owner:
                owner_counter[owner] += 1
        if not owner_counter:
            ignored.append({
                "reason": "frontier_unknown_evidence",
                "statement": statement,
                "tier": tier,
            })
            continue
        paper_id = owner_counter.most_common(1)[0][0]
        owned = tuple(eid for eid in evidence if evidence_owners.get(eid) == paper_id)
        if not owned:
            ignored.append({
                "reason": "frontier_no_owned_evidence",
                "statement": statement,
                "tier": tier,
            })
            continue
        year = years.get(paper_id, 0)
        events.append(EvolutionEvent(
            event_id=deterministic_event_id(
                snapshot_id, paper_id, "synthesis_or_alternative", statement, owned,
            ),
            event_type="synthesis_or_alternative",
            subject=statement,
            paper_id=paper_id,
            year=year,
            evidence_ids=owned,
            valid_at=f"{year}-01-01",
            snapshot_id=snapshot_id,
            analysis_version=analysis_version,
        ))
    return events, ignored


# ---------------------------------------------------------------------------
# Validation gates (contract §5) — exposed for direct testing
# ---------------------------------------------------------------------------


def _validate_events(
    events: Sequence[EvolutionEvent],
    *,
    extractions_by_paper: dict[str, dict],
    years: dict[str, int],
    current_year: int | None = None,
) -> tuple[tuple[EvolutionEvent, ...], tuple[dict, ...], tuple[dict, ...]]:
    """Apply the seven §5 gates to a list of candidate events.

    Returns ``(accepted, ignored, errors)``. Each violation lands in
    ``ignored`` (data-shape problem) or ``errors`` (semantic problem)
    so downstream callers can audit both streams.

    ``current_year`` bounds Gate 3.  It must be supplied by the caller rather
    than read from a module-level constant: a long run that crosses New Year
    would otherwise keep the year frozen at import time and silently drop the
    new year's papers as ``year_mismatch`` — indistinguishable from a paper
    whose year was genuinely inferred wrong.
    """
    year_ceiling = CURRENT_YEAR if current_year is None else int(current_year)
    allowed_evidence, evidence_owners = _build_allowed_and_owners(extractions_by_paper)

    accepted: list[EvolutionEvent] = []
    ignored: list[dict] = []
    errors: list[dict] = []

    for raw_event in events:
        # --- Gate 1: event_type vocabulary ---
        if raw_event.event_type not in EVENT_TYPES:
            ignored.append({
                "reason": "invalid_event_type",
                "event_type": raw_event.event_type,
                "event_id": raw_event.event_id,
            })
            continue

        # --- Gate 2: evidence_ids presence + ownership ---
        if not raw_event.evidence_ids:
            ignored.append({
                "reason": "missing_evidence",
                "event_id": raw_event.event_id,
                "event_type": raw_event.event_type,
            })
            continue
        evidence_problem = None
        for eid in raw_event.evidence_ids:
            if eid not in allowed_evidence:
                evidence_problem = ("unknown_evidence_id", eid)
                break
            owner = evidence_owners.get(eid)
            if owner != raw_event.paper_id:
                evidence_problem = ("evidence_owner_mismatch", eid)
                break
        if evidence_problem is not None:
            code, eid = evidence_problem
            ignored.append({
                "reason": code,
                "event_id": raw_event.event_id,
                "evidence_id": eid,
                "paper_id": raw_event.paper_id,
            })
            continue

        # --- Gate 3: year range + carrier paper year consistency ---
        carrier_year = years.get(raw_event.paper_id)
        if (not isinstance(carrier_year, int)
                or raw_event.year != carrier_year
                or raw_event.year < 1900
                or raw_event.year > year_ceiling):
            # A year past the ceiling is a different failure from a year that
            # disagrees with its carrier, and the audit trail must not merge
            # the two: one is a data problem, the other is a clock problem.
            beyond_horizon = (
                isinstance(raw_event.year, int)
                and isinstance(carrier_year, int)
                and raw_event.year > year_ceiling
            )
            ignored.append({
                "reason": "year_beyond_current_year" if beyond_horizon else "year_mismatch",
                "event_id": raw_event.event_id,
                "year": raw_event.year,
                "carrier_year": carrier_year,
                "current_year": year_ceiling,
            })
            continue

        # --- Gate 4: invalid_at only allowed for counterexample ---
        if raw_event.invalid_at and raw_event.event_type not in _INVALID_AT_ALLOWED:
            ignored.append({
                "reason": "invalid_at_not_allowed",
                "event_id": raw_event.event_id,
                "event_type": raw_event.event_type,
            })
            continue

        # --- Gate 5: self-reference forbidden for "extending" event types ---
        if (raw_event.event_type in _SELF_REF_FORBIDDEN
                and raw_event.paper_id == raw_event.source_paper_id):
            ignored.append({
                "reason": "self_reference",
                "event_id": raw_event.event_id,
                "event_type": raw_event.event_type,
            })
            continue

        # --- Gate 7: valid_at <= invalid_at when invalid_at is set/dated ---
        if raw_event.invalid_at and raw_event.invalid_at != "open":
            if raw_event.valid_at and raw_event.valid_at > raw_event.invalid_at:
                ignored.append({
                    "reason": "invalid_time_range",
                    "event_id": raw_event.event_id,
                    "valid_at": raw_event.valid_at,
                    "invalid_at": raw_event.invalid_at,
                })
                continue

        accepted.append(raw_event)

    return tuple(accepted), tuple(ignored), tuple(errors)


def _dedupe_events(
    events: Sequence[EvolutionEvent],
) -> tuple[tuple[EvolutionEvent, ...], tuple[dict, ...]]:
    """Apply §5 gate 6: same (paper, event_type, subject) -> keep first.

    The keep-first ordering is the deterministic input order — callers
    pre-sort for reproducibility (e.g. by year asc).
    """
    seen: set[tuple[str, str, str]] = set()
    kept: list[EvolutionEvent] = []
    ignored: list[dict] = []
    for event in events:
        key = (event.paper_id, event.event_type, event.subject)
        if key in seen:
            ignored.append({
                "reason": "duplicate_event",
                "kept_event_id": next(e.event_id for e in kept
                                      if (e.paper_id, e.event_type, e.subject) == key),
                "dropped_event_id": event.event_id,
                "paper_id": event.paper_id,
                "event_type": event.event_type,
                "subject": event.subject,
            })
            continue
        seen.add(key)
        kept.append(event)
    return tuple(kept), tuple(ignored)


# ---------------------------------------------------------------------------
# Public entry — bundle assembler
# ---------------------------------------------------------------------------


def normalize_events(
    relation_results: Sequence[Any],
    reconstructions: Sequence[Any],
    merged_frontiers: Sequence[Any],
    extraction_by_id: dict[str, dict],
    years: dict[str, int],
    snapshot_id: str,
    analysis_version: str = "",
    current_year: int | None = None,
) -> TimelineBundle:
    """Build a :class:`TimelineBundle` from three input streams.

    Steps:
      1. Derive deterministic evidence-meta from ``extraction_by_id``.
      2. Produce thread-derived events (concept_origin /
         first_operationalization / method_innovation).
      3. Produce relation-derived events (FR-04 -> event_type mapping).
      4. Produce frontier-derived events (synthesis_or_alternative).
      5. Concat; apply dedup; apply validation gates; assemble bundle.

    Args:
        current_year: Upper bound for event years.  Pass the year the analysis
            is *for*, not the year the process happened to start in — a run
            that spans New Year must not keep the import-time year.  ``None``
            falls back to :data:`CURRENT_YEAR` for backward compatibility.

    Returns:
        TimelineBundle of accepted events plus audit traces.
    """
    year_ceiling = CURRENT_YEAR if current_year is None else int(current_year)
    _, evidence_owners = _build_allowed_and_owners(extraction_by_id)

    # _thread_events gets the RAW argument, not the resolved ceiling: its
    # "recently born" window has always defaulted to the latest paper year in
    # the corpus, and resolving None to CURRENT_YEAR here would silently widen
    # that window.  Only the validation gate uses the resolved ceiling.
    thread_events, thread_ignored = _thread_events(
        reconstructions, evidence_owners, years, snapshot_id, analysis_version,
        current_year=current_year,
    )
    relation_events, relation_ignored, relation_errors = _relation_events(
        relation_results, evidence_owners, years, snapshot_id, analysis_version,
    )
    frontier_events, frontier_ignored = _frontier_events(
        merged_frontiers, evidence_owners, years, snapshot_id, analysis_version,
    )

    # Stable ordering: by year asc, then event_type asc, then paper_id asc.
    raw = thread_events + relation_events + frontier_events
    raw.sort(key=lambda e: (e.year, e.event_type, e.paper_id, e.subject))

    deduped, dup_ignored = _dedupe_events(raw)
    accepted, rejected_ignored, rejected_errors = _validate_events(
        list(deduped),
        extractions_by_paper=extraction_by_id,
        years=years,
        current_year=year_ceiling,
    )

    ignored_events: list[dict] = []
    ignored_events.extend(thread_ignored)
    ignored_events.extend(relation_ignored)
    ignored_events.extend(frontier_ignored)
    ignored_events.extend(dup_ignored)
    ignored_events.extend(list(rejected_ignored))

    return TimelineBundle(
        events=tuple(accepted),
        ignored_events=tuple(ignored_events),
        validation_errors=tuple(list(rejected_errors) + relation_errors),
    )


# ---------------------------------------------------------------------------
# Cutoff filter (contract §8)
# ---------------------------------------------------------------------------


def filter_by_cutoff(
    events: Sequence[EvolutionEvent],
    cutoff_at: str,
) -> tuple[EvolutionEvent, ...]:
    """Return events that are valid at ``cutoff_at`` and not yet invalid.

    Semantics (contract §8): ``valid_at <= cutoff_at`` AND
    (``invalid_at == "open"`` OR ``invalid_at == ""`` OR
    ``invalid_at > cutoff_at``).
    """
    accepted: list[EvolutionEvent] = []
    for event in events:
        valid_at = event.valid_at or f"{event.year}-01-01"
        if valid_at > cutoff_at:
            continue
        invalid_at = event.invalid_at
        if not invalid_at or invalid_at == "open" or invalid_at > cutoff_at:
            accepted.append(event)
    return tuple(accepted)
