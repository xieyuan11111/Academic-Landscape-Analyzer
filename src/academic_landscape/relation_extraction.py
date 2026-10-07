"""FR-04: Per-paper relation extraction with evidence-bound validation.

This module extracts structured relations between paper pairs within a research direction,
validates evidence ownership and relation semantics, and returns audit-ready results.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, Sequence

from pydantic import BaseModel

from .direction_naming import ResearchDirection
from .llm_client import is_retryable_completion
from .relation_types import (
    DENSITY_FLOORED_RELATIONS,
    DENSITY_MIN_PER_SIDE,
    RELATION_PROMPT_VOCABULARY,
    VALID_RELATION_TYPES,
)

# Structured failure code recorded when a direction loses its whole relation layer.
RELATION_EXTRACTION_FAILED = "relation_extraction_failed"

# Recorded on individual bounded-chunk calls that failed during escalation.
PAIR_CHUNK_FAILED = "pair_chunk_failed"

# Recorded when a bounded-chunk call answers about a pair it was not asked about.
PAIR_CHUNK_OUT_OF_SCOPE = "pair_chunk_out_of_scope"

# Truncation and malformed JSON are non-deterministic (measured 2026-09-24: the
# identical request truncated once and finished cleanly the next time), so a
# bounded retry recovers real work. The happy path still makes exactly one call.
_RELATION_MAX_ATTEMPTS = 3

# Escalation for directions whose answer never fits the completion ceiling.
# Measured on MAG-50 r3: a 13-paper direction (78 pairs) truncated on all three
# retries, because the request asks for a verdict on every paper pair at once.
# Retrying cannot fix a request whose output does not fit, so such a direction is
# re-run as small bounded groups of pairs that cover every pair exactly once.
_RELATION_CHUNK_SIZE = 12


def iter_pair_chunks(
    paper_ids: Sequence[str], chunk_size: int = _RELATION_CHUNK_SIZE
) -> Iterator[tuple[tuple[str, str], ...]]:
    """Yield deterministic groups of paper pairs covering every pair once.

    ``itertools.combinations`` over the given order keeps this reproducible
    (PRD 14.3): the same direction always produces the same chunks.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    pairs = list(itertools.combinations(paper_ids, 2))
    for start in range(0, len(pairs), chunk_size):
        yield tuple(pairs[start : start + chunk_size])


def _is_retryable(error: BaseException) -> bool:
    """Whether another draw is worth paying for.

    Delegates to :func:`llm_client.is_retryable_completion` so the retry policy
    is defined once; stage reconstruction uses the same rule.
    """
    return is_retryable_completion(error)

# Relation types requiring target evidence (dual evidence gate)
RELATIONS_REQUIRING_TARGET_EVIDENCE = frozenset({"challenges", "contradicts"})

# Relation types requiring difference_dimension. Declared as an ordered tuple so
# the prompt renders them in a stable order and the gate set cannot drift from it.
_DIFFERENCE_DIMENSION_RELATIONS: tuple[str, ...] = (
    "same_name_different_meaning",
    "surface_conflict",
    "conditional_difference",
)
RELATIONS_REQUIRING_DIFFERENCE_DIMENSION = frozenset(_DIFFERENCE_DIMENSION_RELATIONS)

# Allowed difference_dimension values
VALID_DIFFERENCE_DIMENSIONS = frozenset({
    "definition",
    "object",
    "method",
    "conditions",
    "conclusion",
})

# System prompt for relation extraction
_RELATION_SYSTEM = """You identify structured relations between specific paper pairs within ONE research direction.
Given each paper's key results, A-level claims, and evidence quotes, produce relations as JSON.
Rules:
- Use ONLY the given evidence IDs in your output. Never invent new evidence IDs.
- Each relation must explicitly cite source_evidence_ids (from the source paper) and optionally
  target_evidence_ids (from the target paper, required for challenges/contradictions).
- A "difference" is NOT the same as a "contradiction". Distinguish:
  - contradiction/challenge: directly opposing claims with evidence from both sides
  - conditional_difference: same finding applies under different conditions
  - surface_conflict: apparent disagreement but actually talking about different things
  - same_name_different_meaning: same term used with different definitions
- If you cannot determine the relation, use relation_type "unknown" (target_evidence_ids can be empty).
- If papers are not comparable (different domains, methodologies, or scopes), use "not_comparable".
Return JSON only."""


class PaperRelation(BaseModel, extra="forbid"):
    """Evidence-bound relation between two papers."""

    relation_id: str
    source_paper_id: str
    target_paper_id: str
    relation_type: str
    statement: str
    source_evidence_ids: list[str]
    target_evidence_ids: list[str]
    difference_dimension: str | None = None
    comparability_note: str = ""
    uncertainty: str = ""
    snapshot_id: str


@dataclass(frozen=True)
class RelationExtractionResult:
    """Result of relation extraction for one research direction."""

    direction_id: str
    relations: tuple[PaperRelation, ...]
    ignored_relations: tuple[dict, ...]
    validation_errors: tuple[dict, ...]
    # Relations that stayed valid but cited fewer than DENSITY_MIN_PER_SIDE
    # spans on some side. Advisory: these are under-cited, not wrong. Recorded
    # so a report can disclose evidence density and so a caller can decide
    # which directions deserve a re-run under the stricter prompt.
    density_warnings: tuple[dict, ...] = ()
    # Non-failure record of how the layer was produced, e.g. that an oversized
    # direction had to be re-run as bounded pair chunks. Kept separate from
    # validation_errors because recovering is not a problem to report as one.
    extraction_trace: tuple[dict, ...] = ()


class RelationClient(Protocol):
    """Protocol for LLM client capable of JSON completion."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


def deterministic_relation_id(
    snapshot_id: str,
    source_paper_id: str,
    target_paper_id: str,
    relation_type: str,
    evidence_ids: list[str],
) -> str:
    """Generate deterministic relation_id using SHA256 hash.

    Args:
        snapshot_id: The snapshot identifier
        source_paper_id: Source paper identifier
        target_paper_id: Target paper identifier
        relation_type: Type of relation
        evidence_ids: List of evidence IDs (used for uniqueness)

    Returns:
        Relation ID with 'rel_' prefix and 16-char hex digest
    """
    sorted_evidence = sorted(evidence_ids)
    payload = "|".join((
        snapshot_id,
        source_paper_id,
        target_paper_id,
        relation_type,
        ",".join(sorted_evidence),
    ))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"rel_{digest}"


def _global_evidence_id(paper_id: str, evidence_id: str) -> str:
    """Normalize an evidence id to the global ``paper_id::local`` form.

    Extraction results may carry either local ids (``E1``) or already-global ids
    (``paper_a::E1``); normalize to the global form without double-prefixing.
    """
    return evidence_id if "::" in evidence_id else f"{paper_id}::{evidence_id}"


def _build_relation_paper_block(
    paper_id: str,
    extraction: dict[str, Any],
    year: int | None,
) -> tuple[str, set[str], dict[str, str]]:
    """Build paper block for relation prompt.

    Returns:
        Tuple of (block_text, allowed_evidence_ids, evidence_owners)
    """
    # Quote lookup keyed by GLOBAL evidence id (handles local or global input).
    quote_by_id = {
        _global_evidence_id(paper_id, evidence["evidence_id"]): evidence["quote"]
        for evidence in extraction.get("evidence", [])
    }

    allowed = set(quote_by_id)
    owners = {gid: paper_id for gid in allowed}

    lines = [f"### paper_id={paper_id} year={year}"]

    # Key results with evidence tags
    lines.append("Key results:")
    for result in extraction.get("results", []):
        tags = ", ".join(
            _global_evidence_id(paper_id, rid)
            for rid in result.get("evidence_ids", [])
        )
        lines.append(f"- ({tags}) {result['text']}")

    # A-level claims
    a_claims = [
        claim for claim in extraction.get("claims", [])
        if claim.get("evidence_type") == "A"
    ]
    if a_claims:
        lines.append("A-level claims:")
        for claim in a_claims:
            tags = ", ".join(
                _global_evidence_id(paper_id, rid)
                for rid in claim.get("evidence_ids", [])
            )
            lines.append(f"- ({tags}) {claim['text']}")

    # Evidence quotes
    lines.append("Evidence quotes available:")
    for gid, quote in quote_by_id.items():
        lines.append(f'- {gid}: "{quote}"')

    return "\n".join(lines), allowed, owners


def build_relation_prompt(
    label: str,
    paper_detail: str,
    retrieval_context: str = "",
    pair_scope: Sequence[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """Build system and user prompts for relation extraction.

    Args:
        label: Research direction label
        paper_detail: Formatted paper blocks with evidence
        retrieval_context: Optional M5 retrieval context (not authoritative)
        pair_scope: When set, the model must judge ONLY these pairs. Used by the
            bounded-chunk escalation for directions whose whole-direction answer
            cannot fit the completion ceiling.

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    context = ""
    if retrieval_context:
        context = (
            "\n\nRETRIEVED DISCOVERY CONTEXT (not authoritative evidence; "
            "do not cite it in output):\n"
            f"{retrieval_context}"
        )

    scope = ""
    if pair_scope:
        listed = ", ".join(f"{source} | {target}" for source, target in pair_scope)
        scope = (
            "\nEvaluate ONLY these paper pairs, exactly as listed. Decide for each "
            "pair whether a relation genuinely holds, and report just the ones you "
            "find: returning fewer relations than pairs — including none — is a "
            f"normal and expected answer.\n{listed}\n"
        )

    user_prompt = (
        f"RESEARCH DIRECTION: {label}\n\n"
        f"PAPER DETAIL:\n{paper_detail}{context}\n"
        f"{scope}\n"
        "Identify relations between paper pairs. For each relation:\n"
        "- source_paper_id: the paper making the claim\n"
        "- target_paper_id: the paper being compared against\n"
        f"- relation_type: one of {RELATION_PROMPT_VOCABULARY}\n"
        "- statement: 1-2 sentence relation assertion\n"
        "- source_evidence_ids: cite EVERY evidence ID from the source paper "
        "that bears on this relation (normally 2-4; use 1 only when the paper "
        "genuinely offers just one relevant span). Do not cite one token ID just "
        "to satisfy the schema — under-citing is a validation error.\n"
        "- target_evidence_ids: same rule for the target paper. Required for "
        "challenges/contradicts, and required to be multi-span whenever the "
        "target position is asserted at all.\n"
        f"- difference_dimension: required for "
        f"{'|'.join(_DIFFERENCE_DIMENSION_RELATIONS)} "
        "(definition|object|method|conditions|conclusion)\n"
        "- comparability_note: why papers are/aren't comparable\n"
        "- uncertainty: any caveats about this relation\n\n"
        'Return exactly (note the MULTI-ID evidence arrays — they are the norm, '
        'not the exception):\n'
        '{"relations": [{"source_paper_id": "...", "target_paper_id": "...", '
        '"relation_type": "...", "statement": "...", '
        '"source_evidence_ids": ["paper_id::E1", "paper_id::E4"], '
        '"target_evidence_ids": ["other_id::E2", "other_id::E7"], '
        '"difference_dimension": null, '
        '"comparability_note": "", '
        '"uncertainty": ""}]}'
    )

    return _RELATION_SYSTEM, user_prompt


def _validate_relations(
    relations: list[dict[str, Any]],
    snapshot_id: str,
    roster: set[str],
    allowed_evidence: set[str],
    evidence_owners: dict[str, str],
    direction_id: str,
) -> tuple[
    list[PaperRelation],
    list[dict],
    list[dict],
    list[dict],
]:
    """Validate extracted relations against all nine gates.

    Args:
        relations: Raw relation dictionaries from LLM
        snapshot_id: Snapshot identifier
        roster: Set of valid paper IDs in this direction
        allowed_evidence: Set of valid evidence IDs
        evidence_owners: Mapping of evidence_id -> paper_id
        direction_id: Direction identifier for error reporting

    Returns:
        Tuple of ``(valid_relations, ignored_relations, validation_errors,
        density_warnings)``.  The fourth item is advisory, not a rejection:
        it lists assertion-bearing relations that cited fewer than
        ``DENSITY_MIN_PER_SIDE`` spans on a side.  Those relations stay in
        ``valid`` — a single span can be a legitimate, complete citation —
        but the count lets a caller see that a direction under-cited and
        choose to re-run it under the stricter prompt.
    """
    valid: list[PaperRelation] = []
    ignored: list[dict] = []
    density_warnings: list[dict] = []
    errors: list[dict] = []

    # Track seen (source, target, relation_type) for deduplication (gate 8)
    seen_key: dict[tuple[str, str, str], str] = {}  # key -> relation_id
    seen_relation_ids: set[str] = set()

    for idx, raw in enumerate(relations):
        if not isinstance(raw, dict):
            _record = {"error": "invalid_relation_shape", "detail": f"item {idx} is not an object"}
            ignored.append(_record)
            continue
        raw_errors: list[str] = []
        ignore_reason: str | None = None

        source = raw.get("source_paper_id", "")
        target = raw.get("target_paper_id", "")
        rel_type = raw.get("relation_type", "")
        source_ev = raw.get("source_evidence_ids", [])
        target_ev = raw.get("target_evidence_ids", [])
        diff_dim = raw.get("difference_dimension")

        def _err(code: str, detail: str) -> None:
            raw_errors.append(f"{code}: {detail}")

        # Gate 1: Self-reference forbidden
        if source == target:
            _err("self_reference", f"source == target ({source})")

        # Gate 2: Both papers must be in roster
        if source and source not in roster:
            _err("paper_not_in_roster", f"source paper {source}")
        if target and target not in roster:
            _err("paper_not_in_roster", f"target paper {target}")

        # Gate 3: source_evidence_ids >= 1, each belongs to source
        if len(source_ev) < 1:
            _err("missing_source_evidence", "source_evidence_ids must have at least 1")
        else:
            for ev in source_ev:
                expected_owner = source
                actual_owner = evidence_owners.get(ev)
                if actual_owner != expected_owner:
                    _err(
                        "evidence_owner_mismatch",
                        f"source evidence {ev} owned by {actual_owner}, "
                        f"expected {expected_owner}",
                    )

        # Gate 4: challenges/contradicts require target_evidence_ids >= 1
        if rel_type in RELATIONS_REQUIRING_TARGET_EVIDENCE:
            if len(target_ev) < 1:
                _err(
                    "missing_target_evidence",
                    f"{rel_type} requires target_evidence_ids >= 1",
                )
            else:
                for ev in target_ev:
                    expected_owner = target
                    actual_owner = evidence_owners.get(ev)
                    if actual_owner != expected_owner:
                        _err(
                            "evidence_owner_mismatch",
                            f"target evidence {ev} owned by {actual_owner}, "
                            f"expected {expected_owner}",
                        )

        # Gate 3b: evidence-density floor for assertion-bearing relations.
        # Recorded, NOT dropped.  A one-span relation is often legitimate (the
        # paper may state its position in a single sentence), so silently
        # discarding it would destroy real findings.  The caller receives the
        # count and can decide whether to re-run those directions with a
        # stricter prompt.  See relation_types.DENSITY_FLOORED_RELATIONS.
        if rel_type in DENSITY_FLOORED_RELATIONS:
            if len(source_ev) < DENSITY_MIN_PER_SIDE:
                density_warnings.append(
                    {
                        "direction_id": direction_id,
                        "relation_index": idx,
                        "source_paper_id": source,
                        "target_paper_id": target,
                        "relation_type": rel_type,
                        "side": "source",
                        "cited": len(source_ev),
                        "required": DENSITY_MIN_PER_SIDE,
                    }
                )
            if target_ev and len(target_ev) < DENSITY_MIN_PER_SIDE:
                density_warnings.append(
                    {
                        "direction_id": direction_id,
                        "relation_index": idx,
                        "source_paper_id": source,
                        "target_paper_id": target,
                        "relation_type": rel_type,
                        "side": "target",
                        "cited": len(target_ev),
                        "required": DENSITY_MIN_PER_SIDE,
                    }
                )

        # Gate 5: specific relation types require difference_dimension
        if rel_type in RELATIONS_REQUIRING_DIFFERENCE_DIMENSION:
            if not diff_dim:
                _err(
                    "missing_difference_dimension",
                    f"{rel_type} requires difference_dimension",
                )
            elif diff_dim not in VALID_DIFFERENCE_DIMENSIONS:
                _err("invalid_difference_dimension", str(diff_dim))

        # Gate 6: relation_type must be in vocabulary
        if rel_type not in VALID_RELATION_TYPES:
            _err("invalid_relation_type", str(rel_type))

        # Gate 7: evidence IDs must exist and ownership correct
        all_evidence = source_ev + target_ev
        for ev in all_evidence:
            if ev not in allowed_evidence:
                _err("unknown_evidence_id", str(ev))

        # Gate 9: Check for new evidence IDs not in allowed set
        # (This is covered by Gate 7's "unknown evidence ID" check)

        # Gate 8: Deduplication — only entries that pass validation are registered;
        # a rejected first item must NOT occupy the key and swallow a later valid one.
        dedup_key = (source, target, rel_type)
        all_ev = sorted(set(source_ev + target_ev))
        rel_id = deterministic_relation_id(
            snapshot_id, source, target, rel_type, all_ev
        )
        if dedup_key in seen_key:
            _err("duplicate_relation", f"keeping first: {seen_key[dedup_key]}")
        elif rel_id in seen_relation_ids:
            _err("duplicate_relation_id", rel_id)

        # If there are errors, decide whether to ignore or record error
        if raw_errors:
            # Check if it's an ignorable case (gate 9 style)
            # Gate 9: model outputs new evidence IDs not in verified evidence
            has_unknown_evidence = any(
                ev not in allowed_evidence
                for ev in (source_ev + target_ev)
            )
            if has_unknown_evidence:
                ignore_reason = "model_output_unknown_evidence_ids"
                ignored.append({
                    "direction_id": direction_id,
                    "raw_relation": raw,
                    "reason": ignore_reason,
                    "errors": raw_errors,
                })
            else:
                errors.append({
                    "direction_id": direction_id,
                    "relation_index": idx,
                    "raw_relation": raw,
                    "errors": raw_errors,
                })
            continue

        # All gates passed - register keys and construct PaperRelation
        seen_relation_ids.add(rel_id)
        seen_key[dedup_key] = rel_id
        relation = PaperRelation(
            relation_id=rel_id,
            source_paper_id=source,
            target_paper_id=target,
            relation_type=rel_type,
            statement=raw.get("statement", ""),
            source_evidence_ids=source_ev,
            target_evidence_ids=target_ev,
            difference_dimension=diff_dim,
            comparability_note=raw.get("comparability_note", ""),
            uncertainty=raw.get("uncertainty", ""),
            snapshot_id=snapshot_id,
        )
        valid.append(relation)

    return valid, ignored, errors, density_warnings


def _pair_chunk_extract(
    client: RelationClient,
    direction: ResearchDirection,
    present_papers: Sequence[str],
    blocks_by_paper: dict[str, str],
    retrieval_context: str,
) -> tuple[list[Any], list[dict], int, int]:
    """Evaluate bounded groups of paper pairs for one oversized direction.

    Returns ``(raw_relations, errors, calls, failed_calls)``. Every pair of the
    direction appears in exactly one chunk, so the escalation trades extra calls
    for complete coverage instead of silently dropping comparisons.
    """
    raw: list[Any] = []
    errors: list[dict] = []
    calls = 0
    failed_calls = 0

    for chunk in iter_pair_chunks(tuple(present_papers)):
        involved: list[str] = []
        for source, target in chunk:
            for paper_id in (source, target):
                if paper_id not in involved:
                    involved.append(paper_id)
        detail = "\n\n".join(
            blocks_by_paper[paper_id]
            for paper_id in involved
            if paper_id in blocks_by_paper
        )
        system_prompt, user_prompt = build_relation_prompt(
            direction.label, detail, retrieval_context, pair_scope=chunk
        )
        calls += 1
        try:
            payload = client.complete_json(system_prompt, user_prompt)
        except Exception as e:  # noqa: BLE001 - provider errors vary by backend
            failed_calls += 1
            errors.append({
                "direction_id": direction.direction_id,
                "code": PAIR_CHUNK_FAILED,
                "pairs": len(chunk),
                "reason": str(e),
            })
            continue

        # Chunking exists to bound scope, so a relation for a pair outside this
        # chunk must not be admitted (the direction roster check alone would let
        # it through, since every paper is in the roster).
        chunk_pairs = {frozenset(pair) for pair in chunk}
        for relation in payload.get("relations", []) or []:
            if not isinstance(relation, dict):
                raw.append(relation)  # let the validator report the shape problem
                continue
            pair = frozenset((
                relation.get("source_paper_id"), relation.get("target_paper_id"),
            ))
            if pair in chunk_pairs:
                raw.append(relation)
            else:
                errors.append({
                    "direction_id": direction.direction_id,
                    "code": PAIR_CHUNK_OUT_OF_SCOPE,
                    "reason": "relation returned for a pair outside this chunk",
                    "source_paper_id": relation.get("source_paper_id"),
                    "target_paper_id": relation.get("target_paper_id"),
                })

    return raw, errors, calls, failed_calls


def _result_to_payload(result: "RelationExtractionResult") -> dict[str, Any]:
    return {
        "direction_id": result.direction_id,
        "relations": [relation.model_dump(mode="json") for relation in result.relations],
        "ignored_relations": list(result.ignored_relations),
        "validation_errors": list(result.validation_errors),
        "density_warnings": list(result.density_warnings),
        "extraction_trace": list(result.extraction_trace),
    }


def _result_from_payload(payload: dict[str, Any]) -> "RelationExtractionResult":
    return RelationExtractionResult(
        direction_id=str(payload.get("direction_id", "")),
        relations=tuple(
            PaperRelation.model_validate(relation)
            for relation in payload.get("relations", [])
        ),
        ignored_relations=tuple(payload.get("ignored_relations", [])),
        validation_errors=tuple(payload.get("validation_errors", [])),
        density_warnings=tuple(payload.get("density_warnings", [])),
        extraction_trace=tuple(payload.get("extraction_trace", [])),
    )


def _relation_cache_fingerprint(
    snapshot_id: str,
    direction: ResearchDirection,
    extractions: dict[str, dict[str, Any]],
    years: dict[str, int],
    retrieval_context: str,
) -> str:
    """Hash every input that could change a direction's relation output.

    Paper blocks are built from the extraction dicts and years, and the prompt
    additionally carries the retrieval context. Any drift in those invalidates
    the cached result, so a stale verdict is never served.
    """
    papers = []
    for paper_id in sorted(direction.paper_ids):
        extraction = extractions.get(paper_id)
        if extraction is None:
            continue
        papers.append({
            "paper_id": paper_id,
            "year": years.get(paper_id),
            "extraction": extraction,
        })
    payload = {
        "snapshot_id": snapshot_id,
        "direction_id": direction.direction_id,
        "label": direction.label,
        "papers": papers,
        "retrieval_context": retrieval_context,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class _RelationResultCache:
    """Durable per-direction cache so an interrupted run can resume.

    A MAG-209 run spends hours here; without this, a crash loses every finished
    direction. Each entry is written as soon as a direction completes and is
    keyed by a fingerprint of that direction's inputs, so a rerun skips exactly
    the work that already produced an answer and redoes anything whose inputs
    changed.
    """
    def __init__(self, path: Path | str | None) -> None:
        self._path = Path(path) if path is not None else None
        self._data: dict[str, Any] = {}
        if self._path is not None and self._path.exists():
            try:
                loaded = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                self._data = loaded

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def get(self, direction_id: str, fingerprint: str) -> dict[str, Any] | None:
        entry = self._data.get(direction_id)
        if not isinstance(entry, dict):
            return None
        if entry.get("fingerprint") != fingerprint:
            return None
        result = entry.get("result")
        return result if isinstance(result, dict) else None

    def put(self, direction_id: str, fingerprint: str, result: dict[str, Any]) -> None:
        if self._path is None:
            return
        self._data[direction_id] = {"fingerprint": fingerprint, "result": result}
        self._flush()

    def _flush(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._path)


def extract_paper_relations(
    client: RelationClient,
    directions: list[ResearchDirection],
    extractions: dict[str, dict[str, Any]],
    years: dict[str, int],
    retrieval_context_by_direction: dict[str, str] | None = None,
    snapshot_id: str = "default_snapshot",
    cache_path: Path | str | None = None,
) -> list[RelationExtractionResult]:
    """Extract paper relations for each research direction.

    Args:
        client: LLM client for JSON completion
        directions: List of research directions
        extractions: Per-paper extraction results (paper_id -> extraction dict)
        years: Publication years by paper_id
        retrieval_context_by_direction: Optional M5 context per direction
        snapshot_id: Snapshot identifier for relation IDs
        cache_path: Optional durable per-direction cache; a rerun skips
            directions whose inputs are unchanged and only extracts the rest

    Returns:
        List of RelationExtractionResult, one per direction
    """
    results: list[RelationExtractionResult] = []
    retrieval_map = retrieval_context_by_direction or {}
    cache = _RelationResultCache(cache_path)

    for direction in directions:
        retrieval_context = retrieval_map.get(direction.direction_id, "")
        fingerprint = _relation_cache_fingerprint(
            snapshot_id, direction, extractions, years, retrieval_context
        )
        cached = cache.get(direction.direction_id, fingerprint)
        if cached is not None:
            results.append(_result_from_payload(cached))
            continue

        # Build paper blocks and evidence metadata for this direction
        blocks: list[str] = []
        blocks_by_paper: dict[str, str] = {}
        allowed_evidence: set[str] = set()
        evidence_owners: dict[str, str] = {}
        present_papers: list[str] = []

        for paper_id in direction.paper_ids:
            extraction = extractions.get(paper_id)
            if extraction is None:
                continue

            present_papers.append(paper_id)
            block, paper_allowed, paper_owners = _build_relation_paper_block(
                paper_id, extraction, years.get(paper_id)
            )
            blocks.append(block)
            blocks_by_paper[paper_id] = block
            allowed_evidence |= paper_allowed
            evidence_owners.update(paper_owners)

        # Build prompt and call LLM
        system_prompt, user_prompt = build_relation_prompt(
            direction.label,
            "\n\n".join(blocks),
            retrieval_map.get(direction.direction_id, ""),
        )

        payload: dict[str, Any] | None = None
        attempts = 0
        last_error: BaseException | None = None
        while attempts < _RELATION_MAX_ATTEMPTS:
            attempts += 1
            try:
                payload = client.complete_json(system_prompt, user_prompt)
                break
            except Exception as e:  # noqa: BLE001 - provider errors vary by backend
                last_error = e
                if not _is_retryable(e):
                    break

        trace: tuple[dict, ...] = ()
        escalated_errors: list[dict] = []
        if payload is not None:
            raw_relations = payload.get("relations", [])
        else:
            # Retrying cannot rescue a request whose answer does not fit the
            # completion ceiling, so an oversized direction is re-run as bounded
            # pair chunks that still cover every pair exactly once.
            retrieval_context = retrieval_map.get(direction.direction_id, "")
            raw_relations, escalated_errors, chunk_calls, failed_calls = _pair_chunk_extract(
                client, direction, present_papers, blocks_by_paper, retrieval_context
            )
            if chunk_calls:
                trace = ({
                    "direction_id": direction.direction_id,
                    "fallback": "pair_chunked",
                    "attempts": attempts,
                    "calls": chunk_calls,
                    "failed_calls": failed_calls,
                    "pairs": len(list(itertools.combinations(present_papers, 2))),
                    "chunk_size": _RELATION_CHUNK_SIZE,
                },)

            if not chunk_calls or failed_calls == chunk_calls:
                results.append(RelationExtractionResult(
                    direction_id=direction.direction_id,
                    relations=(),
                    ignored_relations=(),
                    validation_errors=tuple(escalated_errors) + ({
                        "direction_id": direction.direction_id,
                        "code": RELATION_EXTRACTION_FAILED,
                        "attempts": attempts,
                        "reason": str(last_error),
                        "retryable": _is_retryable(last_error) if last_error else False,
                        "error": f"LLM call failed: {last_error}",
                    },),
                    extraction_trace=trace,
                ))
                continue

        # Validate all relations
        roster = set(direction.paper_ids)
        (
            valid_relations,
            ignored,
            validation_errors,
            density_warnings,
        ) = _validate_relations(
            raw_relations,
            snapshot_id,
            roster,
            allowed_evidence,
            evidence_owners,
            direction.direction_id,
        )

        result = RelationExtractionResult(
            direction_id=direction.direction_id,
            relations=tuple(valid_relations),
            ignored_relations=tuple(ignored),
            validation_errors=tuple(escalated_errors) + tuple(validation_errors),
            density_warnings=tuple(density_warnings),
            extraction_trace=trace,
        )
        cache.put(direction.direction_id, fingerprint, _result_to_payload(result))
        results.append(result)

    return results
