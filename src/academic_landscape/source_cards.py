"""FR-10: Traceable source cards linking report judgments to corpus evidence.

A :class:`SourceCard` binds a report-level judgment to the evidence that
supports, qualifies (revises) or opposes it. Every reported judgment of any
importance must travel with one of these cards so the lineage can be inspected.

Semantic rules for the four ``judgment_type`` values (A/B/C/D) are defined by
``claims.evidence_type`` and live in :data:`academic_landscape.semantic._ALLOWED_EVIDENCE_TYPES`.
This module reuses that vocabulary verbatim and does **not** introduce its
own enumeration. Report phrasing rules in PRD §10.1 also apply when the card
is surfaced to the user.

Validation is fail-closed: ``build_source_card`` raises ``ValueError`` when
the judgment type is outside the vocabulary, when the judgment text is blank,
when there is no supporting reference, or when any reference lacks a
globalized ``evidence_id`` (``paper_id::E1`` form).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

# Reuse, do not redefine. Semantic definitions live in semantic.py and PRD §10.
_ALLOWED_JUDGMENT_TYPES = frozenset({"A", "B", "C", "D"})

# Separator that joins snapshot_id and judgment into the card_id hash payload.
# Pipes are not legal in our IDs so collisions are free of ambiguity.
_CARD_ID_SEPARATOR = "|"
_CARD_ID_PREFIX = "card_"
_CARD_ID_DIGEST_LEN = 16
_GLOBAL_EVIDENCE_SEP = "::"


@dataclass(frozen=True)
class SourceRef:
    """One piece of evidence behind a judgment.

    ``evidence_id`` carries the globalized form ``paper_id::E1`` enforced by
    ``academic_landscape.evidence_scope``; ``paper_id`` here is a duplicate
    surface for readability so an evidence appendix can render each row
    without re-parsing the evidence id.
    """

    paper_id: str
    title: str = ""
    authors: str = ""
    year: int | None = None
    evidence_id: str = ""
    quote: str = ""
    locator: str = ""
    # Structured page so a reader — or a calling agent — can jump straight back
    # to the PDF page. Without it the card carried no page information at all
    # (not even the free-text locator), so §13.2 #5 could only be checked by
    # re-reading the extraction cache rather than the artifact we ship.
    page_number: int | None = None
    # When the quote was checked against the page text (authority gate). Carries
    # the timestamp so the evidence appendix can show *when* verification
    # happened, per the FR-10 "时间属性" requirement.
    verified_at: str = ""


@dataclass(frozen=True)
class SourceCard:
    """A judgment plus the source material that supports, qualifies, or opposes it."""

    card_id: str
    judgment: str
    judgment_type: str
    supporting: tuple[SourceRef, ...]
    revising: tuple[SourceRef, ...]
    opposing: tuple[SourceRef, ...]
    time_attribute: str
    # Defaulted fields below: kw_only keeps the spec'd reading order while
    # letting them coexist with the required `snapshot_id` field.
    applicability: str = field(default="", kw_only=True)
    uncertainty: str = field(default="", kw_only=True)
    snapshot_id: str
    # Evidence tier (一手/二手/综合/未决) per PRD §10.1/§10.2, plus the reason
    # it landed there so the report can state how far prose may lean on it.
    # Derived from the relation vocabulary + evidence-family independence.
    tier: str = field(default="", kw_only=True)
    tier_reason: str = field(default="", kw_only=True)


def deterministic_card_id(snapshot_id: str, judgment: str) -> str:
    """Return a stable ``card_<16hex>`` id derived from (snapshot_id, judgment).

    The digest mixes the snapshot and the judgment text so changing either
    input deterministically changes the id. Same inputs → same id.
    """
    payload = f"{snapshot_id}{_CARD_ID_SEPARATOR}{judgment}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:_CARD_ID_DIGEST_LEN]
    return f"{_CARD_ID_PREFIX}{digest}"


def card_id_from_stable_key(snapshot_id: str, stable_key: str) -> str:
    """Derive a card id from a stable, free-text-free key (a relation_id).

    Used when a card is built 1:1 from a relation. The relation already has a
    deterministic ``relation_id`` (snapshot + source + target + type + sorted
    evidence — no LLM wording), so keying the card off it makes the id
    invariant under the statement being rephrased on a re-run. The judgment
    text is still carried on the card for humans; it just no longer feeds the
    hash.
    """
    payload = f"{snapshot_id}{_CARD_ID_SEPARATOR}{stable_key}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:_CARD_ID_DIGEST_LEN]
    return f"{_CARD_ID_PREFIX}{digest}"


def _check_refs(refs: tuple[SourceRef, ...], bucket: str) -> None:
    for ref in refs:
        evidence_id = ref.evidence_id
        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or _GLOBAL_EVIDENCE_SEP not in evidence_id
        ):
            raise ValueError(
                f"{bucket} SourceRef.evidence_id must contain '{_GLOBAL_EVIDENCE_SEP}' "
                "in its globalized form (paper_id::E1)"
            )


def build_source_card(
    *,
    snapshot_id: str,
    judgment: str,
    judgment_type: str,
    supporting: tuple[SourceRef, ...],
    revising: tuple[SourceRef, ...] = (),
    opposing: tuple[SourceRef, ...] = (),
    time_attribute: str = "",
    applicability: str = "",
    uncertainty: str = "",
    stable_key: str | None = None,
    tier: str = "",
    tier_reason: str = "",
) -> SourceCard:
    """Construct a :class:`SourceCard` with fail-closed validation.

    Raises ``ValueError`` for any of:
      * ``judgment_type`` outside the A/B/C/D vocabulary,
      * ``supporting`` empty,
      * ``judgment`` blank,
      * any reference missing a globalized ``evidence_id`` (``paper_id::E1``),
      * ``stable_key`` present but blank (it would collide every card).

    When ``stable_key`` (a relation_id) is supplied the card id is derived from
    it instead of the judgment text, so a rephrased statement cannot change the
    id. Without it the card keeps the legacy judgment-derived id.
    """
    if judgment_type not in _ALLOWED_JUDGMENT_TYPES:
        raise ValueError(
            f"judgment_type must be one of {sorted(_ALLOWED_JUDGMENT_TYPES)}"
        )
    if not isinstance(judgment, str) or not judgment.strip():
        raise ValueError("judgment must be a non-blank string")
    if not isinstance(supporting, tuple) or not supporting:
        raise ValueError("supporting must be a non-empty tuple of SourceRef")
    if stable_key is not None and not isinstance(stable_key, str):
        raise ValueError("stable_key must be a string when provided")
    if stable_key is not None and not stable_key:
        raise ValueError("stable_key must be non-blank when provided")

    _check_refs(supporting, "supporting")
    _check_refs(revising, "revising")
    _check_refs(opposing, "opposing")

    if stable_key is not None:
        card_id = card_id_from_stable_key(snapshot_id, stable_key)
    else:
        card_id = deterministic_card_id(snapshot_id, judgment)
    return SourceCard(
        card_id=card_id,
        judgment=judgment,
        judgment_type=judgment_type,
        supporting=supporting,
        revising=revising,
        opposing=opposing,
        time_attribute=time_attribute,
        applicability=applicability,
        uncertainty=uncertainty,
        snapshot_id=snapshot_id,
        tier=tier,
        tier_reason=tier_reason,
    )


def source_card_to_dict(card: SourceCard) -> dict:
    """Serialize a :class:`SourceCard` as a plain dict for JSON / appendices.

    Tuple reference buckets become lists; nested dataclasses become dicts via
    :func:`dataclasses.asdict`. Field order mirrors the dataclass definition.
    """
    return {
        "card_id": card.card_id,
        "judgment": card.judgment,
        "judgment_type": card.judgment_type,
        "supporting": [_ref_to_dict(ref) for ref in card.supporting],
        "revising": [_ref_to_dict(ref) for ref in card.revising],
        "opposing": [_ref_to_dict(ref) for ref in card.opposing],
        "time_attribute": card.time_attribute,
        "applicability": card.applicability,
        "uncertainty": card.uncertainty,
        "snapshot_id": card.snapshot_id,
    }


def _ref_to_dict(ref: SourceRef) -> dict:
    return {
        "paper_id": ref.paper_id,
        "title": ref.title,
        "authors": ref.authors,
        "year": ref.year,
        "evidence_id": ref.evidence_id,
        "quote": ref.quote,
        "locator": ref.locator,
        "page_number": ref.page_number,
        "verified_at": ref.verified_at,
    }


def build_source_refs(
    *,
    paper_id: str,
    evidence_ids: Sequence[str],
    extraction: dict,
    document: Any | None,
    year: int | None,
    verified_at: str = "",
) -> list[SourceRef]:
    """Build SourceRefs carrying the FR-10 metadata surface.

    Each ref gets ``title``/``authors`` pulled from the document's PDF
    metadata (which was previously dropped), the publication ``year``, the
    structured ``page_number`` and a readable ``locator`` derived from the page
    when the extraction supplied none.  ``verified_at`` records when the
    quote was checked against page text at the authority gate.
    """
    evidence_by_id: dict[str, dict] = {}
    for evidence in extraction.get("evidence", []) or []:
        if not isinstance(evidence, dict):
            continue
        local = evidence.get("evidence_id", "")
        global_id = local if "::" in local else f"{paper_id}::{local}"
        evidence_by_id[global_id] = evidence

    metadata = getattr(document, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
    title = str(metadata.get("title", "") or "")
    authors = str(metadata.get("author", "") or "")

    refs: list[SourceRef] = []
    for evidence_id in evidence_ids:
        evidence = evidence_by_id.get(evidence_id, {})
        page_number = evidence.get("page_number")
        refs.append(
            SourceRef(
                paper_id=paper_id,
                title=title,
                authors=authors,
                year=year,
                evidence_id=evidence_id,
                quote=evidence.get("quote", ""),
                locator=evidence.get("locator")
                or (f"page {page_number}" if page_number else ""),
                page_number=page_number,
                verified_at=verified_at,
            )
        )
    return refs


__all__ = [
    "SourceRef",
    "SourceCard",
    "deterministic_card_id",
    "card_id_from_stable_key",
    "build_source_card",
    "build_source_refs",
    "source_card_to_dict",
]
