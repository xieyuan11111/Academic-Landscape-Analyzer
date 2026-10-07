"""FR-12 on-demand debate entry: candidate cards for genuine controversies.

A ``DebateCard`` records one controversy surface (a question) and the two-or-more
positions held by *different* papers that disagree on it, together with the
premises they share, the premises they diverge on, the supporting/counter-evidence,
comparability caveats and remaining open questions. Cards are constructed only
when there is a real cross-paper disagreement (see ``cross_paper.compare_directions``):
positions all held by the same paper do not constitute an academic controversy.

This module is intentionally pure data + validation: it does not talk to an LLM.
Card construction is fail-closed; any structural violation raises ``ValueError``
so callers can decide whether to retry, drop or surface the error.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from .source_cards import SourceRef

_VALID_FIT = frozenset({"yes", "no", "unclear"})


@dataclass(frozen=True)
class DebatePosition:
    """One paper's stance on the card's question, with evidence references."""

    paper_id: str
    stance: str
    evidence: tuple[SourceRef, ...]


@dataclass(frozen=True)
class DebateCard:
    """Candidate card for a single cross-paper controversy.

    ``card_id`` is a deterministic 16-hex digest prefixed with ``debate_``,
    computed from ``snapshot_id``, ``question`` and the sorted unique paper
    ids that appear across the positions.
    """

    card_id: str
    question: str
    direction_ids: tuple[str, ...]
    positions: tuple[DebatePosition, ...]
    shared_premises: tuple[str, ...]
    divergent_premises: tuple[str, ...]
    support_and_counter: tuple[SourceRef, ...]
    comparability_issues: tuple[str, ...]
    open_questions: tuple[str, ...]
    fit_for_theater: str
    snapshot_id: str


def deterministic_debate_card_id(
    snapshot_id: str, question: str, paper_ids: Sequence[str]
) -> str:
    """Compute the deterministic ``debate_`` prefixed card id.

    The id is derived from the snapshot, the question text and the sorted
    unique paper ids of the positions. Different orderings of the same
    papers produce the same id.
    """

    sorted_papers = ",".join(sorted(set(paper_ids)))
    payload = f"{snapshot_id}|{question}|{sorted_papers}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"debate_{digest}"


def _coerce_str_tuple(value: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _coerce_source_tuple(
    value: Sequence[SourceRef] | SourceRef,
) -> tuple[SourceRef, ...]:
    if isinstance(value, SourceRef):
        return (value,)
    return tuple(value)


def build_debate_card(
    *,
    snapshot_id: str,
    question: str,
    direction_ids: Sequence[str],
    positions: Sequence[DebatePosition],
    shared_premises: Sequence[str] | str = (),
    divergent_premises: Sequence[str] | str = (),
    support_and_counter: Sequence[SourceRef] | SourceRef = (),
    comparability_issues: Sequence[str] | str = (),
    open_questions: Sequence[str] | str = (),
    fit_for_theater: str = "unclear",
) -> DebateCard:
    """Construct a :class:`DebateCard` with fail-closed validation.

    Raises ``ValueError`` if:

    * ``positions`` has fewer than 2 entries, or all entries belong to the
      same ``paper_id`` (a single-paper tension is not an academic controversy);
    * ``fit_for_theater`` is not one of ``{"yes", "no", "unclear"}``;
    * ``question`` is empty or whitespace-only.
    """

    position_tuple = _coerce_positions(positions)
    distinct_paper_ids = {position.paper_id for position in position_tuple}

    if len(position_tuple) < 2 or len(distinct_paper_ids) < 2:
        raise ValueError(
            "DebateCard requires >=2 positions from at least 2 distinct paper_ids; "
            f"got {len(position_tuple)} position(s) across "
            f"{len(distinct_paper_ids)} distinct paper_id(s)."
        )

    if fit_for_theater not in _VALID_FIT:
        raise ValueError(
            f"fit_for_theater must be one of {sorted(_VALID_FIT)}; got "
            f"{fit_for_theater!r}."
        )

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-blank string.")

    direction_tuple = tuple(direction_ids)
    shared_premises_tuple = _coerce_str_tuple(shared_premises)
    divergent_premises_tuple = _coerce_str_tuple(divergent_premises)
    comparability_tuple = _coerce_str_tuple(comparability_issues)
    open_questions_tuple = _coerce_str_tuple(open_questions)
    support_counter_tuple = _coerce_source_tuple(support_and_counter)

    card_id = deterministic_debate_card_id(
        snapshot_id,
        question,
        sorted(distinct_paper_ids),
    )

    return DebateCard(
        card_id=card_id,
        question=question,
        direction_ids=direction_tuple,
        positions=position_tuple,
        shared_premises=shared_premises_tuple,
        divergent_premises=divergent_premises_tuple,
        support_and_counter=support_counter_tuple,
        comparability_issues=comparability_tuple,
        open_questions=open_questions_tuple,
        fit_for_theater=fit_for_theater,
        snapshot_id=snapshot_id,
    )


def _coerce_positions(
    positions: Sequence[DebatePosition] | DebatePosition,
) -> tuple[DebatePosition, ...]:
    if isinstance(positions, DebatePosition):
        return (positions,)
    return tuple(positions)


def _source_ref_to_dict(ref: SourceRef) -> dict[str, Any]:
    """Serialize a ``SourceRef`` to a plain ``dict``.

    Uses ``dataclasses.asdict`` which works for any dataclass instance.
    """

    return asdict(ref)


def _position_to_dict(position: DebatePosition) -> dict[str, Any]:
    return {
        "paper_id": position.paper_id,
        "stance": position.stance,
        "evidence": [_source_ref_to_dict(ref) for ref in position.evidence],
    }


def debate_card_to_dict(card: DebateCard) -> dict[str, Any]:
    """Serialize a :class:`DebateCard` to a JSON-friendly ``dict``."""

    return {
        "card_id": card.card_id,
        "question": card.question,
        "direction_ids": list(card.direction_ids),
        "positions": [_position_to_dict(p) for p in card.positions],
        "shared_premises": list(card.shared_premises),
        "divergent_premises": list(card.divergent_premises),
        "support_and_counter": [
            _source_ref_to_dict(ref) for ref in card.support_and_counter
        ],
        "comparability_issues": list(card.comparability_issues),
        "open_questions": list(card.open_questions),
        "fit_for_theater": card.fit_for_theater,
        "snapshot_id": card.snapshot_id,
    }