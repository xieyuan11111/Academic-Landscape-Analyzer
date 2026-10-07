"""Tests for FR-12 on-demand debate candidate cards (contract §8, TDD RED first).

Covers the 9 required cases: legitimate two-position card + deterministic card_id,
single-position rejection, all-same-paper positions rejection, fit_for_theater
vocabulary rejection, blank question rejection, to_dict round-trip, optional-list
empty defaults, comparability/support recording and snapshot-bound card_id change.

The ``SourceRef`` dependency lives in a sibling module that is being authored in
parallel; if it is not yet present when the suite is collected, the entire module
is skipped gracefully via :func:`pytest.importorskip` so the wait is observable
instead of failing with ``ImportError``.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

# Parallel dependency: SourceRef is defined in academic_landscape.source_cards.
# If the sibling module is absent or not yet importable (e.g. mid-edit by the
# parallel agent), skip the whole module cleanly. ``pytest.importorskip`` only
# catches ImportError, so we additionally catch any other exception raised while
# importing the module — the dependency is simply not ready yet.
_spec = importlib.util.find_spec("academic_landscape.source_cards")
if _spec is None:
    pytest.skip(
        "Waiting for parallel source_cards.py (SourceRef) to land.",
        allow_module_level=True,
    )

try:
    _source_cards = importlib.import_module("academic_landscape.source_cards")
except Exception as _exc:  # noqa: BLE001 — module-not-ready is the only signal we care about
    pytest.skip(
        f"Waiting for parallel source_cards.py to stabilize "
        f"({type(_exc).__name__}: {_exc})",
        allow_module_level=True,
    )

SourceRef = _source_cards.SourceRef

from academic_landscape.debate_cards import (  # noqa: E402
    DebateCard,
    DebatePosition,
    build_debate_card,
    debate_card_to_dict,
    deterministic_debate_card_id,
)

SNAPSHOT_ID = "snapshot_debate"


def _src(paper_id: str, evidence_id: str, quote: str = "x") -> SourceRef:
    """Build a minimal ``SourceRef`` for tests (fields mirror the parallel module)."""

    return SourceRef(
        paper_id=paper_id,
        evidence_id=evidence_id,
        quote=quote,
        locator="page 1",
    )


def _position(
    paper_id: str,
    stance: str,
    evidence_ids: list[str] | None = None,
) -> DebatePosition:
    ids = evidence_ids if evidence_ids is not None else [f"{paper_id}::E1"]
    refs = tuple(_src(paper_id, eid) for eid in ids)
    return DebatePosition(paper_id=paper_id, stance=stance, evidence=refs)


def test_valid_two_position_card_builds_with_deterministic_card_id() -> None:
    positions = (
        _position("paper_a", "stance A", ["paper_a::E1"]),
        _position("paper_b", "stance B", ["paper_b::E1"]),
    )
    card = build_debate_card(
        snapshot_id=SNAPSHOT_ID,
        question="Does X cause Y?",
        direction_ids=["dir_1"],
        positions=positions,
        shared_premises=("Both agree P",),
        divergent_premises=("Diverge on Q",),
        comparability_issues=("Different populations",),
        open_questions=("Under what scope?",),
        fit_for_theater="yes",
    )
    assert isinstance(card, DebateCard)
    assert card.question == "Does X cause Y?"
    assert card.direction_ids == ("dir_1",)
    assert len(card.positions) == 2
    assert card.shared_premises == ("Both agree P",)
    assert card.divergent_premises == ("Diverge on Q",)
    assert card.comparability_issues == ("Different populations",)
    assert card.open_questions == ("Under what scope?",)
    assert card.fit_for_theater == "yes"
    assert card.snapshot_id == SNAPSHOT_ID
    assert card.card_id.startswith("debate_")
    assert len(card.card_id) == len("debate_") + 16

    # Determinism: same inputs -> same card_id; cross-checked against the helper.
    expected_id = deterministic_debate_card_id(
        SNAPSHOT_ID, "Does X cause Y?", ["paper_a", "paper_b"]
    )
    assert card.card_id == expected_id

    again = build_debate_card(
        snapshot_id=SNAPSHOT_ID,
        question="Does X cause Y?",
        direction_ids=["dir_1"],
        positions=positions,
        fit_for_theater="yes",
    )
    assert again.card_id == card.card_id


def test_single_position_card_is_rejected() -> None:
    with pytest.raises(ValueError, match=">=2 positions"):
        build_debate_card(
            snapshot_id=SNAPSHOT_ID,
            question="Lonely question",
            direction_ids=["dir_1"],
            positions=(_position("paper_a", "only one"),),
        )


def test_positions_from_same_paper_id_are_rejected() -> None:
    same_paper_positions = (
        _position("paper_a", "stance A1"),
        _position("paper_a", "stance A2"),
    )
    with pytest.raises(ValueError, match="distinct paper_id"):
        build_debate_card(
            snapshot_id=SNAPSHOT_ID,
            question="A within-paper tension",
            direction_ids=["dir_1"],
            positions=same_paper_positions,
        )


def test_fit_for_theater_outside_vocabulary_is_rejected() -> None:
    positions = (
        _position("paper_a", "stance A"),
        _position("paper_b", "stance B"),
    )
    with pytest.raises(ValueError, match="fit_for_theater"):
        build_debate_card(
            snapshot_id=SNAPSHOT_ID,
            question="Vocab test",
            direction_ids=["dir_1"],
            positions=positions,
            fit_for_theater="maybe",
        )


def test_blank_question_is_rejected() -> None:
    positions = (
        _position("paper_a", "stance A"),
        _position("paper_b", "stance B"),
    )
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(ValueError, match="question"):
            build_debate_card(
                snapshot_id=SNAPSHOT_ID,
                question=bad,
                direction_ids=["dir_1"],
                positions=positions,
            )


def test_to_dict_round_trip_carries_every_field() -> None:
    positions = (
        _position("paper_a", "stance A", ["paper_a::E1"]),
        _position("paper_b", "stance B", ["paper_b::E1"]),
    )
    support_counter = (_src("paper_a", "paper_a::E2", "supporting quote"),)
    card = build_debate_card(
        snapshot_id=SNAPSHOT_ID,
        question="Round-trip?",
        direction_ids=["dir_1", "dir_2"],
        positions=positions,
        shared_premises=("P1", "P2"),
        divergent_premises=("D1",),
        support_and_counter=support_counter,
        comparability_issues=("C1",),
        open_questions=("O1", "O2", "O3"),
        fit_for_theater="unclear",
    )
    payload = debate_card_to_dict(card)
    assert isinstance(payload, dict)
    assert payload["card_id"] == card.card_id
    assert payload["question"] == "Round-trip?"
    assert payload["direction_ids"] == ["dir_1", "dir_2"]
    assert payload["fit_for_theater"] == "unclear"
    assert payload["snapshot_id"] == SNAPSHOT_ID
    assert payload["shared_premises"] == ["P1", "P2"]
    assert payload["divergent_premises"] == ["D1"]
    assert payload["comparability_issues"] == ["C1"]
    assert payload["open_questions"] == ["O1", "O2", "O3"]

    assert isinstance(payload["positions"], list)
    assert len(payload["positions"]) == 2
    first_pos = payload["positions"][0]
    assert first_pos["paper_id"] in {"paper_a", "paper_b"}
    assert "stance" in first_pos
    assert isinstance(first_pos["evidence"], list)
    assert len(first_pos["evidence"]) == 1
    first_evidence = first_pos["evidence"][0]
    assert first_evidence["paper_id"] == first_pos["paper_id"]
    assert first_evidence["evidence_id"] == f"{first_pos['paper_id']}::E1"

    assert isinstance(payload["support_and_counter"], list)
    assert len(payload["support_and_counter"]) == 1
    assert payload["support_and_counter"][0]["paper_id"] == "paper_a"
    assert payload["support_and_counter"][0]["evidence_id"] == "paper_a::E2"
    assert payload["support_and_counter"][0]["quote"] == "supporting quote"


def test_optional_lists_default_to_empty_and_keep_type() -> None:
    positions = (
        _position("paper_a", "stance A"),
        _position("paper_b", "stance B"),
    )
    card = build_debate_card(
        snapshot_id=SNAPSHOT_ID,
        question="Minimal card",
        direction_ids=["dir_1"],
        positions=positions,
    )
    assert card.shared_premises == ()
    assert card.divergent_premises == ()
    assert card.open_questions == ()
    assert card.support_and_counter == ()
    assert card.comparability_issues == ()
    # Tuples, not lists — dataclass contract.
    assert isinstance(card.shared_premises, tuple)
    assert isinstance(card.divergent_premises, tuple)
    assert isinstance(card.open_questions, tuple)
    assert isinstance(card.comparability_issues, tuple)
    assert isinstance(card.support_and_counter, tuple)

    payload = debate_card_to_dict(card)
    assert payload["shared_premises"] == []
    assert payload["divergent_premises"] == []
    assert payload["open_questions"] == []
    assert payload["support_and_counter"] == []
    assert payload["comparability_issues"] == []


def test_comparability_issues_and_support_counter_are_recorded() -> None:
    positions = (
        _position("paper_a", "stance A"),
        _position("paper_b", "stance B"),
    )
    card = build_debate_card(
        snapshot_id=SNAPSHOT_ID,
        question="Recording test",
        direction_ids=["dir_1"],
        positions=positions,
        comparability_issues=("population A vs B", "metric differs"),
        support_and_counter=(
            _src("paper_a", "paper_a::E2", "for A"),
            _src("paper_b", "paper_b::E2", "against A"),
        ),
    )
    assert card.comparability_issues == ("population A vs B", "metric differs")
    assert len(card.support_and_counter) == 2
    assert {ref.paper_id for ref in card.support_and_counter} == {
        "paper_a",
        "paper_b",
    }
    payload = debate_card_to_dict(card)
    assert payload["comparability_issues"] == ["population A vs B", "metric differs"]
    assert len(payload["support_and_counter"]) == 2
    assert payload["support_and_counter"][0]["quote"] in {"for A", "against A"}


def test_card_id_changes_when_snapshot_id_changes() -> None:
    positions = (
        _position("paper_a", "stance A"),
        _position("paper_b", "stance B"),
    )
    base_kwargs = {
        "question": "Snapshot drift",
        "direction_ids": ["dir_1"],
        "positions": positions,
    }
    card_snap_a = build_debate_card(snapshot_id="snap_a", **base_kwargs)
    card_snap_b = build_debate_card(snapshot_id="snap_b", **base_kwargs)
    assert card_snap_a.card_id != card_snap_b.card_id
    assert card_snap_a.card_id.startswith("debate_")
    assert card_snap_b.card_id.startswith("debate_")
    # The helper agrees with the builder, in both directions.
    assert card_snap_a.card_id == deterministic_debate_card_id(
        "snap_a", "Snapshot drift", ["paper_a", "paper_b"]
    )
    assert card_snap_b.card_id == deterministic_debate_card_id(
        "snap_b", "Snapshot drift", ["paper_a", "paper_b"]
    )