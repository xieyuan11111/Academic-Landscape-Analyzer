"""Evidence-density policy tests (added 2026-09-27).

Background: the first full MAG-209 run produced 7,652 relations citing 1.85
evidence spans on average against an authority pool of 11.3 spans per paper,
and 76.6% of source cards carried a single supporting span.  The cause was the
prompt's "at least one" wording plus a one-element JSON example.

These tests lock in three things:
  1. the vocabulary split (which relations are floored, which are exempt);
  2. that the prompt no longer teaches a single-ID answer;
  3. that gate 3b records a warning without discarding a valid relation.
"""

from __future__ import annotations

import pytest

from academic_landscape.relation_extraction import (
    _validate_relations,
    build_relation_prompt,
)
from academic_landscape.relation_types import (
    DENSITY_FLOORED_RELATIONS,
    DENSITY_MIN_PER_SIDE,
    RELATION_TYPES_IN_PROMPT_ORDER,
    VALID_RELATION_TYPES,
)


# ---------------------------------------------------------------- vocabulary


def test_exempt_relations_are_exactly_the_non_asserting_ones() -> None:
    assert VALID_RELATION_TYPES - DENSITY_FLOORED_RELATIONS == frozenset(
        {"parallel", "not_comparable", "unknown"}
    )


def test_floor_covers_every_assertion_bearing_relation() -> None:
    for rel_type in ("supports", "extends", "revises", "qualifies", "challenges",
                     "contradicts", "same_name_different_meaning",
                     "surface_conflict", "conditional_difference"):
        assert rel_type in DENSITY_FLOORED_RELATIONS


def test_density_floor_is_two() -> None:
    assert DENSITY_MIN_PER_SIDE == 2


# -------------------------------------------------------------------- prompt


def _prompt() -> tuple[str, str]:
    return build_relation_prompt(label="Some Direction", paper_detail="")


def test_prompt_does_not_say_at_least_one() -> None:
    _system, user = _prompt()
    assert "at least one evidence ID" not in user


def test_prompt_example_shows_multi_id_evidence() -> None:
    _system, user = _prompt()
    # The example arrays must carry more than one ID, otherwise the model
    # learns the old one-token habit regardless of the prose above it.
    assert '"source_evidence_ids": ["paper_id::E1", "paper_id::E4"]' in user
    assert '"target_evidence_ids": ["other_id::E2", "other_id::E7"]' in user
    assert '["paper_id::E1"],' not in user


def test_prompt_offers_full_vocabulary() -> None:
    _system, user = _prompt()
    for rel_type in RELATION_TYPES_IN_PROMPT_ORDER:
        assert rel_type in user, f"{rel_type} missing from the prompt vocabulary"


# --------------------------------------------------------------------- gate


def _validate(relations: list[dict]) -> tuple[list, list, list, list]:
    snapshot = "snap_test"
    roster = {"paper_a", "paper_b"}
    owners = {
        "paper_a::E1": "paper_a",
        "paper_a::E2": "paper_a",
        "paper_b::E1": "paper_b",
        "paper_b::E2": "paper_b",
    }
    return _validate_relations(
        relations, snapshot, roster, set(owners), owners, "dir_test"
    )


def _rel(**overrides) -> dict:
    base = {
        "source_paper_id": "paper_a",
        "target_paper_id": "paper_b",
        "relation_type": "supports",
        "statement": "A supports B.",
        "source_evidence_ids": ["paper_a::E1"],
        "target_evidence_ids": [],
        "difference_dimension": None,
        "comparability_note": "",
        "uncertainty": "",
    }
    base.update(overrides)
    return base


def test_single_span_relation_is_kept_and_warned() -> None:
    valid, _ignored, errors, density = _validate([_rel()])
    assert len(valid) == 1, "a one-span relation must NOT be discarded"
    assert errors == []
    assert len(density) == 1
    assert density[0]["side"] == "source"
    assert density[0]["cited"] == 1
    assert density[0]["required"] == DENSITY_MIN_PER_SIDE


def test_multi_span_relation_produces_no_warning() -> None:
    valid, _ignored, errors, density = _validate(
        [_rel(source_evidence_ids=["paper_a::E1", "paper_a::E2"])]
    )
    assert len(valid) == 1
    assert errors == []
    assert density == []


def test_exempt_relation_single_span_is_not_warned() -> None:
    for rel_type in ("parallel", "not_comparable", "unknown"):
        _valid, _ignored, errors, density = _validate(
            [_rel(relation_type=rel_type)]
        )
        assert errors == [], f"{rel_type} should still validate"
        assert density == [], f"{rel_type} must not be density-floored"


def test_target_side_warning_when_target_cites_one() -> None:
    _valid, _ignored, _errors, density = _validate(
        [
            _rel(
                relation_type="challenges",
                source_evidence_ids=["paper_a::E1", "paper_a::E2"],
                target_evidence_ids=["paper_b::E1"],
                difference_dimension="conclusion",
            )
        ]
    )
    assert [d["side"] for d in density] == ["target"]


def test_both_sides_warned_when_both_under_cite() -> None:
    _valid, _ignored, _errors, density = _validate(
        [
            _rel(
                relation_type="contradicts",
                source_evidence_ids=["paper_a::E1"],
                target_evidence_ids=["paper_b::E1"],
                difference_dimension="conclusion",
            )
        ]
    )
    assert {d["side"] for d in density} == {"source", "target"}


def test_density_warning_carries_direction_and_pair() -> None:
    _valid, _ignored, _errors, density = _validate([_rel()])
    entry = density[0]
    assert entry["direction_id"] == "dir_test"
    assert entry["source_paper_id"] == "paper_a"
    assert entry["target_paper_id"] == "paper_b"
    assert entry["relation_type"] == "supports"
    assert "relation_index" in entry


def test_density_warning_does_not_mask_real_gate_failure() -> None:
    """An unknown evidence ID is still rejected (fail-closed), not warned past."""
    valid, ignored, _errors, _density = _validate(
        [_rel(source_evidence_ids=["paper_a::E99"])]
    )
    assert valid == [], "unknown evidence must not yield a valid relation"
    assert len(ignored) == 1
    assert ignored[0]["reason"] == "model_output_unknown_evidence_ids"


@pytest.mark.parametrize("cited", [["paper_a::E1"], ["paper_a::E1", "paper_a::E2"]])
def test_return_shape_is_four_tuple(cited: list[str]) -> None:
    result = _validate([_rel(source_evidence_ids=cited)])
    assert len(result) == 4
