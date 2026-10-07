"""Tests for FR-10 traceable source cards (contract §FR-10, TDD RED first).

Covers the eight required cases: legal card construction + deterministic card_id,
judgment_type vocabulary gate, non-empty supporting gate, evidence_id global-shape
gate, dict round-trip consistency, optional revising/opposing, metadata fields
(time_attribute / applicability / uncertainty), and snapshot-id sensitivity of
the deterministic card_id.
"""

from __future__ import annotations

import pytest

from academic_landscape.source_cards import (
    SourceCard,
    SourceRef,
    build_source_card,
    deterministic_card_id,
    source_card_to_dict,
)

SNAPSHOT_ID = "snapshot_test"


# ---- helpers ---------------------------------------------------------------


def _ref(evidence_id: str = "paper_a::E1") -> SourceRef:
    """A minimal but realistic supporting reference (globalized evidence_id)."""
    return SourceRef(
        paper_id="paper_a",
        title="Anchor paper",
        authors="Doe, J.",
        year=2021,
        evidence_id=evidence_id,
        quote="anchor quote about the claim",
        locator="page 3 / section 2.1",
        page_number=3,
    )


# ---- tests -----------------------------------------------------------------


def test_valid_source_card_constructed_with_deterministic_card_id() -> None:
    supporting = (_ref(),)
    card = build_source_card(
        snapshot_id=SNAPSHOT_ID,
        judgment="The anchor paper supports X.",
        judgment_type="A",
        supporting=supporting,
        time_attribute="corpus_earliest_2014",
    )
    assert isinstance(card, SourceCard)
    assert card.judgment == "The anchor paper supports X."
    assert card.judgment_type == "A"
    assert card.supporting == supporting
    assert card.revising == ()
    assert card.opposing == ()
    assert card.snapshot_id == SNAPSHOT_ID
    assert card.time_attribute == "corpus_earliest_2014"

    expected = deterministic_card_id(SNAPSHOT_ID, "The anchor paper supports X.")
    assert card.card_id == expected
    assert card.card_id.startswith("card_")
    assert len(card.card_id) == len("card_") + 16

    # Determinism: same inputs → same card_id
    again = deterministic_card_id(SNAPSHOT_ID, "The anchor paper supports X.")
    assert again == expected


def test_judgment_type_outside_vocabulary_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_source_card(
            snapshot_id=SNAPSHOT_ID,
            judgment="Some judgment.",
            judgment_type="Z",
            supporting=(_ref(),),
        )


def test_empty_supporting_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_source_card(
            snapshot_id=SNAPSHOT_ID,
            judgment="Some judgment.",
            judgment_type="B",
            supporting=(),
        )


def test_blank_judgment_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_source_card(
            snapshot_id=SNAPSHOT_ID,
            judgment="   \t\n  ",
            judgment_type="C",
            supporting=(_ref(),),
        )


def test_evidence_id_must_contain_global_separator() -> None:
    # Plain local "E1" without "paper_a::" prefix must be rejected.
    bad_ref = SourceRef(paper_id="paper_a", evidence_id="E1")
    with pytest.raises(ValueError):
        build_source_card(
            snapshot_id=SNAPSHOT_ID,
            judgment="Some judgment.",
            judgment_type="D",
            supporting=(bad_ref,),
        )

    # Empty evidence_id is also rejected.
    empty_ref = SourceRef(paper_id="paper_a", evidence_id="")
    with pytest.raises(ValueError):
        build_source_card(
            snapshot_id=SNAPSHOT_ID,
            judgment="Some judgment.",
            judgment_type="D",
            supporting=(empty_ref,),
        )

    # Same shape gate applies to revising/opposing, not only supporting.
    bad_revising = SourceRef(paper_id="paper_b", evidence_id="only_local")
    with pytest.raises(ValueError):
        build_source_card(
            snapshot_id=SNAPSHOT_ID,
            judgment="Some judgment.",
            judgment_type="A",
            supporting=(_ref(),),
            revising=(bad_revising,),
        )


def test_source_card_to_dict_roundtrip_consistent() -> None:
    supporting = (_ref("paper_a::E1"),)
    revising = (SourceRef(paper_id="paper_b", evidence_id="paper_b::E3"),)
    opposing = (SourceRef(paper_id="paper_c", evidence_id="paper_c::E7"),)
    card = build_source_card(
        snapshot_id=SNAPSHOT_ID,
        judgment="Judgment with full metadata.",
        judgment_type="B",
        supporting=supporting,
        revising=revising,
        opposing=opposing,
        time_attribute="corpus_earliest_2014",
        applicability="applies to large-scale settings",
        uncertainty="limited to corpus_2014_2024",
    )

    as_dict = source_card_to_dict(card)
    assert isinstance(as_dict, dict)
    assert as_dict["card_id"] == card.card_id
    assert as_dict["snapshot_id"] == SNAPSHOT_ID
    assert as_dict["judgment"] == "Judgment with full metadata."
    assert as_dict["judgment_type"] == "B"
    assert as_dict["time_attribute"] == "corpus_earliest_2014"
    assert as_dict["applicability"] == "applies to large-scale settings"
    assert as_dict["uncertainty"] == "limited to corpus_2014_2024"

    # The three source buckets are serialized as lists of dicts with the
    # exact SourceRef fields. Round-trip back to typed objects should match.
    assert as_dict["supporting"] == [
        {
            "paper_id": "paper_a",
            "title": "Anchor paper",
            "authors": "Doe, J.",
            "year": 2021,
            "evidence_id": "paper_a::E1",
            "quote": "anchor quote about the claim",
            "locator": "page 3 / section 2.1",
            "page_number": 3,
            "verified_at": "",
        }
    ]
    assert as_dict["revising"][0]["paper_id"] == "paper_b"
    assert as_dict["revising"][0]["evidence_id"] == "paper_b::E3"
    assert as_dict["opposing"][0]["paper_id"] == "paper_c"
    assert as_dict["opposing"][0]["evidence_id"] == "paper_c::E7"


def test_revising_and_opposing_default_to_empty_and_optional() -> None:
    # Default arguments: revising/opposing are empty tuples; card still builds.
    card = build_source_card(
        snapshot_id=SNAPSHOT_ID,
        judgment="Single-sided supporting only.",
        judgment_type="A",
        supporting=(_ref(),),
    )
    assert card.revising == ()
    assert card.opposing == ()
    # And the dict form reflects empty buckets as empty lists.
    assert source_card_to_dict(card)["revising"] == []
    assert source_card_to_dict(card)["opposing"] == []


def test_time_attribute_applicability_uncertainty_are_recorded() -> None:
    card = build_source_card(
        snapshot_id=SNAPSHOT_ID,
        judgment="Metadata check.",
        judgment_type="C",
        supporting=(_ref(),),
        time_attribute="corpus_earliest_2014",
        applicability="only when sample size > 100",
        uncertainty="evidence base narrowed to 3 papers",
    )
    assert card.time_attribute == "corpus_earliest_2014"
    assert card.applicability == "only when sample size > 100"
    assert card.uncertainty == "evidence base narrowed to 3 papers"

    serialized = source_card_to_dict(card)
    assert serialized["time_attribute"] == "corpus_earliest_2014"
    assert serialized["applicability"] == "only when sample size > 100"
    assert serialized["uncertainty"] == "evidence base narrowed to 3 papers"


def test_card_id_changes_with_snapshot_id() -> None:
    same_judgment = "Stable wording for the same judgment."
    snap_a = deterministic_card_id("snapshot_a", same_judgment)
    snap_b = deterministic_card_id("snapshot_b", same_judgment)
    assert snap_a != snap_b
    assert snap_a.startswith("card_") and snap_b.startswith("card_")

    # Same snapshot + same judgment → stable.
    assert deterministic_card_id("snapshot_a", same_judgment) == snap_a
