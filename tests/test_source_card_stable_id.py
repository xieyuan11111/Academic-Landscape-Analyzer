"""S1 regression: source-card ids must survive a re-run.

The card id used to be ``sha256(snapshot_id + judgment)`` and ``judgment``
ends with the LLM-generated ``relation.statement``, which is rephrased on every
run. Re-running the **same** snapshot therefore produced almost all-new card
ids (r4 vs r5: only 4 of 248 shared). That breaks PRD §13.2 #3 (idempotency)
and §14.3 (reproducibility).

A relation already carries a stable ``relation_id`` (see
``deterministic_relation_id``: snapshot + source + target + type + sorted
evidence — no free text). When the card is built from a relation, the card id
must derive from that stable key so a rephrased statement leaves it untouched.
"""

from __future__ import annotations

import pytest

from academic_landscape.source_cards import (
    SourceRef,
    build_source_card,
    card_id_from_stable_key,
    deterministic_card_id,
)

SNAP = "snapshot_x"


def _ref(pid: str = "paper_a", eid: str = "paper_a::E1") -> SourceRef:
    return SourceRef(paper_id=pid, evidence_id=eid)


def _build(*, stable_key: str | None, judgment: str):
    return build_source_card(
        snapshot_id=SNAP,
        stable_key=stable_key,
        judgment=judgment,
        judgment_type="B",
        supporting=(_ref(),),
    )


def test_same_stable_key_rephrased_judgment_keeps_card_id() -> None:
    """The core S1 invariant: wording changes, id does not."""
    c1 = _build(stable_key="rel_abc", judgment="paper_a --supports--> paper_b: first wording")
    c2 = _build(stable_key="rel_abc", judgment="paper_a --supports--> paper_b: other wording entirely")
    assert c1.card_id == c2.card_id


def test_different_stable_key_gives_different_card_id() -> None:
    c1 = _build(stable_key="rel_abc", judgment="same judgment text")
    c2 = _build(stable_key="rel_xyz", judgment="same judgment text")
    assert c1.card_id != c2.card_id


def test_stable_key_id_is_deterministic_and_typed() -> None:
    a = card_id_from_stable_key(SNAP, "rel_abc")
    b = card_id_from_stable_key(SNAP, "rel_abc")
    assert a == b
    assert a.startswith("card_")


def test_without_stable_key_falls_back_to_judgment_hash() -> None:
    """Cards not built from a relation keep the legacy judgment-derived id."""
    c1 = _build(stable_key=None, judgment="judgment one")
    c2 = _build(stable_key=None, judgment="judgment two")
    assert c1.card_id != c2.card_id
    assert c1.card_id == deterministic_card_id(SNAP, "judgment one")


def test_blank_stable_key_is_treated_as_missing() -> None:
    """An empty stable key must not silently collapse every card onto one id."""
    with pytest.raises(ValueError):
        _build(stable_key="", judgment="j")
