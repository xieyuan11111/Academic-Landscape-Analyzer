"""Evidence tier classification (PRD §10.1 / §10.2)."""

from __future__ import annotations

from academic_landscape.evidence_tier import (
    EVIDENCE_TIERS,
    TIER_OPEN,
    TIER_PRIMARY,
    TIER_SECONDARY,
    TIER_SYNTHESIS,
    classify_evidence_tier,
)


def test_positive_single_independent_is_primary() -> None:
    tier, reason = classify_evidence_tier("supports", source_independent=True, supporting_papers=1)
    assert tier == TIER_PRIMARY
    assert "单篇" in reason


def test_positive_two_independent_is_synthesis() -> None:
    """Two independent studies = the cross-study phrasing 综合 (§10.1)."""
    tier, reason = classify_evidence_tier("supports", source_independent=True, supporting_papers=2)
    assert tier == TIER_SYNTHESIS
    assert "独立" in reason


def test_dependent_pair_never_becomes_synthesis() -> None:
    """§10.2: 多篇论文 ≠ 多项独立研究. Same evidence family -> 一手."""
    tier, reason = classify_evidence_tier("supports", source_independent=False, supporting_papers=5)
    assert tier == TIER_PRIMARY
    assert "证据家族" in reason


def test_extends_positive_with_two_is_synthesis() -> None:
    tier, _ = classify_evidence_tier("extends", source_independent=True, supporting_papers=2)
    assert tier == TIER_SYNTHESIS


def test_review_like_source_is_secondary_never_primary() -> None:
    """§10.2: 综述转述 must not be promoted to 独立证据."""
    for relation in ("supports", "extends", "revises", "qualifies"):
        tier, reason = classify_evidence_tier(
            relation, source_independent=True, supporting_papers=3,
            source_is_review=True,
        )
        assert tier == TIER_SECONDARY, relation
        assert "综述" in reason


def test_revises_and_qualifies_are_synthesis() -> None:
    for relation in ("revises", "qualifies"):
        tier, reason = classify_evidence_tier(relation, source_independent=True, supporting_papers=1)
        assert tier == TIER_SYNTHESIS
        assert "综合" in reason or "修正" in reason


def test_contested_relations_are_open() -> None:
    for relation in (
        "challenges",
        "contradicts",
        "surface_conflict",
        "conditional_difference",
        "same_name_different_meaning",
    ):
        tier, reason = classify_evidence_tier(relation, source_independent=True, supporting_papers=3)
        assert tier == TIER_OPEN, relation
        assert "尚不能确认" in reason


def test_no_contact_relations_are_open() -> None:
    for relation in ("not_comparable", "unknown", "parallel"):
        tier, _ = classify_evidence_tier(relation, source_independent=True, supporting_papers=1)
        assert tier == TIER_OPEN, relation


def test_unknown_relation_fails_closed_to_open() -> None:
    tier, reason = classify_evidence_tier("made_up_relation", source_independent=True)
    assert tier == TIER_OPEN
    assert "未" in reason


def test_tiers_are_exactly_four() -> None:
    assert set(EVIDENCE_TIERS) == {"一手", "二手", "综合", "未决"}


def test_classification_is_deterministic() -> None:
    a = classify_evidence_tier("supports", source_independent=True, supporting_papers=2)
    b = classify_evidence_tier("supports", source_independent=True, supporting_papers=2)
    assert a == b
