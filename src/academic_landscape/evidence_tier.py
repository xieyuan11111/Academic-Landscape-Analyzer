"""Evidence tiering for report prose (PRD §10.1 / §10.2).

§10.1 allows six phrasings and each phrasing carries an implicit tier:

* "A 提出……" / "A 的结果显示……"      -> **一手** (one paper's own claim or result)
* 综述转述（无新证据）                -> **二手** (a review restating others' work)
* "多项相对独立的研究支持……" /
  "综合当前语料，可以观察到……"        -> **综合** (cross-study / cross-literature)
* "现有材料提示……但尚不能确认……"      -> **未决** (pending, insufficient or contested)

§10.2's forbidden upgrades ARE the tier boundaries:

* "综述转述 → 独立证据" forbidden          ⟹ 二手 is never promoted to 一手
* "多篇论文 → 多项独立研究" forbidden      ⟹ same evidence family stays 一手
* "研究意见不一 → 原观点已经失效" forbidden ⟹ contested relations stay 未决

This module assigns the tier from the relation vocabulary plus the source
independence signal from ``evidence_family``.  It is code-side classification,
not an LLM judgment: the same inputs always produce the same tier, and the
reason is returned so the report can state *why* a judgment sits in that tier.
"""

from __future__ import annotations

from .relation_types import VALID_RELATION_TYPES

__all__ = [
    "TIER_PRIMARY",
    "TIER_SECONDARY",
    "TIER_SYNTHESIS",
    "TIER_OPEN",
    "EVIDENCE_TIERS",
    "classify_evidence_tier",
]

TIER_PRIMARY = "一手"
TIER_SECONDARY = "二手"
TIER_SYNTHESIS = "综合"
TIER_OPEN = "未决"
EVIDENCE_TIERS = (TIER_PRIMARY, TIER_SECONDARY, TIER_SYNTHESIS, TIER_OPEN)

# Relations that assert a positive claim from one paper to another: the report
# may say "A 的结果显示……" or "多项研究支持……" depending on independence.
_POSITIVE = frozenset({"supports", "extends"})

# Relations that revise or qualify an existing claim. The claim still stands,
# but with a boundary attached — closer to synthesis than to raw primary data.
_REVISING = frozenset({"revises", "qualifies"})

# Relations that mark a disagreement or an unbridgeable gap. §10.2 explicitly
# says opposition must not be written up as settled, so these land in 未决.
_CONTESTED = frozenset({
    "challenges",
    "contradicts",
    "surface_conflict",
    "conditional_difference",
    "same_name_different_meaning",
})

_NO_CONTACT = frozenset({"not_comparable", "unknown", "parallel"})


def classify_evidence_tier(
    relation_type: str,
    *,
    source_independent: bool = True,
    supporting_papers: int = 1,
    source_is_review: bool = False,
) -> tuple[str, str]:
    """Return ``(tier, reason)`` for one cross-paper judgment.

    ``source_independent`` comes from ``evidence_family``: True when the papers
    share neither a dataset nor a co-authorship team.  When it is False they
    are one evidence family and must NOT be counted as independent studies
    (§10.2), so the tier stays 一手.

    ``source_is_review`` marks a paper the family analyzer flagged as
    review-like — it restates earlier conclusions without new evidence
    (§10.2 综述转述), so it is 二手 and is never promoted further.

    ``supporting_papers`` is how many *distinct* papers back the judgment;
    two or more independent ones support the cross-study phrasing 综合.
    """
    if relation_type not in VALID_RELATION_TYPES:
        return TIER_OPEN, f"未知关系类型 {relation_type!r}，按未决处理"

    if relation_type in _NO_CONTACT:
        return (
            TIER_OPEN,
            f"关系类型 {relation_type} 表示两篇论文不可比或平行，尚无支持或反对",
        )
    if relation_type in _CONTESTED:
        return (
            TIER_OPEN,
            f"关系类型 {relation_type} 表示存在分歧，尚不能确认结论",
        )

    if source_is_review:
        return (
            TIER_SECONDARY,
            "来源为综述性材料，转述早期结论而未增加新证据，不能作为独立证据",
        )

    if not source_independent:
        return (
            TIER_PRIMARY,
            "论文属同一证据家族（数据集或团队重叠），不能计为多项独立研究",
        )

    if relation_type in _REVISING:
        return TIER_SYNTHESIS, f"关系类型 {relation_type} 是跨文献的边界修正/限定"

    if relation_type in _POSITIVE:
        if supporting_papers >= 2:
            return TIER_SYNTHESIS, f"{supporting_papers} 项相互独立的研究支持"
        return TIER_PRIMARY, "仅有单篇论文报告其自身结果"

    return TIER_SYNTHESIS, f"关系类型 {relation_type} 依据当前语料综合得出"
