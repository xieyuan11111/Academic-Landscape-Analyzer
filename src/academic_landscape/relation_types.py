"""Canonical FR-04 relation vocabulary — the single source of truth.

Why this module exists
----------------------
The first real MAG-50 production run aborted at graph projection with
``invalid_relation_type``: ``build_relation_prompt`` offered the model 12
relation types while the payload validator accepted only 9. Three copies of the
vocabulary had drifted apart, and nothing forced them to agree.

Every consumer now imports from here:

- payload validation (``semantic``)
- relation extraction + its prompt (``relation_extraction``)
- Neo4j relationship labels (``graph_projector``)
- the legacy corpus synthesis prompt (``prompts``)

Never re-declare the vocabulary locally. ``tests/test_relation_vocabulary.py``
fails if a copy drifts again.

This module must stay a dependency-free leaf: importing package modules here
would create cycles.
"""

from __future__ import annotations

# Ordered exactly as the relation prompt presents the vocabulary to the model.
RELATION_TYPES_IN_PROMPT_ORDER: tuple[str, ...] = (
    "supports",
    "extends",
    "revises",
    "qualifies",
    "challenges",
    "contradicts",
    "parallel",
    "same_name_different_meaning",
    "surface_conflict",
    "conditional_difference",
    "not_comparable",
    "unknown",
)

# The definitive membership test: derived from the ordered tuple so the prompt
# order and the accepted set can never disagree.
VALID_RELATION_TYPES: frozenset[str] = frozenset(RELATION_TYPES_IN_PROMPT_ORDER)

# Pipe-joined form used verbatim inside prompts.
RELATION_PROMPT_VOCABULARY: str = "|".join(RELATION_TYPES_IN_PROMPT_ORDER)

# Neo4j relationship label for each relation type.
RELATION_EDGE_LABELS: dict[str, str] = {
    "supports": "SUPPORTS",
    "extends": "EXTENDS",
    "revises": "REVISES",
    "qualifies": "QUALIFIES",
    "challenges": "CHALLENGES",
    "contradicts": "CONTRADICTS",
    "parallel": "PARALLEL_TO",
    "same_name_different_meaning": "SAME_NAME_DIFFERENT_MEANING",
    "surface_conflict": "SURFACE_CONFLICT",
    "conditional_difference": "CONDITIONAL_DIFFERENCE",
    "not_comparable": "NOT_COMPARABLE_WITH",
    "unknown": "UNKNOWN_RELATION",
}


def _self_check() -> None:
    missing = VALID_RELATION_TYPES - set(RELATION_EDGE_LABELS)
    extra = set(RELATION_EDGE_LABELS) - VALID_RELATION_TYPES
    if missing or extra:
        raise AssertionError(
            "relation edge labels must cover the vocabulary exactly; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    if len(RELATION_TYPES_IN_PROMPT_ORDER) != len(VALID_RELATION_TYPES):
        raise AssertionError("prompt order contains duplicate relation types")


_self_check()


# ---------------------------------------------------------------------------
# Evidence-density policy (added 2026-09-27)
# ---------------------------------------------------------------------------
# The first full MAG-209 run produced 7,652 relations citing 1.85 evidence
# spans on average, while the authority pool holds 11.3 spans per paper.  76.6%
# of source cards ended up with a single supporting span.  The cause was the
# prompt's "at least one" wording plus a one-element JSON example, both of
# which taught the model that one ID is a complete answer.
#
# Assertion-bearing relations therefore need a real evidence floor.  The floor
# is deliberately scoped: "parallel", "not_comparable" and "unknown" assert
# nothing about either paper's position, so demanding two spans from them would
# force fabricated citations.  Everything else states or bounds a claim and must
# be traceable to more than one contiguous span.
#
# One span stays legal everywhere (a paper may genuinely contain only one
# relevant sentence) — the floor is a *prompt-and-review* signal surfaced to the
# caller, not a silent drop.  See ``_validate_relations`` gate 3b.

DENSITY_FLOORED_RELATIONS: frozenset[str] = frozenset(
    RELATION_TYPES_IN_PROMPT_ORDER
) - {"parallel", "not_comparable", "unknown"}

# Minimum spans per side for a floored relation.
DENSITY_MIN_PER_SIDE: int = 2


def _density_self_check() -> None:
    unknown = DENSITY_FLOORED_RELATIONS - VALID_RELATION_TYPES
    if unknown:
        raise AssertionError(
            f"density policy names unknown relation types: {sorted(unknown)}"
        )
    floored = DENSITY_FLOORED_RELATIONS
    # These must never be floored: they assert no position to support.
    for must_stay_soft in ("parallel", "not_comparable", "unknown"):
        if must_stay_soft in floored:
            raise AssertionError(
                f"{must_stay_soft} asserts no claim and must not be density-floored"
            )
    # challenges/contradicts already require target evidence (a separate gate);
    # they must be floored or a one-sided rebuttal would pass.
    for must_be_floored in ("challenges", "contradicts"):
        if must_be_floored not in floored:
            raise AssertionError(f"{must_be_floored} must be density-floored")


_density_self_check()
