"""D13: how coherent is a research direction, judged by what we actually know.

KMeans groups papers by topical similarity; "topically similar" is not
"research-comparable". On MAG-50 a 13-paper direction spanned theory modelling,
cyanobacteria, hummingbirds, sticklebacks and anammox engineering — 76 of its 78
evaluated pairs came back ``not_comparable``. Relation extraction measured that
honestly; nothing told the reader.

This module turns those measurements into a per-direction coherence score and a
marking policy.

Numbers
-------
``total_pairs``     = C(n, 2) — every paper pair in the direction.
``judged_pairs``    = pairs relation extraction actually issued a verdict for
                      (a pair with a relation row counts once, regardless of how
                      many rows describe it).
``substantive_pairs`` = judged pairs whose relation is in
                      :data:`SUBSTANTIVE_RELATION_TYPES`.
``coherence``       = substantive_pairs / judged_pairs, or 0.0 when nothing was
                      judged. An empty relation layer is the weakest possible
                      claim about coherence, not a perfect score.
``coverage``        = judged_pairs / total_pairs. ``coherence`` is only an
                      exact score when coverage is 1.0 (pair-chunked directions);
                      otherwise it is computed on a subset and the honest floor
                      is ``lower_bound`` = substantive_pairs / total_pairs.

Policy (decided after the user delegated this call to me)
--------------------------------------------------------
**Mark, do not split, do not exclude.**
  - Splitting or merging changes direction identity → breaks idempotence and
    version comparability (PRD §13.2 #3, §14.3).
  - Excluding makes an incoherent direction invisible — the exact failure mode
    that hid this problem when the direction's extraction kept truncating to
    zero relations.
  - Instead: mark ``low_coherence``, demote the direction to
    ``pending_review``, and say which number we have (exact vs lower bound).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .relation_types import VALID_RELATION_TYPES

# A relation type that counts as substantive contact between two papers.
# Avoidance verdicts — "not_comparable" ("not even comparable") and "unknown"
# ("could not tell") — are exactly what low coherence is made of, and they are
# excluded. ``parallel`` counts as contact: it says the papers do work of the
# same kind and are comparable, just not claiming to extend or challenge each
# other (PRD R5 treats this as "existing but not independent" contact — that is
# contact, not absence of it). Excluding ``parallel`` made 7 of 10 MAG-50
# directions demote to ``pending_review``, which destroys discrimination.
SUBSTANTIVE_RELATION_TYPES: frozenset[str] = frozenset(
    VALID_RELATION_TYPES - {"not_comparable", "unknown"}
)

# Below this coherence a direction is demoted from a stated direction to a
# candidate that needs human review. 0.3 was chosen so that the measured MAG-50
# grab-bag (≈0.08) is flagged while a direction with majority substantive
# contact is not.
LOW_COHERENCE_THRESHOLD = 0.3


@dataclass(frozen=True)
class DirectionCoherence:
    """Coherence verdict for one direction."""

    direction_id: str
    total_pairs: int
    judged_pairs: int
    substantive_pairs: int
    coherence: float
    lower_bound: float
    coverage: float
    is_exact: bool
    low_coherence: bool
    status_override: str


def compute_coherence(direction: Any, relation_result: Any) -> DirectionCoherence:
    """Score one direction from what relation extraction judged.

    ``relation_result`` accepts anything with a ``relations`` sequence of dicts
    or ``PaperRelation``-like objects (``relation_type`` attribute or key).
    """
    paper_ids = tuple(direction.paper_ids)
    total_pairs = len(paper_ids) * (len(paper_ids) - 1) // 2

    judged: set[frozenset[str]] = set()
    substantive: set[frozenset[str]] = set()
    for relation in getattr(relation_result, "relations", ()) or ():
        if isinstance(relation, dict):
            source = relation.get("source_paper_id", "")
            target = relation.get("target_paper_id", "")
            relation_type = relation.get("relation_type", "")
        else:
            source = getattr(relation, "source_paper_id", "")
            target = getattr(relation, "target_paper_id", "")
            relation_type = getattr(relation, "relation_type", "")
        if not source or not target or source == target:
            continue
        pair = frozenset((source, target))
        judged.add(pair)
        if relation_type in SUBSTANTIVE_RELATION_TYPES:
            substantive.add(pair)

    judged_pairs = len(judged)
    substantive_pairs = len(substantive)
    coherence = substantive_pairs / judged_pairs if judged_pairs else 0.0
    lower_bound = substantive_pairs / total_pairs if total_pairs else 0.0
    coverage = judged_pairs / total_pairs if total_pairs else 0.0
    is_exact = bool(total_pairs) and judged_pairs == total_pairs

    low_coherence = coherence < LOW_COHERENCE_THRESHOLD
    return DirectionCoherence(
        direction_id=direction.direction_id,
        total_pairs=total_pairs,
        judged_pairs=judged_pairs,
        substantive_pairs=substantive_pairs,
        coherence=coherence,
        lower_bound=lower_bound,
        coverage=coverage,
        is_exact=is_exact,
        low_coherence=low_coherence,
        status_override="pending_review" if low_coherence else "",
    )


def annotate_coherence(
    directions: Sequence[Any],
    relation_results: Sequence[Any],
) -> Mapping[str, DirectionCoherence]:
    """Coherence for every direction that has a relation extraction result."""
    results = {getattr(rr, "direction_id", ""): rr for rr in relation_results}
    return {
        direction.direction_id: compute_coherence(direction, results.get(direction.direction_id))
        for direction in directions
    }
