"""D13: a research direction must state how coherent its members actually are.

Measured on MAG-50: KMeans groups papers by topical similarity, but "topically
similar" is not "research-comparable". A 13-paper direction spanned theory
modelling, cyanobacteria, hummingbirds, sticklebacks and anammox engineering;
76 of its 78 evaluated pairs came back ``not_comparable``.

The relation layer measured that honestly and then threw the measurement away:
the report never said a direction was a grab-bag. This module computes a
coherence score from what relation extraction actually judged, and the report
has to show it.

Design decisions (from the user's delegation of this call to me):

- **Mark, do not split, do not exclude.** Splitting/merging would change
  direction identity and break idempotence and version comparability;
  excluding would make an incoherent direction invisible — the exact failure
  mode this project keeps being bitten by. Instead the direction is demoted to
  "pending review" and stops being stated as a mainstream direction.
- **Say which number you have.** Pair-chunked directions were evaluated
  exhaustively (coverage 1.0, exact score). Others are a *lower bound* — the
  unjudged pairs might hold substantive relations nobody asked about.
"""

from __future__ import annotations

from typing import Any

from academic_landscape.direction_coherence import (
    SUBSTANTIVE_RELATION_TYPES,
    DirectionCoherence,
    annotate_coherence,
    compute_coherence,
)
from academic_landscape.landscape_report import (
    CorpusFacts,
    LandscapeResult,
    render_landscape_report,
)
from academic_landscape.relation_extraction import (
    PaperRelation,
    RelationExtractionResult,
)


class _Direction:
    def __init__(self, paper_ids: tuple[str, ...], label: str = "D1") -> None:
        self.direction_id = "dir_1"
        self.cluster_id = 0
        self.label = label
        self.description = ""
        self.status = "emerging"
        self.paper_ids = paper_ids
        self.item_ids = ()


def _relation(source: str, target: str, relation_type: str) -> PaperRelation:
    """Real shape: the pipeline hands the report Pydantic ``PaperRelation``s."""
    return PaperRelation(
        relation_id=f"rel_{source}_{target}",
        source_paper_id=source,
        target_paper_id=target,
        relation_type=relation_type,
        statement="s",
        source_evidence_ids=[f"{source}::E1"],
        target_evidence_ids=[],
        difference_dimension=None,
        comparability_note="",
        uncertainty="",
        snapshot_id="snapshot_x",
    )


def _result(relations: list[dict], direction: Any) -> RelationExtractionResult:
    return RelationExtractionResult(
        direction_id=direction.direction_id,
        relations=tuple(relations),
        ignored_relations=(),
        validation_errors=(),
    )


PAPERS = ("paper_a", "paper_b", "paper_c")


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------


def test_total_pairs_is_combinatorial() -> None:
    direction = _Direction(PAPERS)
    coherence = compute_coherence(direction, _result([], direction))
    assert coherence.total_pairs == 3  # C(3,2)


def test_substantive_relations_raise_coherence() -> None:
    """`parallel` now counts as contact (B vocabulary): 2/3 substantive."""
    direction = _Direction(PAPERS)
    relations = [
        _relation("paper_a", "paper_b", "supports"),
        _relation("paper_a", "paper_c", "not_comparable"),
        _relation("paper_b", "paper_c", "parallel"),
    ]
    coherence = compute_coherence(direction, _result(relations, direction))

    assert coherence.judged_pairs == 3
    assert coherence.substantive_pairs == 2  # supports + parallel
    assert coherence.coherence == 2 / 3


def test_coverage_says_whether_the_score_is_exact() -> None:
    """A direction judged pair-by-pair has coverage 1 and an exact score."""
    direction = _Direction(PAPERS)
    relations = [
        _relation("paper_a", "paper_b", "supports"),
        _relation("paper_a", "paper_c", "not_comparable"),
        _relation("paper_b", "paper_c", "parallel"),
    ]
    coherence = compute_coherence(direction, _result(relations, direction))
    assert coherence.coverage == 1.0
    assert coherence.is_exact is True

    partial = _result(relations[:1], direction)
    coherence2 = compute_coherence(direction, partial)
    assert coherence2.coverage < 1.0
    assert coherence2.is_exact is False


def test_lower_bound_counts_unjudged_pairs_against_us() -> None:
    """With one judged pair out of three, the floor is 1/3 even if all substantive."""
    direction = _Direction(PAPERS)
    coherence = compute_coherence(
        direction, _result([_relation("paper_a", "paper_b", "supports")], direction)
    )
    assert coherence.lower_bound == 1 / 3
    assert coherence.coherence == 1.0  # of what was judged
    assert coherence.is_exact is False


def test_substantive_vocabulary_excludes_avoidance_verdicts() -> None:
    """Only true avoidance verdicts count against coherence.

    ``parallel`` is *not* avoidance: it says the two papers do work of the same
    kind and are comparable, just not claiming to extend or challenge each other
    (PRD R5 calls exactly this out as "existing but not independent" contact —
    it is contact, not absence of contact). Counting it as non-substantive made
    7 of 10 MAG-50 directions demote to ``pending_review``, which destroys the
    signal's discrimination power. Only ``not_comparable`` ("these papers are not
    even comparable") and ``unknown`` ("we could not tell") are avoidance.
    """
    assert "not_comparable" not in SUBSTANTIVE_RELATION_TYPES
    assert "unknown" not in SUBSTANTIVE_RELATION_TYPES
    assert "parallel" in SUBSTANTIVE_RELATION_TYPES
    assert "supports" in SUBSTANTIVE_RELATION_TYPES
    assert "challenges" in SUBSTANTIVE_RELATION_TYPES


def test_parallel_counts_as_contact() -> None:
    """The measured sanity check behind the vocabulary choice.

    MAG-50 r5: ``Organic Matter Molecular Composition`` has 9 not_comparable +
    7 parallel + 1 support-ish out of 16 judged pairs. Under the old vocabulary
    (parallel non-substantive) it scored 0% and demoted; under this one it scores
    44% and stays. ``Biodiversity Insurance`` (68 not_comparable of 76) stays
    near 11% and is still demoted — exactly the discrimination we want.
    """
    direction = _Direction(PAPERS)
    relations = [
        _relation("paper_a", "paper_b", "not_comparable"),
        _relation("paper_a", "paper_c", "parallel"),
        _relation("paper_b", "paper_c", "parallel"),
    ]
    coherence = compute_coherence(direction, _result(relations, direction))

    assert coherence.substantive_pairs == 2  # both parallel pairs count
    assert coherence.coherence == 2 / 3
    assert coherence.low_coherence is False


def test_no_judged_pairs_is_zero_not_perfect() -> None:
    """An empty relation layer must not read as a perfectly coherent direction."""
    direction = _Direction(PAPERS)
    coherence = compute_coherence(direction, _result([], direction))
    assert coherence.coherence == 0.0
    assert coherence.lower_bound == 0.0


# ---------------------------------------------------------------------------
# marking and demotion
# ---------------------------------------------------------------------------


def test_low_coherence_is_marked_and_demoted() -> None:
    direction = _Direction(PAPERS, label="grab bag")
    relations = [
        _relation("paper_a", "paper_b", "not_comparable"),
        _relation("paper_a", "paper_c", "not_comparable"),
        _relation("paper_b", "paper_c", "not_comparable"),
    ]
    coherence = compute_coherence(direction, _result(relations, direction))

    assert coherence.low_coherence is True
    assert coherence.status_override == "pending_review"


def test_coherent_direction_keeps_its_status() -> None:
    direction = _Direction(PAPERS)
    relations = [
        _relation("paper_a", "paper_b", "supports"),
        _relation("paper_a", "paper_c", "extends"),
        _relation("paper_b", "paper_c", "supports"),
    ]
    coherence = compute_coherence(direction, _result(relations, direction))

    assert coherence.low_coherence is False
    assert coherence.status_override == ""


def test_annotate_covers_every_direction() -> None:
    directions = [_Direction(PAPERS)]
    results = [_result([], directions[0])]

    annotated = annotate_coherence(directions, results)

    assert set(annotated) == {"dir_1"}
    assert isinstance(annotated["dir_1"], DirectionCoherence)


def test_empty_relation_layer_still_demotes() -> None:
    """'We judged nothing' is the weakest coherence claim of all."""
    direction = _Direction(PAPERS, label="unjudged")
    coherence = compute_coherence(direction, _result([], direction))
    assert coherence.low_coherence is True


# ---------------------------------------------------------------------------
# report visibility
# ---------------------------------------------------------------------------


def _landscape(directions: tuple, relation_results: tuple) -> LandscapeResult:
    facts = CorpusFacts(
        document_count=3, parsed_count=3, paper_count=3,
        year_min=2020, year_max=2022, parse_failures=(), duplicates=(),
    )
    return LandscapeResult(
        snapshot_id="snapshot_x",
        generated_at="2026-09-24T00:00:00+00:00",
        corpus=facts,
        directions=directions, comparisons=(), reconstructions=(),
        family_report=None, frontier_report=None,
        merged_frontiers=(), future_work={},
        relation_results=relation_results,
    )


def test_report_shows_coherence_for_each_direction() -> None:
    direction = _Direction(PAPERS, label="Grab Bag Studies")
    relations = [
        _relation("paper_a", "paper_b", "not_comparable"),
        _relation("paper_a", "paper_c", "not_comparable"),
        _relation("paper_b", "paper_c", "supports"),
    ]
    text = render_landscape_report(_landscape(
        (direction,), (_result(relations, direction),)
    ))

    assert "方向一致性" in text
    assert "33" in text  # 1/3 ≈ 33%


def test_report_demotes_low_coherence_directions() -> None:
    direction = _Direction(PAPERS, label="Grab Bag Studies")
    relations = [
        _relation("paper_a", "paper_b", "not_comparable"),
        _relation("paper_a", "paper_c", "not_comparable"),
        _relation("paper_b", "paper_c", "not_comparable"),
    ]
    text = render_landscape_report(_landscape(
        (direction,), (_result(relations, direction),)
    ))

    assert "low_coherence" in text
    assert "待复核" in text
    # the demotion must be visible in the heading, not buried in prose
    assert "待复核" in text.split("###")[1]


def test_report_says_when_the_score_is_only_a_lower_bound() -> None:
    direction = _Direction(PAPERS)
    relations = [_relation("paper_a", "paper_b", "supports")]
    text = render_landscape_report(_landscape(
        (direction,), (_result(relations, direction),)
    ))

    assert "下界" in text or "lower bound" in text.lower()


def test_report_aggregates_coherence_in_scope_section() -> None:
    direction = _Direction(PAPERS)
    relations = [_relation("paper_a", "paper_b", "not_comparable")]
    text = render_landscape_report(_landscape(
        (direction,), (_result(relations, direction),)
    ))

    scope = text.split("## 9.2")[1].split("## 9.3")[0]
    assert "一致性" in scope
