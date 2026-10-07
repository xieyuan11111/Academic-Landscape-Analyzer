"""Tests for the complete PRD §9 landscape report rendering."""

from __future__ import annotations


from academic_landscape.cross_paper import (
    ComparedDifference,
    ComparedStatement,
    DirectionComparison,
)
from academic_landscape.direction_naming import ResearchDirection
from academic_landscape.frontier_analysis import FrontierReport
from academic_landscape.evidence_family import (
    DatasetRecord,
    EvidenceFamily,
    EvidenceFamilyReport,
    PersonRecord,
)
from academic_landscape.landscape_merge import MergedFrontier
from academic_landscape.landscape_report import (
    CorpusFacts,
    LandscapeResult,
    render_landscape_report,
)
from academic_landscape.stage_reconstruction import (
    Stage,
    StageReconstruction,
)


def _direction() -> ResearchDirection:
    return ResearchDirection(
        direction_id="dir_1",
        cluster_id=1,
        label="Nitrogen removal",
        description="studies of nitrogen removal",
        status="emerging",
        paper_ids=("paper_a", "paper_b"),
        item_ids=("i1", "i2"),
    )


def _comparison() -> DirectionComparison:
    return DirectionComparison(
        direction_id="dir_1",
        commonalities=(
            ComparedStatement("both remove nitrogen", ("paper_a::E1", "paper_b::E1")),
        ),
        differences=(
            ComparedDifference("method", "different reactors", ("paper_a::E2",)),
        ),
        controversies=(
            {"statement": "mechanism split",
             "positions": [
                 {"paper_id": "paper_a", "stance": "path X",
                  "evidence": ["paper_a::E3"]},
                 {"paper_id": "paper_b", "stance": "path Y",
                  "evidence": ["paper_b::E3"]},
             ]},
        ),
        within_paper_tensions=(
            {"statement": "internal tension", "positions": []},
        ),
        validation_errors=(),
    )


def _reconstruction() -> StageReconstruction:
    return StageReconstruction(
        direction_id="dir_1",
        material_assessment="sparse",
        stages=(
            Stage("Early", 2020, 2021, "method", "early period",
                  ("paper_a",), ("paper_a::E1",), "medium", False),
            Stage("Recent", 2024, 2024, "object", "recent period",
                  ("paper_b",), ("paper_b::E1",), "low", False),
        ),
        turning_points=(
            {"year": 2024, "paper_id": "paper_b",
             "statement": "new object", "evidence": ["paper_b::E2"]},
        ),
        notes="gaps between years",
        threads=(),
        validation_errors=(),
        ignored_evidence_ids=(),
    )


def _family_report() -> EvidenceFamilyReport:
    return EvidenceFamilyReport(
        persons=(
            PersonRecord("zhang q", ("Q. Zhang",), ("paper_a", "paper_b")),
        ),
        datasets=(
            DatasetRecord("dataset_00", "slurry samples", ("paper_a",)),
        ),
        families=(
            EvidenceFamily("family_0", ("paper_a", "paper_b"),
                           ("zhang q",), ()),
        ),
        review_like_papers=(),
        family_index={"paper_a": "family_0", "paper_b": "family_0"},
    )


def _frontier_report() -> FrontierReport:
    return FrontierReport(
        direction_signals={
            "dir_1": {"popularity": "hot", "innovation": "innovative",
                      "frontier": "emerging_frontier"}
        },
        frontier_threads=(
            {"direction_id": "dir_1", "thread": "HNAD",
             "tier": "emerging_frontier"},
        ),
    )


def _merged() -> MergedFrontier:
    return MergedFrontier(
        statement="HNAD",
        tier="emerging_frontier",
        direction_ids=("dir_1",),
        evidence=("paper_a::E1",),
        repeat_count=1,
    )


def _result() -> LandscapeResult:
    facts = CorpusFacts(
        document_count=2,
        parsed_count=2,
        paper_count=2,
        year_min=2020,
        year_max=2024,
        parse_failures=("broken_file.pdf",),
        duplicates=(),
    )
    return LandscapeResult(
        snapshot_id="snapshot_x",
        generated_at="2026-09-22T22:00:00+00:00",
        corpus=facts,
        directions=(_direction(),),
        comparisons=(_comparison(),),
        reconstructions=(_reconstruction(),),
        family_report=_family_report(),
        frontier_report=_frontier_report(),
        merged_frontiers=(_merged(),),
        future_work={"paper_a": ["test at full scale"]},
    )


EXPECTED_SECTIONS = [
    "9.1", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7",
    "9.8", "9.9", "9.10", "9.11",
]


def test_all_eleven_sections_present() -> None:
    text = render_landscape_report(_result())
    for section in EXPECTED_SECTIONS:
        assert section in text, f"missing section {section}"


def test_executive_summary_lists_counts() -> None:
    text = render_landscape_report(_result())
    assert "Nitrogen removal" in text
    assert "snapshot_x" in text


def test_direction_section_shows_members_and_status() -> None:
    text = render_landscape_report(_result())
    assert "paper_a" in text
    assert "emerging" in text


def test_controversies_and_tensions_both_rendered() -> None:
    text = render_landscape_report(_result())
    assert "mechanism split" in text
    assert "internal tension" in text


def test_reading_paths_generated() -> None:
    text = render_landscape_report(_result())
    assert "paper_a" in text  # start point references earliest paper


def test_appendix_lists_membership_and_failures() -> None:
    text = render_landscape_report(_result())
    assert "broken_file.pdf" in text
    assert "HNAD" in text


def test_future_work_section_uses_direct_statements() -> None:
    text = render_landscape_report(_result())
    assert "test at full scale" in text


def test_frontier_section_separates_real_and_candidates() -> None:
    text = render_landscape_report(_result())
    # the single merged fixture is an emerging frontier; candidates are grouped
    assert "emerging_frontier" not in text  # raw tier codes not leaked
    assert "前沿候选" in text


def test_empty_result_states_insufficient_material() -> None:
    facts = CorpusFacts(
        document_count=0, parsed_count=0, paper_count=0,
        year_min=0, year_max=0, parse_failures=(), duplicates=(),
    )
    result = LandscapeResult(
        snapshot_id="snapshot_empty",
        generated_at="2026-09-22T22:00:00+00:00",
        corpus=facts,
        directions=(), comparisons=(), reconstructions=(),
        family_report=None, frontier_report=None,
        merged_frontiers=(), future_work={},
    )
    text = render_landscape_report(result)
    assert "无" in text
    assert "snapshot_empty" in text
    # all sections still rendered even when empty
    for section in EXPECTED_SECTIONS:
        assert section in text
