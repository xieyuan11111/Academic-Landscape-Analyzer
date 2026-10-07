"""Tests for FR-08 popularity / innovation / frontier classification."""

from __future__ import annotations

from academic_landscape.frontier_analysis import (
    score_popularity,
    score_innovation,
    classify_thread_frontier,
    build_frontier_report,
)
from academic_landscape.stage_reconstruction import ConceptThread

CURRENT_YEAR = 2026


# ---------- popularity ----------


def test_popularity_hot() -> None:
    years = [2010, 2024, 2024, 2025, 2025, 2026]
    result = score_popularity(years, current_year=CURRENT_YEAR)
    assert result.label == "hot"
    assert result.recent_count == 5


def test_popularity_normal_when_too_few_recent() -> None:
    years = [2010, 2024, 2025]
    assert score_popularity(years, current_year=CURRENT_YEAR).label == "normal"


def test_popularity_cooling() -> None:
    years = [1999, 2010]
    assert score_popularity(years, current_year=CURRENT_YEAR).label == "cooling"


# ---------- innovation ----------


def _thread(name: str, earliest: int, papers: int = 1) -> ConceptThread:
    return ConceptThread(
        thread_id=f"thread_{name}",
        canonical_name=name,
        variants=(name,),
        paper_ids=tuple(f"paper_{name}_{k}" for k in range(papers)),
        years=(earliest,),
        earliest_year=earliest,
        evidence_ids=(),
    )


def test_innovation_innovative_when_many_recent_threads() -> None:
    threads = [
        _thread("a", 2024), _thread("b", 2025), _thread("c", 2026),
        _thread("d", 2010),
    ]
    result = score_innovation(threads, current_year=CURRENT_YEAR)
    assert result.label == "innovative"
    assert result.recent_threads == 3


def test_innovation_incremental_when_mostly_old() -> None:
    threads = [_thread("a", 2010), _thread("b", 2012), _thread("c", 2025)]
    assert score_innovation(threads, current_year=CURRENT_YEAR).label == "incremental"


# ---------- frontier (thread level) ----------


def _propagated_thread(name: str, papers: list[str], years: list[int]) -> ConceptThread:
    return ConceptThread(
        thread_id=f"thread_{name}",
        canonical_name=name,
        variants=(name,),
        paper_ids=tuple(papers),
        years=tuple(years),
        earliest_year=years[0],
        evidence_ids=(),
    )


def test_frontier_cross_team_growth_is_emerging() -> None:
    thread = _propagated_thread(
        "hnad", ["paper_a", "paper_b"], [2025, 2026]
    )
    family_index = {"paper_a": "family_0", "paper_b": "family_1"}
    result = classify_thread_frontier(
        thread, family_index, current_year=CURRENT_YEAR
    )
    assert result == "emerging_frontier"


def test_frontier_single_team_growth_is_candidate() -> None:
    thread = _propagated_thread(
        "x", ["paper_a", "paper_b"], [2025, 2026]
    )
    family_index = {"paper_a": "family_0", "paper_b": "family_0"}
    result = classify_thread_frontier(
        thread, family_index, current_year=CURRENT_YEAR
    )
    assert result == "candidate"


def test_frontier_old_thread_is_established() -> None:
    thread = _propagated_thread(
        "old", ["paper_a", "paper_b"], [2010, 2013]
    )
    family_index = {"paper_a": "family_0", "paper_b": "family_1"}
    result = classify_thread_frontier(
        thread, family_index, current_year=CURRENT_YEAR
    )
    assert result == "established"


def test_old_single_paper_thread_is_candidate_not_frontier() -> None:
    thread = _propagated_thread("solo", ["paper_a"], [2026])
    result = classify_thread_frontier(
        thread, {"paper_a": "family_0"}, current_year=CURRENT_YEAR
    )
    assert result == "candidate"


def test_frontier_report_excludes_stale_isolated_mentions() -> None:
    # a single-paper thread born long ago never propagated: not a frontier candidate
    stale = _propagated_thread("old_solo", ["paper_old"], [2010])
    fresh = _propagated_thread("new_solo", ["paper_new"], [2026])
    direction_papers = {"dir_1": ["paper_old", "paper_new"]}
    years = {"paper_old": 2010, "paper_new": 2026}
    family_index = {"paper_old": "family_0", "paper_new": "family_1"}
    report = build_frontier_report(
        direction_papers=direction_papers,
        years_by_id=years,
        threads_by_direction={"dir_1": [stale, fresh]},
        family_index=family_index,
        current_year=CURRENT_YEAR,
    )
    names = [entry["thread"] for entry in report.frontier_threads]
    assert "old_solo" not in names
    assert "new_solo" in names


# ---------- aggregated report ----------


def test_build_frontier_report_shape() -> None:
    threads = [
        _propagated_thread("hnad", ["paper_a", "paper_b"], [2025, 2026]),
    ]
    direction_papers = {
        "dir_1": ["paper_a", "paper_b"],
    }
    years = {"paper_a": 2025, "paper_b": 2026}
    family_index = {"paper_a": "family_0", "paper_b": "family_1"}
    report = build_frontier_report(
        direction_papers=direction_papers,
        years_by_id=years,
        threads_by_direction={"dir_1": threads},
        family_index=family_index,
        current_year=CURRENT_YEAR,
    )
    assert "dir_1" in report.direction_signals
    signal = report.direction_signals["dir_1"]
    assert signal["frontier"] == "emerging_frontier"
    assert signal["popularity"] in {"hot", "normal"}
    assert len(report.frontier_threads) == 1


# ---------- FR-08 candidate items (full fields) ----------


def _build_report(threads, family_index, years, direction="dir_1", papers=None):
    if papers is None:
        papers = [paper_id for thread in threads for paper_id in thread.paper_ids]
    return build_frontier_report(
        direction_papers={direction: papers},
        years_by_id=years,
        threads_by_direction={direction: threads},
        family_index=family_index,
        current_year=CURRENT_YEAR,
    )


def test_candidate_cross_family_is_frontier_formed() -> None:
    # emerging_frontier + cross-family -> frontier_formed / formed / cross_team True
    thread = _propagated_thread("hnad", ["paper_a", "paper_b"], [2025, 2026])
    family_index = {"paper_a": "family_0", "paper_b": "family_1"}
    years = {"paper_a": 2025, "paper_b": 2026}
    report = _build_report([thread], family_index, years)
    assert len(report.candidates) == 1
    item = report.candidates[0]
    assert item.thread == "hnad"
    assert item.tier == "emerging_frontier"
    assert item.category == "frontier_formed"
    assert item.maturity == "formed"
    assert item.cross_team is True


def test_candidate_recent_birth_is_innovation() -> None:
    # single-paper thread born in the recent window -> category "innovation"
    thread = _propagated_thread("new_solo", ["paper_new"], [2026])
    family_index = {"paper_new": "family_1"}
    years = {"paper_new": 2026}
    report = _build_report([thread], family_index, years)
    names = [item.thread for item in report.candidates]
    assert "new_solo" in names
    item = next(item for item in report.candidates if item.thread == "new_solo")
    assert item.tier == "candidate"
    assert item.category == "innovation"
    assert item.maturity == "early"


def test_candidates_exclude_stale_isolated_mentions() -> None:
    # regression: the stale single-paper filter also applies to candidates
    stale = _propagated_thread("old_solo", ["paper_old"], [2010])
    fresh = _propagated_thread("new_solo", ["paper_new"], [2026])
    family_index = {"paper_old": "family_0", "paper_new": "family_1"}
    years = {"paper_old": 2010, "paper_new": 2026}
    report = _build_report([stale, fresh], family_index, years)
    names = [item.thread for item in report.candidates]
    assert "old_solo" not in names
    assert "new_solo" in names


def test_candidate_earliest_corpus_year_matches_thread() -> None:
    thread = _propagated_thread("hnad", ["paper_a", "paper_b", "paper_c"], [2019, 2025, 2026])
    family_index = {
        "paper_a": "family_0",
        "paper_b": "family_1",
        "paper_c": "family_2",
    }
    years = {"paper_a": 2019, "paper_b": 2025, "paper_c": 2026}
    report = _build_report([thread], family_index, years)
    item = next(item for item in report.candidates if item.thread == "hnad")
    assert item.earliest_corpus_year == thread.earliest_year == 2019


def test_candidate_recent_growth_expanding_and_stable() -> None:
    # >= 2 distinct years inside the recent window -> expanding
    expanding = _propagated_thread("grow", ["paper_a", "paper_b"], [2025, 2026])
    # only one distinct recent year -> stable (no descending data => never "declining")
    stable = _propagated_thread("flat", ["paper_c"], [2026])
    family_index = {
        "paper_a": "family_0",
        "paper_b": "family_1",
        "paper_c": "family_2",
    }
    years = {"paper_a": 2025, "paper_b": 2026, "paper_c": 2026}
    report = _build_report([expanding, stable], family_index, years)
    by_name = {item.thread: item for item in report.candidates}
    assert by_name["grow"].recent_growth == "expanding"
    assert by_name["flat"].recent_growth == "stable"


def test_candidate_popularity_category_for_old_single_team() -> None:
    # tier candidate, earliest_year outside the recent window -> "popularity"
    thread = _propagated_thread("old_team", ["paper_a", "paper_b"], [2010, 2026])
    family_index = {"paper_a": "family_0", "paper_b": "family_0"}
    years = {"paper_a": 2010, "paper_b": 2026}
    report = _build_report([thread], family_index, years)
    item = next(item for item in report.candidates if item.thread == "old_team")
    assert item.tier == "candidate"
    assert item.category == "popularity"
    assert item.cross_team is False


def test_candidate_unfilled_fields_stay_empty() -> None:
    # PRD P6: code never fabricates content for LLM-owned fields
    thread = _propagated_thread("hnad", ["paper_a", "paper_b"], [2025, 2026])
    family_index = {"paper_a": "family_0", "paper_b": "family_1"}
    years = {"paper_a": 2025, "paper_b": 2026}
    report = _build_report([thread], family_index, years, papers=["paper_a", "paper_b"])
    item = report.candidates[0]
    assert item.novelty_note == ""
    assert item.relation_to_mainstream == ""
    assert item.limitations_and_counter == ""
    assert item.direction_ids == ("dir_1",)
    assert item.evidence == tuple(thread.evidence_ids)


def test_frontier_report_default_candidates_empty_tuple() -> None:
    # backward compatibility: old 2-field construction must keep working
    from academic_landscape.frontier_analysis import FrontierReport

    report = FrontierReport(direction_signals={}, frontier_threads=())
    assert report.candidates == ()
