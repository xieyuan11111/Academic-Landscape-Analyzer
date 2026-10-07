"""Regression tests for the CURRENT_YEAR freeze fix (timeline_events).

The bug: the Gate-3 year ceiling came from a module-level
``CURRENT_YEAR = datetime.now().year`` evaluated at IMPORT time.  A run that
crosses New Year keeps the old ceiling and drops the new year's papers as
``year_mismatch`` — indistinguishable from a paper whose year was genuinely
inferred wrong, because both landed on the same reason string.
"""
from __future__ import annotations

from academic_landscape import timeline_events as te
from academic_landscape.timeline_events import (
    EvolutionEvent,
    _validate_events,
    normalize_events,
)


def _extraction(paper_id: str, quote: str = "quoted text here") -> dict:
    return {
        "paper_id": paper_id,
        "evidence": [
            {
                "evidence_id": f"{paper_id}::E1",
                "artifact_id": f"art_{paper_id}",
                "page_number": 1,
                "quote": quote,
                "locator": "page 1",
            }
        ],
    }


def _event(paper_id: str, year: int) -> EvolutionEvent:
    return EvolutionEvent(
        event_id=f"ev_{paper_id}_{year}",
        event_type="concept_origin",
        subject="subject",
        paper_id=paper_id,
        year=year,
        evidence_ids=[f"{paper_id}::E1"],
        valid_at=f"{year}-01-01",
        snapshot_id="snap",
        analysis_version="v1",
    )


# --------------------------------------------------------------------------
# The parameter must exist and be honoured.
# --------------------------------------------------------------------------

def test_normalize_events_accepts_current_year() -> None:
    import inspect

    params = inspect.signature(normalize_events).parameters
    assert "current_year" in params, "normalize_events must accept current_year"
    assert params["current_year"].default is None, "default must keep old behaviour"


def test_validate_events_uses_the_supplied_ceiling() -> None:
    """A paper dated beyond the injected ceiling is rejected; the same paper
    passes when the ceiling is raised.  Without the parameter the test cannot
    express this at all."""
    extractions = {"p1": _extraction("p1")}
    years = {"p1": 2027}
    events = [_event("p1", 2027)]

    accepted, ignored, _ = _validate_events(
        events, extractions_by_paper=extractions, years=years, current_year=2026
    )
    assert accepted == ()
    assert len(ignored) == 1
    assert ignored[0]["reason"] == "year_beyond_current_year"
    assert ignored[0]["current_year"] == 2026

    accepted2, ignored2, _ = _validate_events(
        events, extractions_by_paper=extractions, years=years, current_year=2027
    )
    assert len(accepted2) == 1, "raising the ceiling must admit the event"
    assert ignored2 == ()


def test_year_beyond_ceiling_is_distinguishable_from_a_wrong_year() -> None:
    """The audit trail must not merge a clock problem with a data problem."""
    extractions = {
        "future": _extraction("future"),
        "mismatch": _extraction("mismatch"),
    }
    # 'future' genuinely sits past the ceiling; 'mismatch' disagrees with its
    # carrier year but is NOT past the ceiling.
    years = {"future": 2030, "mismatch": 1999}
    events = [_event("future", 2030), _event("mismatch", 2020)]

    _, ignored, _ = _validate_events(
        events, extractions_by_paper=extractions, years=years, current_year=2026
    )
    by_paper = {i["event_id"]: i["reason"] for i in ignored}
    assert by_paper["ev_future_2030"] == "year_beyond_current_year"
    assert by_paper["ev_mismatch_2020"] == "year_mismatch"


def test_default_ceiling_still_falls_back_to_the_module_constant() -> None:
    """Backward compatibility: omitting the argument must behave as before."""
    extractions = {"p1": _extraction("p1")}
    years = {"p1": 2020}
    events = [_event("p1", 2020)]
    accepted, ignored, _ = _validate_events(
        events, extractions_by_paper=extractions, years=years
    )
    assert len(accepted) == 1
    assert ignored == ()


def test_year_below_the_floor_is_still_a_mismatch() -> None:
    """The 1900 floor is untouched by this change."""
    extractions = {"p1": _extraction("p1")}
    years = {"p1": 1850}
    events = [_event("p1", 1850)]
    _, ignored, _ = _validate_events(
        events, extractions_by_paper=extractions, years=years, current_year=2026
    )
    assert ignored[0]["reason"] == "year_mismatch"


# --------------------------------------------------------------------------
# The thread-event horizon must NOT change when the argument is omitted.
# --------------------------------------------------------------------------

def test_thread_horizon_default_is_corpus_based_not_clock_based() -> None:
    """_thread_events has always derived its window from the corpus's own
    latest year.  Resolving None to CURRENT_YEAR upstream would silently widen
    it, so assert the default is corpus-derived."""
    import inspect

    src = inspect.getsource(te._thread_events)
    assert "max(years.values())" in src, (
        "the corpus-derived default must survive: the window is about the "
        "corpus's own horizon, not the wall clock"
    )


def test_normalize_events_with_no_years_at_all_does_not_crash() -> None:
    """The import-time constant is the last-resort fallback and must still work."""
    bundle = normalize_events([], [], [], {}, {}, "snap_empty")
    assert len(bundle.events) == 0
    assert bundle.validation_errors == ()


def test_normalize_events_accepts_an_explicit_year_end_to_end() -> None:
    """Passing current_year must thread all the way through without TypeError."""
    bundle = normalize_events(
        [], [], [], {}, {}, "snap_empty", current_year=2031
    )
    assert len(bundle.events) == 0
