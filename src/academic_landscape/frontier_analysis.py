"""FR-08 popularity, innovation and frontier classification (code-side scoring).

The three notions are scored separately and never merged into one hot list:
- popularity: share/output of papers in the recent year window;
- innovation: how many concept threads are newly born recently;
- frontier: a propagated, still-growing recent thread that is carried by papers
  from DIFFERENT evidence families (cross-team). Single-team growth stays a
  candidate and a single-paper mention is never a frontier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .stage_reconstruction import ConceptThread


@dataclass(frozen=True)
class PopularityScore:
    label: str
    recent_count: int
    total: int
    recent_share: float


def score_popularity(
    years: Sequence[int],
    current_year: int,
    window: int = 3,
    min_recent: int = 3,
    hot_share: float = 0.5,
) -> PopularityScore:
    cutoff = current_year - window + 1
    recent = [year for year in years if year >= cutoff]
    total = len(years)
    share = len(recent) / total if total else 0.0
    if len(recent) >= min_recent and share >= hot_share:
        label = "hot"
    elif total and not recent:
        label = "cooling"
    else:
        label = "normal"
    return PopularityScore(
        label=label,
        recent_count=len(recent),
        total=total,
        recent_share=round(share, 3),
    )


@dataclass(frozen=True)
class InnovationScore:
    label: str
    recent_threads: int
    total: int


def score_innovation(
    threads: Sequence[ConceptThread],
    current_year: int,
    window: int = 3,
    min_recent: int = 3,
    min_share: float = 0.4,
) -> InnovationScore:
    cutoff = current_year - window + 1
    recent = [thread for thread in threads if thread.earliest_year >= cutoff]
    total = len(threads)
    share = len(recent) / total if total else 0.0
    label = (
        "innovative"
        if len(recent) >= min_recent and share >= min_share
        else "incremental"
    )
    return InnovationScore(
        label=label, recent_threads=len(recent), total=total
    )


_TIER_ORDER = {"established": 1, "candidate": 2, "emerging_frontier": 3}


def classify_thread_frontier(
    thread: ConceptThread,
    family_index: dict[str, str],
    current_year: int,
    window: int = 3,
) -> str:
    if len(thread.paper_ids) < 2:
        return "candidate"
    cutoff = current_year - window + 1
    if not thread.years or thread.years[-1] < cutoff:
        return "established"
    families = {family_index.get(paper_id) for paper_id in thread.paper_ids}
    cross_team = len(families) >= 2
    return "emerging_frontier" if cross_team else "candidate"


# FR-08 requires every candidate to expose the following fields so a later
# LLM pass can fill the qualitative ones without reshaping the schema:
#   type/category             -> category (frontier_formed|frontier_candidate|innovation|popularity)
#   what is new               -> novelty_note            (LLM-filled, default "")
#   earliest corpus record    -> earliest_corpus_year    (thread.earliest_year)
#   recent growth/spread      -> recent_growth           (expanding|stable|declining, derived)
#   multiple teams?           -> cross_team              (evidence-family judgement)
#   relation to mainstream    -> relation_to_mainstream  (LLM-filled, default "")
#   current maturity          -> maturity                (formed|emerging|early, derived)
#   limits & counter-evidence -> limitations_and_counter (LLM-filled, default "")
# A single recent paper is never automatically a "frontier".
FR08_CANDIDATE_FIELDS = (
    "thread",
    "tier",
    "category",
    "novelty_note",
    "earliest_corpus_year",
    "recent_growth",
    "cross_team",
    "relation_to_mainstream",
    "maturity",
    "limitations_and_counter",
    "direction_ids",
    "evidence",
)


@dataclass(frozen=True)
class FrontierCandidateItem:
    thread: str
    tier: str
    category: str
    novelty_note: str
    earliest_corpus_year: int | None
    recent_growth: str
    cross_team: bool
    relation_to_mainstream: str
    maturity: str
    limitations_and_counter: str
    direction_ids: tuple[str, ...]
    evidence: tuple[str, ...]


def _derive_recent_growth(thread: ConceptThread, cutoff: int) -> str:
    recent_years = {year for year in thread.years if year >= cutoff}
    if len(recent_years) >= 2:
        return "expanding"
    return "stable"


def _derive_category(tier: str, maturity: str, earliest_year: int, cutoff: int) -> str:
    if tier == "emerging_frontier":
        return "frontier_formed" if maturity == "formed" else "frontier_candidate"
    if tier == "candidate" and earliest_year >= cutoff:
        return "innovation"
    return "popularity"


def _derive_maturity(tier: str, cross_team: bool) -> str:
    if tier == "emerging_frontier":
        return "formed" if cross_team else "emerging"
    return "early"


@dataclass(frozen=True)
class FrontierReport:
    direction_signals: dict[str, dict[str, str]]
    frontier_threads: tuple[dict[str, str], ...]
    candidates: tuple[FrontierCandidateItem, ...] = ()


def build_frontier_report(
    direction_papers: dict[str, Sequence[str]],
    years_by_id: dict[str, int],
    threads_by_direction: dict[str, Sequence[ConceptThread]],
    family_index: dict[str, str],
    current_year: int,
) -> FrontierReport:
    direction_signals: dict[str, dict[str, str]] = {}
    frontier_threads: list[dict[str, str]] = []
    # thread name -> (item template kwargs, direction_ids set) — items are
    # materialized at the end so direction_ids is complete and frozen for real
    candidate_drafts: dict[str, dict] = {}
    candidate_directions: dict[str, set[str]] = {}

    for direction_id, papers in direction_papers.items():
        cutoff = current_year - 3 + 1
        years = [
            years_by_id[paper_id]
            for paper_id in papers
            if years_by_id.get(paper_id) is not None
        ]
        popularity = score_popularity(years, current_year)

        threads = threads_by_direction.get(direction_id, [])
        innovation = score_innovation(threads, current_year)

        best_tier = "established"
        for thread in threads:
            tier = classify_thread_frontier(thread, family_index, current_year)
            # a single-paper mention born outside the recent window never
            # propagated: it is a stale mention, not a frontier candidate
            if (
                tier == "candidate"
                and len(thread.paper_ids) < 2
                and thread.earliest_year < cutoff
            ):
                continue
            if _TIER_ORDER[tier] > _TIER_ORDER[best_tier]:
                best_tier = tier
            if tier != "established":
                frontier_threads.append(
                    {
                        "direction_id": direction_id,
                        "thread": thread.canonical_name,
                        "tier": tier,
                    }
                )
                families = {
                    family_index.get(paper_id) for paper_id in thread.paper_ids
                }
                cross_team = len(families) >= 2
                maturity = _derive_maturity(tier, cross_team)
                name = thread.canonical_name
                if name not in candidate_directions:
                    candidate_directions[name] = set()
                    candidate_drafts[name] = {
                        "thread": name,
                        "tier": tier,
                        "category": _derive_category(
                            tier, maturity, thread.earliest_year, cutoff
                        ),
                        "novelty_note": "",
                        "earliest_corpus_year": thread.earliest_year,
                        "recent_growth": _derive_recent_growth(thread, cutoff),
                        "cross_team": cross_team,
                        "relation_to_mainstream": "",
                        "maturity": maturity,
                        "limitations_and_counter": "",
                        "evidence": tuple(thread.evidence_ids),
                    }
                candidate_directions[name].add(direction_id)

        direction_signals[direction_id] = {
            "popularity": popularity.label,
            "innovation": innovation.label,
            "frontier": best_tier,
        }

    candidates: list[FrontierCandidateItem] = []
    for name, draft in candidate_drafts.items():
        candidates.append(
            FrontierCandidateItem(
                direction_ids=tuple(sorted(candidate_directions[name])),
                **draft,
            )
        )

    return FrontierReport(
        direction_signals=direction_signals,
        frontier_threads=tuple(frontier_threads),
        candidates=tuple(candidates),
    )
