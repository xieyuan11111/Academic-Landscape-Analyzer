"""FR-09 evidence families and source independence (code-side heuristics).

Flags the situations that create false consensus: the same first author recurring
across papers, the same dataset/described sample used by multiple papers, and
review-like papers that add no raw evidence. Persons and datasets are linked with
union-find into evidence families. The signals are heuristic (PDF metadata often
carries only the first author), so they annotate rather than adjudicate.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

DATASET_JACCARD = 0.5

# Separators used by PDF author metadata.  A comma inside a suffix like
# "Jr." is rare enough in this corpus that simple comma splitting wins.
_AUTHOR_LIST_SPLIT_RE = re.compile(
    r"\s*[;,/]\s*|\s+and\s+|\s*&\s*", re.IGNORECASE
)
# Printer/producer metadata glued to a name: "(Thomson Press India)" and a
# trailing date/time stamp like "2000 Apr 26 13:33:04".
_AFFILIATION_RE = re.compile(r"\s*\([^)]*\)\s*")
_TRAILING_STAMP_RE = re.compile(r"\s+\d{2,4}(?:\s+\w+)*\s+\d{1,2}:\d{2}(?::\d{2})?\s*$")


def split_author_list(raw: str) -> list[str]:
    """Split a PDF author metadata string into individual author names.

    PDF ``/Author`` often carries the whole author list as one string:
    "A. X, B. Y, C. Z".  Passing the raw string to ``canonical_person_key``
    collapses all of them into one impossible surname key, which corrupts
    every evidence-family inference downstream.  This function splits on
    comma / semicolon / slash / ``and`` / ``&`` boundaries, trims each entry
    and drops empties.  It is intentionally conservative: it never rewrites
    individual name spellings.
    """
    text = raw.strip()
    if not text:
        return []
    parts = [part.strip() for part in _AUTHOR_LIST_SPLIT_RE.split(text)]
    cleaned: list[str] = []
    for part in parts:
        if not part:
            continue
        part = _AFFILIATION_RE.sub(" ", part).strip()
        part = _TRAILING_STAMP_RE.sub("", part).strip()
        if part:
            cleaned.append(part)
    return cleaned


def _initials(text: str) -> str:
    parts = re.findall(r"[A-Za-z]+", text)
    return " ".join(part[0].lower() for part in parts)


def canonical_person_key(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if "," in text:  # "Zhang, Wei"
        left, right = text.split(",", 1)
        surname = left.split()[-1] if left.split() else ""
        initials = _initials(right)
    else:
        words = text.split()
        surname = words[-1]
        initials = _initials(" ".join(words[:-1]))
    key = f"{surname.lower()} {initials}".strip()
    return re.sub(r"\s+", " ", key)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 2 or token.isdigit()
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass(frozen=True)
class PersonRecord:
    key: str
    display_names: tuple[str, ...]
    paper_ids: tuple[str, ...]


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    label: str
    paper_ids: tuple[str, ...]


def deterministic_team_id(person_keys: Sequence[str]) -> str:
    """Deterministic, order-insensitive team identifier from person keys."""
    digest = hashlib.sha256("|".join(sorted(person_keys)).encode("utf-8")).hexdigest()
    return f"team_{digest[:16]}"


@dataclass(frozen=True)
class TeamRecord:
    team_id: str
    person_keys: tuple[str, ...]
    paper_ids: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class EvidenceFamily:
    family_id: str
    paper_ids: tuple[str, ...]
    shared_person_keys: tuple[str, ...]
    shared_dataset_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceFamilyReport:
    persons: tuple[PersonRecord, ...]
    datasets: tuple[DatasetRecord, ...]
    families: tuple[EvidenceFamily, ...]
    review_like_papers: tuple[str, ...]
    family_index: dict[str, str]
    teams: tuple[TeamRecord, ...] = field(default=())

    def family_of(self, paper_id: str) -> EvidenceFamily:
        return next(f for f in self.families if f.family_id == self.family_index[paper_id])

    def is_independent(self, paper_id: str) -> bool:
        return len(self.family_of(paper_id).paper_ids) == 1


def analyze_evidence_families(
    authors_by_paper: dict[str, list[str]],
    data_samples_by_paper: dict[str, list[str]],
    results_by_paper: dict[str, list[Any]] | None = None,
    dataset_jaccard: float = DATASET_JACCARD,
) -> EvidenceFamilyReport:
    results_by_paper = results_by_paper or {}
    all_papers = sorted(set(authors_by_paper) | set(data_samples_by_paper))
    paper_pos = {paper: index for index, paper in enumerate(all_papers)}

    # persons
    key_papers: dict[str, set[str]] = {}
    key_names: dict[str, set[str]] = {}
    for paper_id, names in authors_by_paper.items():
        for raw in names:
            key = canonical_person_key(raw)
            if not key:
                continue
            key_papers.setdefault(key, set()).add(paper_id)
            key_names.setdefault(key, set()).add(raw.strip())
    persons = tuple(
        PersonRecord(
            key=key,
            display_names=tuple(sorted(key_names[key])),
            paper_ids=tuple(sorted(key_papers[key])),
        )
        for key in sorted(key_papers)
    )

    # dataset clusters over described samples
    sample_records = [
        (paper_id, text)
        for paper_id, samples in data_samples_by_paper.items()
        for text in samples
        if text.strip()
    ]
    sample_token_sets = [_tokens(text) for _, text in sample_records]
    sample_dsu = _DSU(len(sample_records))
    for a in range(len(sample_records)):
        for b in range(a + 1, len(sample_records)):
            if _jaccard(sample_token_sets[a], sample_token_sets[b]) >= dataset_jaccard:
                sample_dsu.union(a, b)
    sample_groups: dict[int, list[int]] = {}
    for index in range(len(sample_records)):
        sample_groups.setdefault(sample_dsu.find(index), []).append(index)

    datasets: list[DatasetRecord] = []
    dataset_member_papers: list[set[str]] = []
    for group_index, member_indices in enumerate(
        sorted(sample_groups.values(), key=lambda g: min(g))
    ):
        descriptions = [sample_records[index][1] for index in member_indices]
        papers = sorted({sample_records[index][0] for index in member_indices})
        label = min(set(descriptions), key=lambda x: (len(x), descriptions.index(x)))
        dataset_id = f"dataset_{group_index:02d}"
        datasets.append(
            DatasetRecord(
                dataset_id=dataset_id, label=label, paper_ids=tuple(papers)
            )
        )
        dataset_member_papers.append(set(papers))

    # teams: co-authorship union-find over canonical person keys (FR-09)
    paper_person_keys: dict[str, tuple[str, ...]] = {
        paper_id: tuple(sorted({canonical_person_key(raw) for raw in names if canonical_person_key(raw)}))
        for paper_id, names in authors_by_paper.items()
    }
    all_keys = sorted(key_papers)
    key_pos = {key: index for index, key in enumerate(all_keys)}
    team_dsu = _DSU(len(all_keys))
    for keys in paper_person_keys.values():
        members = sorted(keys)
        for index in range(1, len(members)):
            team_dsu.union(key_pos[members[0]], key_pos[members[index]])

    key_groups: dict[int, list[int]] = {}
    for index in range(len(all_keys)):
        key_groups.setdefault(team_dsu.find(index), []).append(index)

    teams: list[TeamRecord] = []
    for member_indices in sorted(key_groups.values(), key=lambda g: min(g)):
        person_keys = tuple(all_keys[index] for index in member_indices)
        if len(person_keys) < 2:
            continue
        member_set = set(person_keys)
        team_papers = sorted(
            paper_id
            for paper_id, keys in paper_person_keys.items()
            if len(set(keys) & member_set) >= 2
        )
        teams.append(
            TeamRecord(
                team_id=deterministic_team_id(person_keys),
                person_keys=person_keys,
                paper_ids=tuple(team_papers),
                label=" + ".join(person_keys),
            )
        )

    # paper-level families
    paper_dsu = _DSU(len(all_papers))
    for paper_set in key_papers.values():
        members = sorted(paper_set)
        for index in range(1, len(members)):
            paper_dsu.union(paper_pos[members[0]], paper_pos[members[index]])
    for paper_set in dataset_member_papers:
        members = sorted(paper_set)
        for index in range(1, len(members)):
            paper_dsu.union(paper_pos[members[0]], paper_pos[members[index]])

    paper_groups: dict[int, list[int]] = {}
    for index in range(len(all_papers)):
        paper_groups.setdefault(paper_dsu.find(index), []).append(index)

    families: list[EvidenceFamily] = []
    family_index: dict[str, str] = {}
    for family_number, member_indices in enumerate(
        sorted(paper_groups.values(), key=lambda g: min(g))
    ):
        family_id = f"family_{family_number:02d}"
        member_papers = tuple(all_papers[index] for index in member_indices)
        member_set = set(member_papers)
        for paper in member_papers:
            family_index[paper] = family_id

        shared_person_keys = tuple(
            sorted(
                key
                for key, papers in key_papers.items()
                if len(papers & member_set) >= 2
            )
        )
        shared_dataset_ids = tuple(
            sorted(
                datasets[index].dataset_id
                for index in range(len(datasets))
                if len(dataset_member_papers[index] & member_set) >= 2
            )
        )
        families.append(
            EvidenceFamily(
                family_id=family_id,
                paper_ids=member_papers,
                shared_person_keys=shared_person_keys,
                shared_dataset_ids=shared_dataset_ids,
            )
        )

    review_like = tuple(
        paper_id
        for paper_id in all_papers
        if not data_samples_by_paper.get(paper_id)
        and not results_by_paper.get(paper_id)
    )

    return EvidenceFamilyReport(
        persons=persons,
        datasets=tuple(datasets),
        families=tuple(families),
        review_like_papers=review_like,
        family_index=family_index,
        teams=tuple(teams),
    )
