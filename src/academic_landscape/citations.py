"""Author–year in-text citation formatting for report prose.

Report readers follow papers by author and year, not by internal ``paper_<hash>``.
This module turns the PDF author metadata string plus an inferred publication
year into the conventional short form (Knight 2021 / Knight & Caporaso 2021 /
Opulente et al. 2019), falling back to the internal paper_id only when the
metadata genuinely carries nothing usable.

It reuses ``split_author_list`` so the citation never repeats the
"whole author list treated as one person" bug that was fixed in
``evidence_family``.
"""

from __future__ import annotations

import re
from typing import Any

from .evidence_family import split_author_list

__all__ = ["author_year_citation", "author_year_map", "surname_of"]


def surname_of(name: str) -> str:
    """Return the family name of one author string.

    Handles "Family, Given" as well as "Given Middle Family"; drops
    punctuation and returns "" when no alphabetic surname can be found.
    """
    text = name.strip()
    if not text:
        return ""
    if "," in text:
        candidate = text.split(",", 1)[0].strip()
    else:
        candidate = text.split()[-1].strip() if text.split() else ""
    candidate = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿĀ-ž'’-]+", "", candidate)
    return candidate


def author_year_citation(
    authors_raw: str,
    year: int | None,
    paper_id: str,
) -> str:
    """Format one author–year citation, falling back to paper_id as needed.

    Rules (PRD §10.1 keeps these prose-safe):
    * one surname  -> "Knight (2021)"
    * two surnames -> "Knight & Caporaso (2021)"
    * three or more -> "Opulente et al. (2019)"
    * missing year -> "(n.d.)"
    * missing/empty authors -> paper_id in place of the name; if the year is
      also missing, the whole citation degrades to the bare paper_id.
    """
    surnames: list[str] = []
    for name in split_author_list(authors_raw):
        surname = surname_of(name)
        if surname and surname not in surnames:
            surnames.append(surname)

    year_part = str(year) if year else "n.d."
    if not surnames:
        if year:
            return f"{paper_id} ({year_part})"
        return paper_id
    if len(surnames) == 1:
        return f"{surnames[0]} ({year_part})"
    if len(surnames) == 2:
        return f"{surnames[0]} & {surnames[1]} ({year_part})"
    return f"{surnames[0]} et al. ({year_part})"


def author_year_map(
    doc_by_pid: dict[str, Any],
    years: dict[str, int | None],
) -> dict[str, str]:
    """Build ``paper_id -> author-year citation`` for every known paper.

    ``doc_by_pid`` maps paper_id to a DocumentRecord exposing
    ``metadata["author"]``; ``years`` maps paper_id to a publication year
    (may be missing or None).  Unknown papers fall back to their raw id.
    """
    citations: dict[str, str] = {}
    for paper_id, document in doc_by_pid.items():
        metadata = getattr(document, "metadata", {}) or {}
        authors_raw = str(metadata.get("author", "") or "")
        citations[paper_id] = author_year_citation(
            authors_raw,
            years.get(paper_id),
            paper_id,
        )
    for paper_id, year in years.items():
        citations.setdefault(paper_id, author_year_citation("", year, paper_id))
    return citations
