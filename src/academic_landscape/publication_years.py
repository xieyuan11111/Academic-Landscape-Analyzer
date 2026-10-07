"""Infer a paper's publication year from robust document-derived signals.

Priority: explicit year in filename -> Elsevier PII year code -> copyright line
on early pages -> repeated year in early-page text -> predated year when the
current/download year contaminates. Always returns a corpus-scoped guess with
its source; a value of 0 means no signal was found.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .models import DocumentRecord


@dataclass(frozen=True)
class YearGuess:
    year: int
    source: str


def _page_text(document: DocumentRecord, max_pages: int = 2) -> str:
    return "\n".join(
        page.text for page in document.pages[:max_pages]
    )


def infer_publication_year(
    document: DocumentRecord,
    current_year: int = 2026,
) -> YearGuess:
    filename = document.filename

    explicit = [
        int(match)
        for match in re.findall(r"(?:19|20)\d{2}", filename)
        if 1990 <= int(match) <= current_year
    ]
    if explicit:
        return YearGuess(min(explicit), "filename")

    pii_match = re.search(r"S\d{8}(\d{2})\d{6}", filename)
    if pii_match:
        digits = int(pii_match.group(1))
        # 1990s PIIs use codes 90-99; 2000+ use 00-89
        year = 1900 + digits if digits >= 90 else 2000 + digits
        return YearGuess(year, "pii")

    text = _page_text(document)

    copyright_match = re.search(
        r"(?:©|\(c\))[^\d]{0,12}((?:19|20)\d{2})"
        r"|((?:19|20)\d{2})[^\d]{0,3}©",
        text,
        flags=re.IGNORECASE,
    )
    if copyright_match:
        year = int(copyright_match.group(1) or copyright_match.group(2))
        if 1990 <= year <= current_year:
            return YearGuess(year, "copyright")

    counter = Counter(
        int(match)
        for match in re.findall(r"(?:19|20)\d{2}", text)
        if 1990 <= int(match) <= current_year
    )
    if counter:
        year, frequency = counter.most_common(1)[0]
        if frequency >= 2:
            return YearGuess(year, "text_frequency")
        years = sorted(counter)
        if years[-1] >= current_year - 1 and len(years) > 1:
            return YearGuess(years[-2], "predated_text")
        return YearGuess(years[-1], "text_max")

    return YearGuess(0, "unknown")
