"""Locate a guaranteed-contiguous source span from a model-provided hint.

The LLM is a poor copyist: when asked to reproduce a quote verbatim it often
rewrites, merges or drops words, and those fabricated quotes are correctly
rejected by the strict ``quote_exists`` check. This module flips the contract:
the model only supplies a short *hint* (a phrase it believes occurs in the
paper), and code locates a contiguous span of the real page text around that
hint. The returned ``quote`` is therefore always a verbatim substring of the
page and always passes evidence validation without any relaxation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .models import PageText


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")

_SENTENCE_END = ".!?"
_CJK_SENTENCE_END = "\u3002\uff01\uff1f"

# Trailing periods of these words do not end a sentence. Bounded and testable;
# an out-of-list abbreviation just under-splits, which is the safe direction.
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "fig", "figs", "eq", "eqs",
    "ref", "refs", "no", "nos", "vol", "vols", "vs", "etc", "cf",
    "al", "approx", "st", "jr", "sr", "inc", "ltd", "co", "dept",
    "univ", "proc", "conf", "natl", "int", "sci", "eng", "acc",
    "app", "chap", "ed", "eds", "trans", "pp", "p", "ph",
})


@dataclass(frozen=True)
class ResolvedQuote:
    page_number: int
    quote: str
    score: float
    matched_tokens: int
    hint_tokens: int
    # The anchor phrase the model named. Kept so an auditor can see what the
    # model pointed at, separate from the sentence the code expanded to.
    hint: str = ""


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0), match.start(), match.end())
        for match in _TOKEN_RE.finditer(text)
    ]


_DEFAULT_MAX_GAP = 20
_DEFAULT_MAX_SPAN = 30


def _ordered_match(
    page_tokens: list[tuple[str, int, int]],
    needle: list[str],
    max_gap: int = _DEFAULT_MAX_GAP,
    max_span: int = _DEFAULT_MAX_SPAN,
) -> tuple[int, int, int] | None:
    """Return (matched_count, first_token_index, last_token_index) of the best ordered match.

    Hint tokens are matched as an ordered subsequence of the page tokens. A hint
    token that is absent from the page is skipped so later hint tokens can still
    match; the caller decides via ``score`` whether too much of the hint was
    missing. Consecutive matched tokens must stay within ``max_gap`` page tokens,
    and the whole first→last span within ``max_span``, otherwise a long/common
    hint could match across the entire page.
    """
    if not needle or not page_tokens:
        return None
    folded_page = [token.casefold() for token, _, _ in page_tokens]
    folded_needle = [token.casefold() for token in needle]
    first = folded_needle[0]
    page_len = len(page_tokens)
    best: tuple[int, int, int] | None = None
    for start in range(page_len):
        if folded_page[start] != first:
            continue
        matched = 1
        last = start
        search_from = start + 1
        for target in folded_needle[1:]:
            found = _index_after(folded_page, target, search_from, search_from + max_gap)
            if found == -1:
                continue
            matched += 1
            last = found
            search_from = found + 1
        if last - start > max_span:
            continue
        candidate = (matched, start, last)
        if best is None or (candidate[0], -candidate[2]) > (best[0], -best[2]):
            best = candidate
        if matched == len(folded_needle):
            break
    return best


def _index_after(tokens: list[str], target: str, start: int, limit: int) -> int:
    for index in range(start, min(len(tokens), limit)):
        if tokens[index] == target:
            return index
    return -1


def _is_sentence_end(text: str, index: int) -> bool:
    """Whether the character at ``index`` terminates a sentence.

    A bare period is a boundary only when it is neither a decimal point
    (``0.48``), nor the trailing period of a known abbreviation (``Fig.``,
    ``et al.``), nor an author initial (``J. Gregory Caporaso``).
    """
    ch = text[index]
    if ch in _CJK_SENTENCE_END:
        return True
    if ch in "!?":
        return True
    if ch != ".":
        return False
    before_digit = index > 0 and text[index - 1].isdigit()
    after_digit = index + 1 < len(text) and text[index + 1].isdigit()
    if before_digit and after_digit:
        return False
    start = index
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "'\u2019"):
        start -= 1
    word = text[start:index]
    if word:
        lowered = word.casefold()
        if lowered in _ABBREVIATIONS:
            return False
        if len(word) == 1 and word.isalpha():
            return False
    return True


def _clamp_window(
    text: str,
    left: int,
    right: int,
    anchor_start: int,
    anchor_end: int,
    max_len: int,
) -> str:
    """Trim ``text[left:right]`` to ``max_len`` while keeping the anchor inside.

    Clamping a long sentence at its start would drop an anchor that sits near
    the end, breaking the "quote still contains the anchor" invariant. Grow a
    window around the anchor instead, then trim at word boundaries so the result
    stays a verbatim substring.
    """
    window_start, window_end = left, right
    if window_end - window_start > max_len:
        center = (anchor_start + anchor_end) // 2
        window_start = max(left, center - max_len // 2)
        window_end = min(right, window_start + max_len)
        window_start = max(left, window_end - max_len)
    # Snap both ends to word boundaries: a quote that starts or ends mid-word
    # reads as corrupted evidence ("d on the comparison of") and a dangling
    # hyphen breaks the normalised quote_exists match (``N-`` vs ``N-cycling``
    # -> ``ncycling``). A cut is mid-word when the character across the cut is
    # alphanumeric OR hyphen, since both bind into the same token. Snapping
    # never crosses the anchor, so the anchor survives and the result stays a
    # verbatim substring.
    def _binds_into_word(ch: str) -> bool:
        return ch.isalnum() or ch in "-\u2010\u2011\u2012\u2013\u2014"

    if (
        window_start > 0
        and window_start < len(text)
        and _binds_into_word(text[window_start])
        and window_start > 0
        and _binds_into_word(text[window_start - 1])
        and window_start < anchor_start
    ):
        while (
            window_start < min(window_end, anchor_start)
            and _binds_into_word(text[window_start])
        ):
            window_start += 1
    if (
        window_end < len(text)
        and _binds_into_word(text[window_end - 1])
        and _binds_into_word(text[window_end])
        and window_end > anchor_end
    ):
        while (
            window_end > max(window_start, anchor_end)
            and _binds_into_word(text[window_end - 1])
        ):
            window_end -= 1
    quote = text[window_start:window_end].strip()
    if not quote:
        return ""
    # A hyphen at a cut is always wrong: either mid-word (``N-`` of
    # ``N-cycling``) or a line-wrap hyphen (``HCO3-`` before a newline), and
    # the match gate normalises those differently in quote vs page, so the
    # normalised quote would stop matching. Trim it; the quote stays verbatim.
    # Trim from the *stripped* quote: the page can carry trailing spaces after
    # the hyphen, so checking text[window_end-1] alone would see those spaces.
    quote = quote.rstrip("-").lstrip("-")
    return quote


def _expand_to_sentence(text: str, start: int, end: int) -> tuple[int, int]:
    """Grow ``[start, end)`` outward to the enclosing sentence boundaries.

    The anchor phrase is often a two-word glossary entry; the sentence it sits
    in is the actual assertion. Both bounds stay inside ``text``, so the result
    is still a verbatim substring and the strict ``quote_exists`` gate passes.
    """
    left = 0
    index = start - 1
    while index >= 0:
        if _is_sentence_end(text, index):
            left = index + 1
            break
        index -= 1
    right = len(text)
    index = max(end, start)
    while index < len(text):
        if _is_sentence_end(text, index):
            right = index + 1
            break
        index += 1
    return left, right


def resolve_quote(
    pages: Sequence[PageText],
    hint: str,
    candidate_page: int | None = None,
    min_score: float = 0.6,
    max_quote_len: int = 500,
) -> ResolvedQuote | None:
    """Resolve a hint into a contiguous source span on one of ``pages``.

    ``candidate_page`` narrows the search to one page when valid; otherwise all
    pages are searched. A result is returned only when the fraction of hint
    tokens found in order reaches ``min_score`` and the resulting span stays
    within ``max_quote_len`` characters.
    """
    if not hint or not hint.strip():
        return None
    needle = [token for token, _, _ in _tokenize(hint)]
    if not needle:
        return None

    candidates: list[tuple[int, PageText]] = []
    if candidate_page is not None:
        page_by_number = {page.page_number: page for page in pages}
        page = page_by_number.get(candidate_page)
        if page is not None:
            candidates = [(candidate_page, page)]
    if not candidates:
        candidates = [(page.page_number, page) for page in pages]

    best: tuple[ResolvedQuote, int] | None = None
    for page_number, page in candidates:
        page_tokens = _tokenize(page.text)
        match = _ordered_match(page_tokens, needle)
        if match is None:
            continue
        matched, first_index, last_index = match
        score = matched / len(needle)
        if score < min_score:
            continue
        anchor_start = page_tokens[first_index][1]
        anchor_end = page_tokens[last_index][2]
        # The anchor is what the model pointed at; the sentence is the assertion.
        # Expand to the enclosing sentence so the quote can stand on its own.
        start, end = _expand_to_sentence(page.text, anchor_start, anchor_end)
        quote = _clamp_window(
            page.text, start, end, anchor_start, anchor_end, max_quote_len
        )
        if not quote:
            continue
        resolved = ResolvedQuote(
            page_number=page_number,
            quote=quote,
            score=score,
            matched_tokens=matched,
            hint_tokens=len(needle),
            hint=hint.strip(),
        )
        if best is None or resolved.score > best[0].score:
            best = (resolved, matched)

    return best[0] if best is not None else None