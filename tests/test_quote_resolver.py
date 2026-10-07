"""Contract tests for locating a contiguous source span from a model-provided hint.

The resolver is the mechanism that lets the model only name a short phrase and
lets code pick a guaranteed-contiguous substring out of the page text, so the
resulting quote always passes the strict quote_exists check.
"""

from __future__ import annotations

from academic_landscape.models import PageText
from academic_landscape.quote_resolver import resolve_quote


def test_resolves_exact_hint_to_contiguous_span() -> None:
    page = PageText(
        page_number=1,
        text="The insurance hypothesis states that biodiversity insures ecosystems.",
    )
    hint = "insurance hypothesis states that biodiversity"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.page_number == 1
    # The anchor locates the sentence; the returned quote is the whole sentence.
    assert result.quote == "The insurance hypothesis states that biodiversity insures ecosystems."
    assert result.quote in page.text
    assert hint in result.quote
    assert result.score == 1.0


def test_resolves_hint_with_case_and_whitespace_differences() -> None:
    page = PageText(
        page_number=1,
        text="Biodiversity insures ecosystems against declines in functioning.",
    )
    hint = "  BIODIVERSITY   INSURES  ecosystems "

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote in page.text
    assert "Biodiversity insures ecosystems" in result.quote
    assert result.score == 1.0


def test_resolves_partially_rewritten_hint_by_ordered_tokens() -> None:
    page = PageText(
        page_number=1,
        text="Biodiversity insures ecosystems against declines in functioning.",
    )
    # The model swapped one word and dropped another; ordered tokens still locate a span.
    hint = "biodiversity protects ecosystems against declines"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote in page.text
    # "biodiversity ... ecosystems against declines" are all present and ordered.
    assert result.score >= 0.6


def test_returns_none_when_hint_is_absent() -> None:
    page = PageText(page_number=1, text="Only a real sentence is present here.")

    result = resolve_quote([page], "totally absent phrase xyz")

    assert result is None


def test_respects_candidate_page() -> None:
    page_one = PageText(page_number=1, text="First page has no such phrase.")
    page_two = PageText(page_number=2, text="The target phrase lives on page two.")

    result = resolve_quote([page_one, page_two], "target phrase", candidate_page=2)

    assert result is not None
    assert result.page_number == 2
    assert result.quote in page_two.text


def test_candidate_page_out_of_range_falls_back_to_search_all() -> None:
    page_one = PageText(page_number=1, text="The target phrase lives on page one.")

    result = resolve_quote([page_one], "target phrase", candidate_page=99)

    assert result is not None
    assert result.page_number == 1


def test_supports_unicode_tokens() -> None:
    page = PageText(page_number=1, text="城市污水处理厂现状分析")

    result = resolve_quote([page], "污水处理")

    assert result is not None
    # CJK text has no sentence-final punctuation here, so the whole page run is
    # one sentence; the quote expands to it and still contains the anchor.
    assert result.quote == "城市污水处理厂现状分析"
    assert "污水处理" in result.quote
    assert result.quote in page.text


def test_missing_middle_token_never_yields_empty_quote() -> None:
    page = PageText(page_number=1, text="Class " + "middle " * 40 + "defining end")

    result = resolve_quote([page], "defining zzznope Class")

    # Either a real contiguous span is located or the resolver bails; it must never
    # return an empty quote (the old cursor-wrap bug produced '' here).
    assert result is None or (result.quote.strip() != "" and result.quote in page.text)


def test_common_distant_tokens_are_rejected() -> None:
    page = PageText(page_number=1, text="NMO " + "fill " * 40 + "Class " + "fill " * 40 + "I")

    result = resolve_quote([page], "NMO Class I")

    assert result is None


def test_match_total_span_is_bounded() -> None:
    # Each adjacent gap is small (< max_gap) but the total first→last span is huge,
    # so the match must be rejected rather than returning a multi-hundred-word quote.
    tokens = " ".join(f"w{i}" for i in range(100))
    page = PageText(page_number=1, text=tokens)
    hint = "w0 w10 w20 w30 w40 w50"

    result = resolve_quote([page], hint)

    assert result is None