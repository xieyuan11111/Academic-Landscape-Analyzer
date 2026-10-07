"""Contract tests for hint resolution: locate a phrase, expand to the sentence.

The resolver's job is to let the model name only a short *anchor phrase* and
have code pick a guaranteed-contiguous span out of the real page text. The
original implementation returned exactly the anchor span — two words — which
is a glossary entry, not evidence. A reader cannot see what the author asserted
about it.

These tests pin the expansion contract: the returned quote is still a verbatim
substring of the page (the strict quote_exists gate keeps passing), but it now
covers the whole sentence the anchor sits in, with the anchor preserved for
audit.
"""

from __future__ import annotations

from academic_landscape.models import PageText
from academic_landscape.quote_resolver import resolve_quote


def test_expands_anchor_to_full_sentence() -> None:
    page = PageText(
        page_number=1,
        text=(
            "There is, however, a consistent trend in psychrophilic enzymes: "
            "low levels of conformational stability allow high reaction rates "
            "at low temperatures."
        ),
    )
    hint = "conformational stability"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote == (
        "There is, however, a consistent trend in psychrophilic enzymes: "
        "low levels of conformational stability allow high reaction rates "
        "at low temperatures."
    )
    assert result.quote in page.text
    assert hint in result.quote


def test_anchor_hint_is_preserved_for_audit() -> None:
    page = PageText(page_number=1, text="Redundancy buffers the community.")
    hint = "Redundancy buffers"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.hint == hint


def test_expansion_does_not_cross_sentence_boundary_before_anchor() -> None:
    page = PageText(
        page_number=1,
        text=(
            "Figure 3 reports the mixing regime. Conformational stability is low."
        ),
    )
    hint = "Conformational stability"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote == "Conformational stability is low."
    assert "mixing regime" not in result.quote


def test_expansion_does_not_cross_sentence_boundary_after_anchor() -> None:
    page = PageText(
        page_number=1,
        text=(
            "Functional redundancy declined sharply to 0.48. Reactor performance "
            "then collapsed."
        ),
    )
    hint = "Functional redundancy"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote == "Functional redundancy declined sharply to 0.48."
    assert "Reactor performance" not in result.quote


def test_decimal_point_is_not_a_sentence_boundary() -> None:
    page = PageText(
        page_number=1,
        text="The buffering ratio fell to 0.48 and functional redundancy declined.",
    )
    hint = "functional redundancy"

    result = resolve_quote([page], hint)

    assert result is not None
    assert "0.48" in result.quote
    assert result.quote == page.text


def test_abbreviation_period_is_not_a_sentence_boundary() -> None:
    page = PageText(
        page_number=1,
        text="See Fig. 3 for the layout. Conformational stability is low.",
    )
    hint = "Conformational stability"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote == "Conformational stability is low."
    assert "Fig." not in result.quote


def test_author_initial_period_is_not_a_sentence_boundary() -> None:
    page = PageText(
        page_number=1,
        text="The method follows J. Gregory Caporaso. Conformational stability is low.",
    )
    hint = "Conformational stability"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote == "Conformational stability is low."


def test_chinese_sentence_boundary_is_respected() -> None:
    page = PageText(
        page_number=1,
        text="城市污水处理厂运行稳定。功能冗余能够缓冲群落功能的波动。",
    )
    hint = "功能冗余"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote == "功能冗余能够缓冲群落功能的波动。"


def test_anchor_in_first_sentence_expands_from_page_start() -> None:
    page = PageText(
        page_number=1,
        text="An enemy-free nesting space is created by predator avoidance. Later work followed.",
    )
    hint = "enemy-free nesting space"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote.startswith("An enemy-free nesting space")
    assert result.quote.endswith("avoidance.")


def test_anchor_in_last_sentence_expands_to_page_end() -> None:
    page = PageText(
        page_number=1,
        text="Earlier work preceded. Functional redundancy insures ecosystem function.",
    )
    hint = "Functional redundancy"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote == "Functional redundancy insures ecosystem function."


def test_expansion_is_clamped_to_max_quote_len_without_rejecting() -> None:
    sentence = (
        "Functional redundancy buffers community function against environmental "
        "perturbations across many taxa and many environments in a way that is "
        "robust to species loss. " * 6
    ).strip()
    page = PageText(page_number=1, text=sentence)
    hint = "Functional redundancy"

    result = resolve_quote([page], hint, max_quote_len=120)

    assert result is not None
    assert len(result.quote) <= 120
    assert result.quote in page.text
    assert result.quote.startswith("Functional redundancy")


def test_result_stays_verbatim_substring_after_expansion() -> None:
    page = PageText(
        page_number=1,
        text=(
            "These taxa-function robustness differences are most strongly "
            "associated with differences in overall functional redundancy, "
            "though other aspects of gene distribution also contribute."
        ),
    )
    hint = "functional redundancy"

    result = resolve_quote([page], hint)

    assert result is not None
    assert result.quote in page.text
    assert result.quote.count("functional redundancy") == 1


def test_clamped_window_never_starts_or_ends_mid_word() -> None:
    # A long sentence clamped around an anchor near its end must not begin with
    # a word fragment like "d on the comparison of" — the old bug produced this
    # from a 500-char sentence ending at an anchor.
    head = "Start of the abstract sets the scene with a lot of preamble words. " * 6
    tail = "Based on the comparison of the structural parameters we conclude that global-flexibility mechanism confers cold activity."
    page = PageText(page_number=1, text=head + tail)
    hint = "global-flexibility mechanism confers cold activity"

    result = resolve_quote([page], hint, max_quote_len=120)

    assert result is not None
    assert len(result.quote) <= 120
    assert result.quote in page.text
    assert hint in result.quote
    # The quote must not begin or end inside a word.
    start_in_page = page.text.find(result.quote)
    assert start_in_page == 0 or not page.text[start_in_page - 1].isalnum()
    end_in_page = start_in_page + len(result.quote)
    assert end_in_page == len(page.text) or not page.text[end_in_page].isalnum()


def test_clamped_window_never_ends_on_dangling_hyphen() -> None:
    """A cut at ``N-|cycling`` is mid-word and breaks quote_exists.

    The matching gate strips inter-alnum hyphens (``N-cycling`` -> ``ncycling``)
    but a dangling trailing hyphen survives normalisation, so the normalised
    quote stops matching the normalised page. This is not hypothetical: two
    MAG-209 papers (abstract keyword blocks with no sentence punctuation)
    failed extraction exactly this way at the 500-char clamp boundary.
    """
    from academic_landscape.quote_resolver import _clamp_window
    from academic_landscape.semantic import _quote_exists

    text = "Kinetics N-cycling gene abundance metagenome assembled genomes here"
    cut = text.index("cycling")  # window ends right after "N-"
    anchor_start = text.index("Kinetics")
    anchor_end = anchor_start + len("Kinetics")

    quote = _clamp_window(text, 0, cut, anchor_start, anchor_end, max_len=500)

    assert quote in text
    assert not quote.rstrip().endswith("-")
    assert _quote_exists(text, quote)


def test_clamped_window_never_starts_after_dangling_hyphen() -> None:
    """A cut inside ``N-cycling`` must not begin the quote at ``cycling``."""
    from academic_landscape.quote_resolver import _clamp_window
    from academic_landscape.semantic import _quote_exists

    text = "pretext N-cycling gene abundance delta epsilon"
    cut = text.index("cycling")  # window starts at "c|ycling"
    anchor_start = text.index("gene")
    anchor_end = anchor_start + len("gene")

    quote = _clamp_window(text, cut, len(text), anchor_start, anchor_end, max_len=500)

    assert quote in text
    assert not quote.startswith("cycling")
    assert _quote_exists(text, quote)


def test_window_never_ends_on_line_wrap_hyphen() -> None:
    """A quote ending at ``HCO3-`` before a page line-break breaks quote_exists.

    The match gate deletes ``-`` + newline in the *page* normalisation
    (``HCO3-\n(2)`` -> ``hco3 (2)``) but the quote's trailing ``-`` has no
    newline after it and survives, so the normalised quote is no longer a
    substring. Real failure: MAG-209 ``paper_ab957ad6beab5a7a38690303`` E10,
    a chemical equation on PDF page 5.
    """
    from academic_landscape.quote_resolver import _clamp_window
    from academic_landscape.semantic import _quote_exists

    text = "limestone addition neutralizes H+ as shown:\nCaCO3 + H+ \u2192 Ca2+ + HCO3-\n(2) after which pH rises"
    cut = text.index("\n(2)")  # window ends right after the trailing hyphen
    anchor_start = text.index("HCO3")
    anchor_end = anchor_start + len("HCO3")

    quote = _clamp_window(text, 0, cut, anchor_start, anchor_end, max_len=500)

    assert quote in text
    assert not quote.rstrip().endswith("-")
    assert _quote_exists(text, quote)


def test_window_never_starts_on_line_wrap_hyphen() -> None:
    """The mirror case: a window starting at ``-`` after a wrapped word."""
    from academic_landscape.quote_resolver import _clamp_window
    from academic_landscape.semantic import _quote_exists

    text = "the nitro-\nalkane oxidase pathway was confirmed clearly here"
    cut = text.index("-\n") + 1  # window starts on the dangling hyphen
    anchor_start = text.index("oxidase")
    anchor_end = anchor_start + len("oxidase")

    quote = _clamp_window(text, cut, len(text), anchor_start, anchor_end, max_len=500)

    assert quote in text
    assert not quote.startswith("-")
    assert _quote_exists(text, quote)
