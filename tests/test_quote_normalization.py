from academic_landscape.semantic import _normalise_for_match, _quote_exists


def test_quote_matching_normalizes_pdf_line_break_hyphenation_and_unicode_punctuation():
    page = "next-\ngeneration optoelectronic devices; material’s limits"
    quote = "next-generation optoelectronic devices; material's limits"

    assert _normalise_for_match(page) == _normalise_for_match(quote)
    assert _quote_exists(page, quote) is True


def test_quote_matching_does_not_accept_missing_heading():
    page = "A material parameter describing transport."
    quote = "Hole mobility A material parameter describing transport."

    assert _quote_exists(page, quote) is False
