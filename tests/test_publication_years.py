"""Tests for publication year inference heuristics."""

from __future__ import annotations

from academic_landscape.models import DocumentRecord, PageText
from academic_landscape.publication_years import infer_publication_year


def _document(
    filename: str,
    pages: tuple[PageText, ...] = (),
    metadata: dict[str, str] | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        artifact_id="artifact_x",
        paper_id="paper_x",
        relative_path=filename,
        filename=filename,
        sha256="sha",
        size_bytes=1,
        status="parsed",
        metadata=metadata or {},
        pages=pages,
    )


def test_year_from_filename() -> None:
    guess = infer_publication_year(
        _document("yachi-loreau-1999-biodiversity.pdf")
    )
    assert guess.year == 1999
    assert guess.source == "filename"


def test_year_from_elsevier_pii() -> None:
    guess = infer_publication_year(
        _document("1-s2.0-S0043135426001168-main.pdf")
    )
    assert guess.year == 2026
    assert guess.source == "pii"


def test_pii_two_digit_code_99_is_1999() -> None:
    # 1990s PIIs encode years as 90-99, not 2090-2099
    guess = infer_publication_year(
        _document("1-s2.0-S0043135499000226-main.pdf")
    )
    assert guess.year == 1999
    assert guess.source == "pii"


def test_year_from_copyright_line() -> None:
    pages = (PageText(1, "© 2014 by The American Society. Some other text."),)
    guess = infer_publication_year(_document("article.pdf", pages=pages))
    assert guess.year == 2014
    assert guess.source == "copyright"


def test_copyright_year_before_symbol() -> None:
    pages = (PageText(1, "2015 © The Authors, some rights reserved"),)
    guess = infer_publication_year(_document("article.pdf", pages=pages))
    assert guess.year == 2015


def test_year_from_repeated_text() -> None:
    text = "Header 2013\n\nbody\n\nfooter 2013 again"
    guess = infer_publication_year(
        _document("article.pdf", pages=(PageText(1, text),))
    )
    assert guess.year == 2013
    assert guess.source == "text_frequency"


def test_download_year_contamination_uses_predated_year() -> None:
    text = "old study from 2015\n\ndownloaded in 2026"
    guess = infer_publication_year(
        _document("article.pdf", pages=(PageText(1, text),))
    )
    assert guess.year == 2015


def test_no_signal_returns_zero() -> None:
    guess = infer_publication_year(
        _document("article.pdf", pages=(PageText(1, "no year here"),))
    )
    assert guess.year == 0
    assert guess.source == "unknown"
