"""SourceRef metadata completeness and author–year citation formatting.

Covers the two report-quality gaps called out against r6:

1. ``SourceRef`` was built with ``title``/``authors`` left blank, and the
   source card carried no evidence timestamp — PRD FR-10 requires "论文标题、
   作者、年份与唯一标识" plus "时间属性".
2. Report prose cited papers as raw ``paper_<hash>`` strings instead of the
   author–year form a reader can follow.
"""
from __future__ import annotations

from academic_landscape.citations import author_year_citation
from academic_landscape.source_cards import build_source_refs


class _Document:
    """Minimal DocumentRecord stand-in exposing the metadata surface."""

    def __init__(self, paper_id: str, title: str = "", author: str = "") -> None:
        self.paper_id = paper_id
        self.artifact_id = f"artifact_{paper_id}"
        self.metadata = {"title": title, "author": author}


_EXTRACTION = {
    "paper_id": "paper_a",
    "evidence": [
        {
            "evidence_id": "E1",
            "artifact_id": "artifact_paper_a",
            "page_number": 3,
            "quote": "exact quote one",
            "locator": "Results",
            "evidence_type": "A",
        },
        {
            "evidence_id": "E2",
            "artifact_id": "artifact_paper_a",
            "page_number": 7,
            "quote": "exact quote two",
            "locator": "",
            "evidence_type": "B",
        },
    ],
}


# ---------------------------------------------------------------------------
# build_source_refs — metadata completeness
# ---------------------------------------------------------------------------


def test_source_ref_carries_title_authors_year_and_page() -> None:
    doc = _Document(
        "paper_a",
        title="Cold Adaptation in Yeast",
        author="Dana A. Opulente, Chris Todd Hittinger",
    )
    refs = build_source_refs(
        paper_id="paper_a",
        evidence_ids=["paper_a::E1"],
        extraction=_EXTRACTION,
        document=doc,
        year=2019,
    )
    assert len(refs) == 1
    ref = refs[0]
    assert ref.title == "Cold Adaptation in Yeast"
    assert ref.authors == "Dana A. Opulente, Chris Todd Hittinger"
    assert ref.year == 2019
    assert ref.page_number == 3
    assert ref.locator == "Results"
    assert ref.quote == "exact quote one"
    assert ref.evidence_id == "paper_a::E1"


def test_source_ref_derives_page_locator_when_extraction_has_none() -> None:
    doc = _Document("paper_a", title="T", author="A B")
    refs = build_source_refs(
        paper_id="paper_a",
        evidence_ids=["paper_a::E2"],
        extraction=_EXTRACTION,
        document=doc,
        year=None,
    )
    assert refs[0].locator == "page 7"
    assert refs[0].page_number == 7


def test_source_ref_carries_snapshot_verification_time() -> None:
    doc = _Document("paper_a", title="T", author="A B")
    refs = build_source_refs(
        paper_id="paper_a",
        evidence_ids=["paper_a::E1"],
        extraction=_EXTRACTION,
        document=doc,
        year=2019,
        verified_at="2026-09-26T00:00:00+00:00",
    )
    assert refs[0].verified_at == "2026-09-26T00:00:00+00:00"


def test_source_ref_tolerates_missing_document_metadata() -> None:
    doc = _Document("paper_a")
    refs = build_source_refs(
        paper_id="paper_a",
        evidence_ids=["paper_a::E1"],
        extraction=_EXTRACTION,
        document=doc,
        year=None,
    )
    assert refs[0].title == ""
    assert refs[0].authors == ""
    assert refs[0].year is None


def test_source_ref_accepts_document_without_metadata_attribute() -> None:
    class _Bare:
        paper_id = "paper_a"

    refs = build_source_refs(
        paper_id="paper_a",
        evidence_ids=["paper_a::E1"],
        extraction=_EXTRACTION,
        document=_Bare(),
        year=None,
    )
    assert refs[0].title == ""
    assert refs[0].authors == ""


# ---------------------------------------------------------------------------
# author_year_citation — readable in-text references
# ---------------------------------------------------------------------------


def test_citation_single_author() -> None:
    assert author_year_citation("Rob Knight", 2021, "paper_x") == "Knight (2021)"


def test_citation_two_authors() -> None:
    assert (
        author_year_citation("Rob Knight, J. Gregory Caporaso", 2021, "paper_x")
        == "Knight & Caporaso (2021)"
    )


def test_citation_three_or_more_uses_et_al() -> None:
    raw = "Dana A. Opulente, Abigail Leavitt LaBella, Chris Todd Hittinger"
    assert author_year_citation(raw, 2019, "paper_x") == "Opulente et al. (2019)"


def test_citation_semicolon_separated_list() -> None:
    raw = "Anna Banach-Wiśniewska; Filip Gamoń; Aleksandra Ziembińska-Buczyńska"
    assert (
        author_year_citation(raw, 2020, "paper_x")
        == "Banach-Wiśniewska et al. (2020)"
    )


def test_citation_falls_back_to_paper_id_without_year() -> None:
    assert author_year_citation("Rob Knight", None, "paper_x") == "Knight (n.d.)"


def test_citation_falls_back_to_paper_id_without_authors() -> None:
    assert author_year_citation("", 2021, "paper_x") == "paper_x (2021)"


def test_citation_falls_back_to_paper_id_when_nothing_known() -> None:
    assert author_year_citation("", None, "paper_x") == "paper_x"


def test_citation_is_stable_for_same_input() -> None:
    raw = "Dana A. Opulente, Chris Todd Hittinger"
    assert author_year_citation(raw, 2019, "p") == author_year_citation(raw, 2019, "p")
