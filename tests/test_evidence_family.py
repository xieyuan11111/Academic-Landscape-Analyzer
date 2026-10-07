"""Tests for FR-09 evidence family and source independence analysis."""

from __future__ import annotations

from academic_landscape.evidence_family import (
    canonical_person_key,
    analyze_evidence_families,
)


# ---------- person key parsing ----------


def test_person_key_given_name_first() -> None:
    assert canonical_person_key("Hoang Phuc Trinh") == "trinh h p"


def test_person_key_initials_format() -> None:
    assert canonical_person_key("Q.H. Zhang") == "zhang q h"


def test_person_key_surname_comma_first() -> None:
    assert canonical_person_key("Zhang, Wei") == "zhang w"


def test_person_key_blank_returns_empty() -> None:
    assert canonical_person_key("   ") == ""


# ---------- author-list splitting (PDF metadata hygiene) ----------


def test_split_author_list_comma_separated_long_list() -> None:
    """A 32-person PDF metadata author string must become 32 names."""
    from academic_landscape.evidence_family import split_author_list

    raw = (
        "Dana A. Opulente, Abigail Leavitt LaBella, Marie-Claire Harrison, "
        "Chris Todd Hittinger"
    )
    names = split_author_list(raw)
    assert names == [
        "Dana A. Opulente",
        "Abigail Leavitt LaBella",
        "Marie-Claire Harrison",
        "Chris Todd Hittinger",
    ]


def test_split_author_list_semicolon_separated() -> None:
    from academic_landscape.evidence_family import split_author_list

    raw = "Anna Banach-Wiśniewska; Filip Gamoń; Aleksandra Ziembińska-Buczyńska;"
    names = split_author_list(raw)
    assert names == [
        "Anna Banach-Wiśniewska",
        "Filip Gamoń",
        "Aleksandra Ziembińska-Buczyńska",
    ]


def test_split_author_list_and_conjunction() -> None:
    from academic_landscape.evidence_family import split_author_list

    raw = "Nur Athirah Yusof, Noor Haza Fazlin Hashim and Izwan Bharudin"
    names = split_author_list(raw)
    assert names == [
        "Nur Athirah Yusof",
        "Noor Haza Fazlin Hashim",
        "Izwan Bharudin",
    ]


def test_split_author_list_single_name_unchanged() -> None:
    from academic_landscape.evidence_family import split_author_list

    assert split_author_list("Rob Knight") == ["Rob Knight"]
    assert split_author_list("") == []
    assert split_author_list("   ") == []


def test_split_author_list_then_key_is_single_person() -> None:
    """The regression: 32-person list must NOT collapse into one fake key."""
    from academic_landscape.evidence_family import split_author_list

    raw = "Dana A. Opulente, Chris Todd Hittinger"
    keys = [canonical_person_key(name) for name in split_author_list(raw)]
    assert keys == ["opulente d a", "hittinger c t"]


def test_split_author_list_slash_separated() -> None:
    """PDF /Author fields sometimes separate names with slashes."""
    from academic_landscape.evidence_family import split_author_list

    raw = "Michel. DuBois/K. A. Gilles/J. K. Hamilton/P. A. Rebers/Fred. Smith"
    names = split_author_list(raw)
    assert names == [
        "Michel. DuBois",
        "K. A. Gilles",
        "J. K. Hamilton",
        "P. A. Rebers",
        "Fred. Smith",
    ]


def test_split_author_strips_print_timestamp_and_affiliation() -> None:
    """Producer metadata glued onto a name must not become part of the name."""
    from academic_landscape.evidence_family import split_author_list

    raw = "B L Ganju (Thomson Press India) 1267 2000 Apr 26 13:33:04"
    assert split_author_list(raw) == ["B L Ganju"]


def test_split_author_name_ending_with_digits_is_not_truncated() -> None:
    """A legitimate name containing a roman-numeral suffix survives."""
    from academic_landscape.evidence_family import split_author_list

    assert split_author_list("John Henry VIII") == ["John Henry VIII"]


# ---------- evidence families ----------


def _authors() -> dict[str, list[str]]:
    return {
        "paper_a": ["Q.H. Zhang"],
        "paper_b": ["Qinghui Zhang"],   # different canonical key -> not linked
        "paper_c": ["Q.H. Zhang"],     # same first author as paper_a
        "paper_d": ["Jun Zuo"],
    }


def _data_samples() -> dict[str, list[str]]:
    return {
        "paper_a": ["swine slurry samples from the farm"],
        "paper_b": ["swine slurry samples from the farm site"],  # links to a
        "paper_c": ["municipal wastewater grab samples"],
        "paper_d": ["marine sediment core samples"],
    }


def test_shared_person_links_family() -> None:
    report = analyze_evidence_families(_authors(), _data_samples())
    family = report.family_of("paper_a")
    assert "paper_c" in family.paper_ids
    assert "paper_c" in report.family_of("paper_a").paper_ids


def test_shared_dataset_links_family() -> None:
    report = analyze_evidence_families(_authors(), _data_samples())
    family_a = report.family_of("paper_a")
    assert "paper_b" in family_a.paper_ids


def test_families_transitively_merged() -> None:
    # paper_a <-person- paper_c, paper_a <-dataset- paper_b: one big family
    report = analyze_evidence_families(_authors(), _data_samples())
    family = report.family_of("paper_a")
    assert set(family.paper_ids) == {"paper_a", "paper_b", "paper_c"}


def test_independent_paper_gets_own_family() -> None:
    report = analyze_evidence_families(_authors(), _data_samples())
    family = report.family_of("paper_d")
    assert family.paper_ids == ("paper_d",)
    assert report.is_independent("paper_d") is True


def test_person_records_listed() -> None:
    report = analyze_evidence_families(_authors(), _data_samples())
    keys = {person.key for person in report.persons}
    assert "zhang q h" in keys
    assert "trinh" not in keys  # no Trinh in fixture


def test_dataset_records_group_papers() -> None:
    report = analyze_evidence_families(_authors(), _data_samples())
    shared = next(
        dataset
        for dataset in report.datasets
        if len(dataset.paper_ids) >= 2 and "swine" in dataset.label
    )
    assert set(shared.paper_ids) == {"paper_a", "paper_b"}


def test_review_like_paper_detected() -> None:
    authors = {"paper_r": ["Someone Else"], "paper_x": ["X Y"]}
    data_samples = {"paper_r": [], "paper_x": ["raw field samples"]}
    has_results = {"paper_r": [], "paper_x": ["finding one"]}
    report = analyze_evidence_families(
        authors, data_samples, results_by_paper=has_results
    )
    assert "paper_r" in report.review_like_papers
    assert "paper_x" not in report.review_like_papers


def test_family_carries_shared_reasons() -> None:
    report = analyze_evidence_families(_authors(), _data_samples())
    family = report.family_of("paper_a")
    assert "zhang q h" in family.shared_person_keys
    assert len(family.shared_dataset_ids) >= 1


# ---------- teams (FR-09 team identity) ----------

import academic_landscape.evidence_family as ev  # noqa: E402


def _team_authors() -> dict[str, list[str]]:
    # Zhang + Li co-author both papers (order reversed on paper_2)
    return {
        "paper_1": ["Q.H. Zhang", "Wei Li"],
        "paper_2": ["Wei Li", "Q.H. Zhang"],
    }


def test_team_two_persons_coauthor_two_papers_merged() -> None:
    report = analyze_evidence_families(_team_authors(), {})
    assert len(report.teams) == 1
    team = report.teams[0]
    assert set(team.person_keys) == {"zhang q h", "li w"}
    assert set(team.paper_ids) == {"paper_1", "paper_2"}


def test_single_author_papers_form_no_team() -> None:
    report = analyze_evidence_families(
        {"paper_x": ["Q.H. Zhang"], "paper_y": ["Wei Li"]}, {}
    )
    assert report.teams == ()


def test_team_id_deterministic_and_order_insensitive() -> None:
    first = ev.deterministic_team_id(["zhang q h", "li w"])
    second = ev.deterministic_team_id(["li w", "zhang q h"])
    assert first == second
    assert first.startswith("team_")
    assert first != ev.deterministic_team_id(["zhang q h", "zuo j"])


def test_persons_without_coauthorship_stay_separate_teams() -> None:
    authors = {
        "paper_1": ["Q.H. Zhang", "Wei Li"],
        "paper_2": ["Jun Zuo", "M. N. Op"],  # disjoint pair -> own team
        "paper_3": ["Solo Person"],          # alone -> no team
    }
    report = analyze_evidence_families(authors, {})
    assert len(report.teams) == 2
    key_sets = [set(team.person_keys) for team in report.teams]
    assert {"zhang q h", "li w"} in key_sets
    assert {"op m n", "zuo j"} in key_sets
    for team in report.teams:
        assert "person s" not in team.person_keys


def test_report_teams_default_empty_backward_compatible() -> None:
    report = ev.EvidenceFamilyReport(
        persons=(),
        datasets=(),
        families=(),
        review_like_papers=(),
        family_index={},
    )
    assert report.teams == ()


def test_team_label_contains_both_person_keys() -> None:
    report = analyze_evidence_families(_team_authors(), {})
    label = report.teams[0].label
    assert label
    assert "zhang q h" in label
    assert "li w" in label
    assert " + " in label


def test_team_merges_transitively_across_papers() -> None:
    # Zhang-Li on paper_1, Li-Zuo on paper_2 -> one team of three
    authors = {
        "paper_1": ["Q.H. Zhang", "Wei Li"],
        "paper_2": ["Wei Li", "Jun Zuo"],
    }
    report = analyze_evidence_families(authors, {})
    assert len(report.teams) == 1
    team = report.teams[0]
    assert set(team.person_keys) == {"zhang q h", "li w", "zuo j"}
    assert set(team.paper_ids) == {"paper_1", "paper_2"}
    assert team.person_keys == tuple(sorted(team.person_keys))
