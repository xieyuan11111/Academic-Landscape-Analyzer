"""Tests for FR-04 per-paper relation candidates (contract §7, TDD RED first).

Covers the 11 required cases: validation gates (reflexive, single-sided evidence,
owner mismatch, unknown evidence, missing difference_dimension, vocabulary),
dedup of (source, target, relation_type), deterministic relation ids, per-direction
bounded calls with scoped prompts, and empty-target-evidence allowance for
not_comparable / unknown.
"""

from __future__ import annotations

from typing import Any

from academic_landscape.direction_naming import ResearchDirection
from academic_landscape.relation_extraction import (
    PaperRelation,
    RelationExtractionResult,
    build_relation_prompt,
    deterministic_relation_id,
    extract_paper_relations,
)

SNAPSHOT_ID = "snapshot_test"


class ScriptedClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append((system_prompt, user_prompt))
        return self.payload


def _extraction(paper_id: str, evidence_ids: list[str]) -> dict[str, Any]:
    """Paper extraction dict with already globally scoped evidence ids.

    ``evidence_ids`` must be the globalized ``paper_id::E1`` form, matching the
    output of ``evidence_scope.globalize_evidence_ids`` (the contract's input).
    """
    return {
        "paper_id": paper_id,
        "snapshot_id": SNAPSHOT_ID,
        "research_question": {
            "text": f"RQ of {paper_id}",
            "evidence_ids": evidence_ids[:1],
        },
        "results": [
            {"text": f"finding one from {paper_id}", "evidence_ids": evidence_ids[:1]},
        ],
        "claims": [
            {"claim_id": "C1", "text": f"claim from {paper_id}",
             "evidence_type": "A", "evidence_ids": evidence_ids[:1]},
        ],
        "evidence": [
            {"evidence_id": eid, "artifact_id": f"artifact_{paper_id}",
             "page_number": 1, "quote": f"quote {eid} text", "locator": "page 1"}
            for eid in evidence_ids
        ],
        "concepts": [],
        "theoretical_frameworks": [],
        "methods": [],
        "data_samples": [],
        "contexts": [],
        "limitations": [],
    }


def _direction(direction_id: str, papers: list[str]) -> ResearchDirection:
    return ResearchDirection(
        direction_id=direction_id,
        cluster_id=int(direction_id[-1]),
        label=f"Direction {direction_id}",
        description="desc",
        status="emerging",
        paper_ids=tuple(papers),
        item_ids=tuple(f"item_{p}" for p in papers),
    )


def _relation(
    source: str,
    target: str,
    relation_type: str,
    *,
    statement: str = "relation statement",
    source_evidence_ids: list[str] | None = None,
    target_evidence_ids: list[str] | None = None,
    difference_dimension: str | None = None,
) -> dict[str, Any]:
    return {
        "source_paper_id": source,
        "target_paper_id": target,
        "relation_type": relation_type,
        "statement": statement,
        "source_evidence_ids": (
            source_evidence_ids if source_evidence_ids is not None
            else [f"{source}::E1"]
        ),
        "target_evidence_ids": (
            target_evidence_ids if target_evidence_ids is not None else []
        ),
        "difference_dimension": difference_dimension,
        "comparability_note": "",
        "uncertainty": "",
        "snapshot_id": SNAPSHOT_ID,
    }


def _context() -> tuple[list[ResearchDirection], dict[str, dict], dict[str, int]]:
    directions = [_direction("dir_1", ["paper_a", "paper_b"])]
    extractions = {
        "paper_a": _extraction("paper_a", ["paper_a::E1", "paper_a::E2"]),
        "paper_b": _extraction("paper_b", ["paper_b::E1", "paper_b::E2"]),
    }
    years = {"paper_a": 2020, "paper_b": 2024}
    return directions, extractions, years


def _run(
    payload: dict[str, Any],
    *,
    directions: list[ResearchDirection] | None = None,
    extractions: dict[str, dict] | None = None,
    years: dict[str, int] | None = None,
    retrieval: dict[str, str] | None = None,
) -> tuple[ScriptedClient, list[RelationExtractionResult]]:
    default_directions, default_extractions, default_years = _context()
    client = ScriptedClient(payload)
    results = extract_paper_relations(
        client=client,
        directions=directions if directions is not None else default_directions,
        extractions=extractions if extractions is not None else default_extractions,
        years=years if years is not None else default_years,
        retrieval_context_by_direction=retrieval,
    )
    return client, results


def _record_text(result: RelationExtractionResult) -> str:
    return " ".join(str(item) for item in result.validation_errors) + " " + " ".join(
        str(item) for item in result.ignored_relations
    )


def _has_code(result: RelationExtractionResult, code: str) -> bool:
    return code in _record_text(result)


def test_valid_supports_relation_passes_with_deterministic_relation_id() -> None:
    payload = {
        "relations": [
            _relation("paper_a", "paper_b", "supports",
                      source_evidence_ids=["paper_a::E1"])
        ]
    }
    _, first_results = _run(payload)
    assert len(first_results) == 1
    first = first_results[0]
    assert isinstance(first, RelationExtractionResult)
    assert first.direction_id == "dir_1"
    assert len(first.relations) == 1
    relation = first.relations[0]
    assert isinstance(relation, PaperRelation)
    assert relation.relation_type == "supports"
    assert relation.source_paper_id == "paper_a"
    assert relation.target_paper_id == "paper_b"
    assert not first.validation_errors
    assert relation.relation_id.startswith("rel_")

    rid_first = deterministic_relation_id(
        SNAPSHOT_ID, "paper_a", "paper_b", "supports", ["paper_a::E1"]
    )
    rid_second = deterministic_relation_id(
        SNAPSHOT_ID, "paper_a", "paper_b", "supports", ["paper_a::E1"]
    )
    assert rid_first == rid_second
    assert rid_first.startswith("rel_")

    _, second_results = _run(payload)
    assert second_results[0].relations[0].relation_id == relation.relation_id


def test_reflexive_relation_is_rejected() -> None:
    payload = {
        "relations": [
            _relation("paper_a", "paper_a", "supports",
                      source_evidence_ids=["paper_a::E1"])
        ]
    }
    _, results = _run(payload)
    result = results[0]
    assert result.relations == ()
    assert result.ignored_relations or result.validation_errors


def test_challenges_relation_without_target_evidence_is_rejected() -> None:
    payload = {
        "relations": [
            _relation("paper_a", "paper_b", "challenges",
                      source_evidence_ids=["paper_a::E1"],
                      target_evidence_ids=[])
        ]
    }
    _, results = _run(payload)
    result = results[0]
    assert result.relations == ()
    assert result.ignored_relations or result.validation_errors


def test_challenges_relation_with_two_sided_evidence_passes() -> None:
    payload = {
        "relations": [
            _relation("paper_a", "paper_b", "challenges",
                      source_evidence_ids=["paper_a::E1"],
                      target_evidence_ids=["paper_b::E1"])
        ]
    }
    _, results = _run(payload)
    result = results[0]
    assert len(result.relations) == 1
    relation = result.relations[0]
    assert relation.relation_type == "challenges"
    assert relation.source_evidence_ids == ["paper_a::E1"]
    assert relation.target_evidence_ids == ["paper_b::E1"]
    assert not result.validation_errors


def test_source_evidence_owned_by_target_paper_reports_owner_mismatch() -> None:
    payload = {
        "relations": [
            _relation("paper_a", "paper_b", "supports",
                      source_evidence_ids=["paper_b::E1"])
        ]
    }
    _, results = _run(payload)
    result = results[0]
    assert result.relations == ()
    # contract §4.7: evidence owned by the other paper -> evidence_owner_mismatch
    assert _has_code(result, "evidence_owner_mismatch")


def test_unknown_evidence_id_rejects_relation_into_ignored_relations() -> None:
    payload = {
        "relations": [
            _relation("paper_a", "paper_b", "supports",
                      source_evidence_ids=["paper_a::E99"])
        ]
    }
    _, results = _run(payload)
    result = results[0]
    assert result.relations == ()
    assert result.ignored_relations
    assert "E99" in " ".join(str(item) for item in result.ignored_relations)


def test_surface_conflict_requires_difference_dimension() -> None:
    payload = {
        "relations": [
            _relation("paper_a", "paper_b", "surface_conflict",
                      source_evidence_ids=["paper_a::E1"],
                      difference_dimension=None)
        ]
    }
    _, results = _run(payload)
    result = results[0]
    assert result.relations == ()
    assert result.ignored_relations or result.validation_errors


def test_relation_type_outside_vocabulary_is_rejected() -> None:
    payload = {
        "relations": [
            _relation("paper_a", "paper_b", "vibes",
                      source_evidence_ids=["paper_a::E1"])
        ]
    }
    _, results = _run(payload)
    result = results[0]
    assert result.relations == ()
    assert result.ignored_relations or result.validation_errors
    # contract §4.6: out-of-vocabulary relation_type -> invalid_relation_type
    assert _has_code(result, "invalid_relation_type")


def test_duplicate_source_target_type_tuple_keeps_first() -> None:
    payload = {
        "relations": [
            _relation("paper_a", "paper_b", "supports",
                      statement="first statement",
                      source_evidence_ids=["paper_a::E1"]),
            _relation("paper_a", "paper_b", "supports",
                      statement="second statement",
                      source_evidence_ids=["paper_a::E2"]),
        ]
    }
    _, results = _run(payload)
    result = results[0]
    assert len(result.relations) == 1
    assert result.relations[0].statement == "first statement"


def test_one_call_per_direction_and_prompt_scoped_to_direction_papers() -> None:
    directions = [
        _direction("dir_1", ["paper_a", "paper_b"]),
        _direction("dir_2", ["paper_c", "paper_d"]),
    ]
    extractions = {
        "paper_a": _extraction("paper_a", ["paper_a::E1"]),
        "paper_b": _extraction("paper_b", ["paper_b::E1"]),
        "paper_c": _extraction("paper_c", ["paper_c::E1"]),
        "paper_d": _extraction("paper_d", ["paper_d::E1"]),
    }
    years = {"paper_a": 2020, "paper_b": 2021, "paper_c": 2022, "paper_d": 2023}
    client, results = _run(
        {"relations": []},
        directions=directions,
        extractions=extractions,
        years=years,
        retrieval={"dir_1": "retrieved hint text"},
    )
    assert [result.direction_id for result in results] == ["dir_1", "dir_2"]
    assert len(client.calls) == 2

    first_system, first_user = client.calls[0]
    second_system, second_user = client.calls[1]
    assert first_system
    assert second_system
    assert "paper_a" in first_user
    assert "paper_b" in first_user
    assert "quote paper_a::E1 text" in first_user
    assert "quote paper_b::E1 text" in first_user
    assert "paper_c" not in first_user
    assert "paper_d" not in first_user
    assert "retrieved hint text" in first_user
    assert "paper_c" in second_user
    assert "paper_d" in second_user
    assert "paper_a" not in second_user
    assert "paper_b" not in second_user

    system, user = build_relation_prompt("Dir label", "PAPER DETAIL BODY", "ctx hint")
    assert isinstance(system, str) and isinstance(user, str)
    assert system and user
    combined = system + "\n" + user
    assert "Dir label" in combined
    assert "PAPER DETAIL BODY" in combined
    assert "RETRIEVED DISCOVERY CONTEXT" in combined

    bare_system, bare_user = build_relation_prompt("Dir label", "PAPER DETAIL BODY")
    assert "RETRIEVED DISCOVERY CONTEXT" not in (bare_system + "\n" + bare_user)


def test_not_comparable_and_unknown_allow_empty_target_evidence() -> None:
    payload = {
        "relations": [
            _relation("paper_a", "paper_b", "not_comparable",
                      source_evidence_ids=["paper_a::E1"],
                      target_evidence_ids=[]),
            _relation("paper_b", "paper_a", "unknown",
                      source_evidence_ids=["paper_b::E1"],
                      target_evidence_ids=[]),
        ]
    }
    _, results = _run(payload)
    result = results[0]
    assert len(result.relations) == 2
    assert {relation.relation_type for relation in result.relations} == {
        "not_comparable", "unknown",
    }
    assert all(relation.target_evidence_ids == [] for relation in result.relations)
    assert not result.validation_errors
