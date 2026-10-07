"""Tests for cluster-local cross-paper comparison."""

from __future__ import annotations

from typing import Any

from academic_landscape.cross_paper import (
    compare_directions,
)
from academic_landscape.direction_naming import ResearchDirection

SNAPSHOT_ID = "snapshot_test"


class ScriptedClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append((system_prompt, user_prompt))
        return self.payload


def _extraction(paper_id: str, evidence_ids: list[str]) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "research_question": {"text": f"RQ of {paper_id}", "evidence_ids": []},
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


def _happy_payload() -> dict[str, Any]:
    return {
        "commonalities": [
            {"statement": "Both study nitrogen removal",
             "evidence": ["paper_a::E1", "paper_b::E1"]}
        ],
        "differences": [
            {"dimension": "method", "statement": "different methods",
             "evidence": ["paper_a::E2"]}
        ],
        "controversies": [
            {"statement": "mechanism X versus Y",
             "positions": [
                 {"paper_id": "paper_a", "stance": "mechanism X",
                  "evidence": ["paper_a::E3"]},
                 {"paper_id": "paper_b", "stance": "mechanism Y",
                  "evidence": ["paper_b::E3"]},
             ]}
        ],
    }


def _context() -> tuple[list[ResearchDirection], dict[str, dict], dict[str, int]]:
    directions = [_direction("dir_1", ["paper_a", "paper_b"])]
    extractions = {
        "paper_a": _extraction("paper_a", ["E1", "E2", "E3"]),
        "paper_b": _extraction("paper_b", ["E1", "E3"]),
    }
    years = {"paper_a": 2020, "paper_b": 2024}
    return directions, extractions, years


def test_happy_path_produces_all_sections() -> None:
    directions, extractions, years = _context()
    client = ScriptedClient(_happy_payload())
    comparisons = compare_directions(directions, extractions, years, client)
    assert len(comparisons) == 1
    result = comparisons[0]
    assert result.direction_id == "dir_1"
    assert len(result.commonalities) == 1
    assert len(result.differences) == 1
    assert len(result.controversies) == 1
    assert not result.validation_errors


def test_one_call_per_direction() -> None:
    directions, extractions, years = _context()
    client = ScriptedClient(_happy_payload())
    compare_directions(directions, extractions, years, client)
    assert len(client.calls) == 1


def test_unknown_evidence_is_reported() -> None:
    directions, extractions, years = _context()
    payload = _happy_payload()
    payload["commonalities"][0]["evidence"].append("paper_a::E99")
    client = ScriptedClient(payload)
    result = compare_directions(directions, extractions, years, client)[0]
    assert any("E99" in error for error in result.validation_errors)


def test_controversy_positions_must_be_different_papers() -> None:
    directions, extractions, years = _context()
    payload = _happy_payload()
    payload["controversies"][0]["positions"][1]["paper_id"] = "paper_a"
    payload["controversies"][0]["positions"][1]["evidence"] = ["paper_a::E2"]
    client = ScriptedClient(payload)
    result = compare_directions(directions, extractions, years, client)[0]
    assert result.controversies == ()
    assert len(result.within_paper_tensions) == 1
    tension = result.within_paper_tensions[0]
    assert {position["paper_id"] for position in tension["positions"]} == {"paper_a"}


def test_position_citing_evidence_owned_by_other_paper_is_error() -> None:
    directions, extractions, years = _context()
    payload = _happy_payload()
    payload["controversies"][0]["positions"][0]["evidence"] = ["paper_b::E1"]
    client = ScriptedClient(payload)
    result = compare_directions(directions, extractions, years, client)[0]
    assert any("not owned" in error for error in result.validation_errors)


def test_invalid_difference_dimension_is_error() -> None:
    directions, extractions, years = _context()
    payload = _happy_payload()
    payload["differences"][0]["dimension"] = "vibes"
    client = ScriptedClient(payload)
    result = compare_directions(directions, extractions, years, client)[0]
    assert any("dimension" in error for error in result.validation_errors)


def test_empty_sections_are_valid() -> None:
    directions, extractions, years = _context()
    client = ScriptedClient(
        {"commonalities": [], "differences": [], "controversies": []}
    )
    result = compare_directions(directions, extractions, years, client)[0]
    assert result.commonalities == ()
    assert result.differences == ()
    assert result.controversies == ()
    assert not result.validation_errors


def test_each_call_sees_only_its_own_direction_papers() -> None:
    directions = [
        _direction("dir_1", ["paper_a"]),
        _direction("dir_2", ["paper_c"]),
    ]
    extractions = {
        "paper_a": _extraction("paper_a", ["E1"]),
        "paper_c": _extraction("paper_c", ["E1"]),
    }
    years = {"paper_a": 2020, "paper_c": 2021}
    client = ScriptedClient(
        {"commonalities": [], "differences": [], "controversies": []}
    )
    compare_directions(directions, extractions, years, client)
    assert len(client.calls) == 2
    first_call_user = client.calls[0][1]
    assert "paper_a" in first_call_user
    assert "paper_c" not in first_call_user
