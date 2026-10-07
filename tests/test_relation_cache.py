"""Contract tests: relation extraction must be resumable per direction.

A MAG-209 run spends hours inside ``extract_paper_relations`` and holds every
finished direction in memory only — a crash loses all of it. These tests pin a
durable per-direction cache: results are written as soon as a direction
finishes, a rerun skips cached directions, and a direction whose inputs changed
is re-extracted rather than served stale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from academic_landscape.direction_naming import ResearchDirection
from academic_landscape.relation_extraction import (
    RelationExtractionResult,
    extract_paper_relations,
)

SNAPSHOT_ID = "snapshot_test"


class CountingClient:
    """Records calls; returns a fixed relation payload."""

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls += 1
        return {"relations": [
            {
                "source_paper_id": "paper_a",
                "target_paper_id": "paper_b",
                "relation_type": "supports",
                "statement": "A supports B",
                "source_evidence_ids": ["paper_a::E1"],
                "target_evidence_ids": ["paper_b::E1"],
                "difference_dimension": None,
                "comparability_note": "",
                "uncertainty": "",
            }
        ]}


class ExplodingClient:
    """Fails loudly if called — proves the cache prevented the call."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise AssertionError("cache should have prevented this LLM call")


def _extraction(paper_id: str, evidence_ids: list[str]) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "snapshot_id": SNAPSHOT_ID,
        "research_question": {"text": f"RQ {paper_id}", "evidence_ids": evidence_ids[:1]},
        "results": [{"text": f"finding {paper_id}", "evidence_ids": evidence_ids[:1]}],
        "claims": [{"claim_id": "C1", "text": f"claim {paper_id}",
                    "evidence_type": "A", "evidence_ids": evidence_ids[:1]}],
        "evidence": [
            {"evidence_id": eid, "artifact_id": f"artifact_{paper_id}",
             "page_number": 1, "quote": f"quote {eid}", "locator": "page 1"}
            for eid in evidence_ids
        ],
        "concepts": [], "theoretical_frameworks": [], "methods": [],
        "data_samples": [], "contexts": [], "limitations": [],
    }


def _direction(direction_id: str, papers: list[str]) -> ResearchDirection:
    return ResearchDirection(
        direction_id=direction_id,
        cluster_id=1,
        label=f"Direction {direction_id}",
        description="desc",
        status="emerging",
        paper_ids=tuple(papers),
        item_ids=tuple(f"item_{p}" for p in papers),
    )


def _extractions() -> dict[str, dict[str, Any]]:
    return {
        "paper_a": _extraction("paper_a", ["paper_a::E1", "paper_a::E2"]),
        "paper_b": _extraction("paper_b", ["paper_b::E1", "paper_b::E2"]),
    }


def _years() -> dict[str, int]:
    return {"paper_a": 2020, "paper_b": 2024}


def test_cache_write_then_skip_on_rerun(tmp_path: Path) -> None:
    cache_path = tmp_path / "relation_cache.json"
    directions = [_direction("dir_1", ["paper_a", "paper_b"])]

    first_client = CountingClient()
    extract_paper_relations(
        first_client, directions, _extractions(), _years(),
        snapshot_id=SNAPSHOT_ID, cache_path=cache_path,
    )
    assert first_client.calls == 1
    assert cache_path.exists(), "finished directions must be written durably"

    # A rerun must not call the model again for the same direction.
    extract_paper_relations(
        ExplodingClient(), directions, _extractions(), _years(),
        snapshot_id=SNAPSHOT_ID, cache_path=cache_path,
    )


def test_cached_results_survive_process_restart(tmp_path: Path) -> None:
    cache_path = tmp_path / "relation_cache.json"
    directions = [_direction("dir_1", ["paper_a", "paper_b"])]
    extract_paper_relations(
        CountingClient(), directions, _extractions(), _years(),
        snapshot_id=SNAPSHOT_ID, cache_path=cache_path,
    )

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "dir_1" in payload, "cache is keyed per direction"
    entry = payload["dir_1"]
    assert entry["result"]["direction_id"] == "dir_1"
    assert entry["result"]["relations"], "relations must round-trip"

    results = extract_paper_relations(
        ExplodingClient(), directions, _extractions(), _years(),
        snapshot_id=SNAPSHOT_ID, cache_path=cache_path,
    )
    assert len(results) == 1
    assert results[0].direction_id == "dir_1"
    assert results[0].relations, "cached result must be rehydrated, not an empty shell"


def test_changed_inputs_are_not_served_from_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "relation_cache.json"
    directions = [_direction("dir_1", ["paper_a", "paper_b"])]
    extract_paper_relations(
        CountingClient(), directions, _extractions(), _years(),
        snapshot_id=SNAPSHOT_ID, cache_path=cache_path,
    )

    changed = _extractions()
    changed["paper_a"] = _extraction("paper_a", ["paper_a::E1"])  # evidence set changed
    client = CountingClient()
    extract_paper_relations(
        client, directions, changed, _years(),
        snapshot_id=SNAPSHOT_ID, cache_path=cache_path,
    )
    assert client.calls == 1, "changed inputs must re-extract, not reuse stale cache"


def test_uncached_directions_still_run(tmp_path: Path) -> None:
    cache_path = tmp_path / "relation_cache.json"
    extractions = _extractions()
    extractions["paper_c"] = _extraction("paper_c", ["paper_c::E1"])
    years = {**_years(), "paper_c": 2022}

    # warm the cache for dir_1 only
    extract_paper_relations(
        CountingClient(), [_direction("dir_1", ["paper_a", "paper_b"])],
        extractions, years, snapshot_id=SNAPSHOT_ID, cache_path=cache_path,
    )

    client = CountingClient()
    results = extract_paper_relations(
        client,
        [_direction("dir_1", ["paper_a", "paper_b"]), _direction("dir_2", ["paper_c", "paper_a"])],
        extractions, years, snapshot_id=SNAPSHOT_ID, cache_path=cache_path,
    )
    assert client.calls == 1, "only the uncached direction may call the model"
    assert [r.direction_id for r in results] == ["dir_1", "dir_2"]


def test_cache_none_behaves_like_today(tmp_path: Path) -> None:
    """Without a cache path the call path is unchanged (single call, no file)."""
    client = CountingClient()
    results = extract_paper_relations(
        client, [_direction("dir_1", ["paper_a", "paper_b"])],
        _extractions(), _years(), snapshot_id=SNAPSHOT_ID,
    )
    assert client.calls == 1
    assert len(results) == 1
    assert isinstance(results[0], RelationExtractionResult)
    assert not list(tmp_path.glob("*.json"))
