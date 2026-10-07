"""Tests for cross-direction frontier signal merging."""

from __future__ import annotations

import numpy as np

from academic_landscape.landscape_merge import merge_frontier_signals


def _signal(
    direction_id: str,
    name: str,
    tier: str,
    evidence: list[str] | None = None,
) -> dict:
    payload = {"direction_id": direction_id, "thread": name, "tier": tier}
    if evidence is not None:
        payload["evidence"] = evidence
    return payload


def test_same_thread_name_across_directions_merges() -> None:
    signals = [
        _signal("dir_1", "HNAD", "emerging_frontier", ["paper_a::E1"]),
        _signal("dir_2", "HNAD", "candidate", ["paper_c::E1"]),
    ]
    merged = merge_frontier_signals(signals)
    assert len(merged) == 1
    result = merged[0]
    assert set(result.direction_ids) == {"dir_1", "dir_2"}
    assert result.repeat_count == 2


def test_merged_tier_takes_strongest() -> None:
    signals = [
        _signal("dir_1", "HNAD", "candidate"),
        _signal("dir_2", "HNAD", "emerging_frontier"),
    ]
    assert merge_frontier_signals(signals)[0].tier == "emerging_frontier"


def test_evidence_union_on_merge() -> None:
    signals = [
        _signal("dir_1", "HNAD", "emerging_frontier", ["paper_a::E1"]),
        _signal("dir_2", "HNAD", "emerging_frontier", ["paper_c::E2"]),
    ]
    result = merge_frontier_signals(signals)[0]
    assert set(result.evidence) == {"paper_a::E1", "paper_c::E2"}


def test_different_threads_stay_separate() -> None:
    signals = [
        _signal("dir_1", "HNAD", "emerging_frontier"),
        _signal("dir_2", "3WDOM photochemistry", "candidate"),
    ]
    merged = merge_frontier_signals(signals)
    assert len(merged) == 2


def test_semantically_similar_names_merge_with_vectors() -> None:
    signals = [
        _signal("dir_1", "HNAD", "emerging_frontier", ["paper_a::E1"]),
        _signal("dir_2", "Heterotrophic nitrification aerobic denitrification",
                "candidate", ["paper_b::E1"]),
    ]
    name_vectors = {
        "dir_1::HNAD": np.asarray([1.0, 0.0]),
        "dir_2::Heterotrophic nitrification aerobic denitrification":
            np.asarray([0.98, 0.1]),
    }
    merged = merge_frontier_signals(
        signals, name_vectors=name_vectors, semantic_threshold=0.90
    )
    assert len(merged) == 1
    assert set(merged[0].direction_ids) == {"dir_1", "dir_2"}


def test_dissimilar_vectors_do_not_merge() -> None:
    signals = [
        _signal("dir_1", "HNAD", "candidate"),
        _signal("dir_2", "Photolysis", "candidate"),
    ]
    name_vectors = {
        "dir_1::HNAD": np.asarray([1.0, 0.0]),
        "dir_2::Photolysis": np.asarray([0.0, 1.0]),
    }
    assert len(merge_frontier_signals(signals, name_vectors=name_vectors)) == 2


def test_canonical_statement_prefers_longest_on_semantic_merge() -> None:
    long_name = "Heterotrophic nitrification aerobic denitrification"
    signals = [
        _signal("dir_1", "HNAD", "emerging_frontier"),
        _signal("dir_2", long_name, "candidate"),
    ]
    name_vectors = {
        "dir_1::HNAD": np.asarray([1.0, 0.0]),
        f"dir_2::{long_name}": np.asarray([0.98, 0.1]),
    }
    result = merge_frontier_signals(
        signals, name_vectors=name_vectors, semantic_threshold=0.90
    )[0]
    # same-name merges keep name; semantic merges take the longest (most explicit)
    assert result.statement == long_name
