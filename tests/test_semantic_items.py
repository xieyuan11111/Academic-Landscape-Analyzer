"""Tests for semantic item flattening from validated extraction results."""

from __future__ import annotations

from academic_landscape.semantic_items import (
    SemanticItem,
    build_semantic_items,
    deterministic_item_id,
)

SNAPSHOT_ID = "snapshot_test123"
PAPER_ID = "paper_abc123"

EXTRACTION = {
    "paper_id": PAPER_ID,
    "concepts": [
        {"name": "Heterotrophic nitrification", "definition": "HN process",
         "evidence_ids": ["E1", "E2"]},
    ],
    "theoretical_frameworks": [
        {"name": "Insurance hypothesis", "description": "buffering model",
         "evidence_ids": ["E3"]},
    ],
    "methods": [
        {"name": "qPCR", "description": "gene quantification",
         "evidence_ids": ["E4"]},
    ],
    "evidence": [
        {"evidence_id": "E1", "page_number": 1},
        {"evidence_id": "E2", "page_number": 2},
        {"evidence_id": "E3", "page_number": 3},
        {"evidence_id": "E4", "page_number": 4},
    ],
}


def test_items_flatten_all_three_groups() -> None:
    items = build_semantic_items(SNAPSHOT_ID, [EXTRACTION])
    assert len(items) == 3
    by_type = {item.item_type: item for item in items}
    assert set(by_type) == {"concept", "framework", "method"}
    assert by_type["concept"].name == "Heterotrophic nitrification"
    assert by_type["concept"].description == "HN process"
    assert by_type["method"].name == "qPCR"


def test_evidence_ids_are_globally_scoped() -> None:
    items = build_semantic_items(SNAPSHOT_ID, [EXTRACTION])
    concept = next(item for item in items if item.item_type == "concept")
    assert concept.evidence_ids == (f"{PAPER_ID}::E1", f"{PAPER_ID}::E2")
    framework = next(item for item in items if item.item_type == "framework")
    assert framework.evidence_ids == (f"{PAPER_ID}::E3",)


def test_item_id_is_deterministic_and_scoped() -> None:
    first = deterministic_item_id(SNAPSHOT_ID, PAPER_ID, "concept", "Heterotrophic nitrification")
    same = deterministic_item_id(SNAPSHOT_ID, PAPER_ID, "concept", "Heterotrophic  nitrification")
    assert first == same
    assert first.startswith("item_")
    other_paper = deterministic_item_id(SNAPSHOT_ID, "paper_other", "concept",
                                        "Heterotrophic nitrification")
    other_type = deterministic_item_id(SNAPSHOT_ID, PAPER_ID, "method",
                                       "Heterotrophic nitrification")
    assert first != other_paper
    assert first != other_type


def test_build_is_idempotent_across_papers() -> None:
    once = build_semantic_items(SNAPSHOT_ID, [EXTRACTION])
    twice = build_semantic_items(SNAPSHOT_ID, [EXTRACTION, EXTRACTION])
    assert len(once) == len(twice)
    assert {item.item_id for item in twice} == {item.item_id for item in once}


def test_snapshot_and_paper_bound_to_item() -> None:
    items = build_semantic_items(SNAPSHOT_ID, [EXTRACTION])
    for item in items:
        assert item.snapshot_id == SNAPSHOT_ID
        assert item.paper_id == PAPER_ID


def test_to_dict_roundtrip_shape() -> None:
    items = build_semantic_items(SNAPSHOT_ID, [EXTRACTION])
    serialized = items[0].to_dict()
    assert serialized["item_id"] == items[0].item_id
    assert serialized["evidence_ids"] == list(items[0].evidence_ids)
    restored = SemanticItem(**serialized)
    assert restored == items[0]


def test_entries_without_evidence_get_empty_tuple() -> None:
    extraction = {
        "paper_id": PAPER_ID,
        "concepts": [{"name": "Loose concept", "definition": "", "evidence_ids": []}],
        "theoretical_frameworks": [],
        "methods": [],
        "evidence": [],
    }
    items = build_semantic_items(SNAPSHOT_ID, [extraction])
    assert items[0].evidence_ids == ()
