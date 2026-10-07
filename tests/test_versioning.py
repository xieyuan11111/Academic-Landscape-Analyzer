"""Tests for FR-11 analysis versioning (src/academic_landscape/versioning.py)."""

from __future__ import annotations

import pytest

from academic_landscape.versioning import (
    AnalysisVersion,
    VersionStore,
    compute_config_hash,
    diff_versions,
    new_version,
)


def _payload_a() -> dict:
    return {
        "papers": [{"id": "p1", "title": "Paper 1"}, {"id": "p2", "title": "Paper 2"}],
        "directions": [{"name": "Direction A"}],
        "relations": [{"source": "p1", "target": "p2"}],
        "frontier_candidates": [{"topic": "X"}],
    }


def _payload_b() -> dict:
    return {
        "papers": [{"id": "p2", "title": "Paper 2"}, {"id": "p3", "title": "Paper 3"}],
        "directions": [{"name": "Direction B"}],
        "relations": [],
        "frontier_candidates": [],
    }


def test_compute_config_hash_is_key_order_independent():
    a = {"b": 2, "a": 1, "c": {"y": 2, "x": 1}}
    b = {"a": 1, "c": {"x": 1, "y": 2}, "b": 2}
    assert compute_config_hash(a) == compute_config_hash(b)


def test_compute_config_hash_changes_when_value_changes():
    base = {"temperature": 0.0, "top_p": 0.9}
    mutated = {"temperature": 0.1, "top_p": 0.9}
    assert compute_config_hash(base) != compute_config_hash(mutated)


def test_new_version_is_deterministic_and_isolates_inputs():
    config = {"temperature": 0.0}
    v1 = new_version(
        snapshot_id="snap1", config=config, model="m1", prompt_version="p1"
    )
    v2 = new_version(
        snapshot_id="snap1", config=config, model="m1", prompt_version="p1"
    )
    # Deterministic: same inputs yield same version_id and config_hash
    assert v1.version_id == v2.version_id
    assert v1.config_hash == v2.config_hash
    # version_id format: 'ver_' + 16 hex chars
    assert v1.version_id.startswith("ver_")
    assert len(v1.version_id) == len("ver_") + 16

    # Each input axis is isolated
    diff_snapshot = new_version(
        snapshot_id="snap2", config=config, model="m1", prompt_version="p1"
    )
    diff_config = new_version(
        snapshot_id="snap1", config={"temperature": 0.1}, model="m1", prompt_version="p1"
    )
    diff_model = new_version(
        snapshot_id="snap1", config=config, model="m2", prompt_version="p1"
    )
    diff_prompt = new_version(
        snapshot_id="snap1", config=config, model="m1", prompt_version="p2"
    )
    ids = {
        v1.version_id,
        diff_snapshot.version_id,
        diff_config.version_id,
        diff_model.version_id,
        diff_prompt.version_id,
    }
    assert len(ids) == 5


def test_version_store_save_load_roundtrip(tmp_path):
    store = VersionStore(tmp_path / "versions")
    v = new_version(
        snapshot_id="snap1",
        config={"temperature": 0.0},
        model="m1",
        prompt_version="p1",
        note="first analysis",
    )
    payload = _payload_a()
    path = store.save(v, payload)

    assert path.exists()
    assert path.name == f"{v.version_id}.json"

    loaded_v, loaded_p = store.load(v.version_id)
    assert loaded_v == v
    assert loaded_p == payload


def test_version_store_save_is_idempotent_for_identical_content(tmp_path):
    store = VersionStore(tmp_path / "versions")
    v = new_version(
        snapshot_id="snap1", config={"temperature": 0.0}, model="m1", prompt_version="p1"
    )
    payload = _payload_a()
    p1 = store.save(v, payload)
    p2 = store.save(v, payload)
    assert p1 == p2

    # Reload still yields original payload
    _, loaded = store.load(v.version_id)
    assert loaded == payload


def test_version_store_save_rejects_overwrite_with_different_content(tmp_path):
    store = VersionStore(tmp_path / "versions")
    v = new_version(
        snapshot_id="snap1", config={"temperature": 0.0}, model="m1", prompt_version="p1"
    )
    store.save(v, _payload_a())
    with pytest.raises(ValueError):
        store.save(v, _payload_b())
    # File content must remain the original
    _, loaded = store.load(v.version_id)
    assert loaded == _payload_a()


def test_diff_versions_added_removed_and_missing_key_tolerance():
    base = {
        "papers": [{"id": "p1"}, {"id": "p2"}],
        "directions": [{"name": "A"}],
        "relations": [{"source": "p1", "target": "p2"}],
        "frontier_candidates": [{"topic": "X"}],
    }
    # target omits 'relations' entirely; frontier_candidates is present but empty
    target = {
        "papers": [{"id": "p2"}, {"id": "p3"}],
        "directions": [{"name": "B"}],
        "frontier_candidates": [],
    }
    diff = diff_versions(base, target)

    assert diff["added_papers"] == [{"id": "p3"}]
    assert diff["removed_papers"] == [{"id": "p1"}]
    assert diff["added_directions"] == [{"name": "B"}]
    assert diff["removed_directions"] == [{"name": "A"}]
    # Missing key => treated as empty list, so all base relations are 'removed'
    assert diff["added_relations"] == []
    assert diff["removed_relations"] == [{"source": "p1", "target": "p2"}]
    assert diff["added_frontiers"] == []
    assert diff["removed_frontiers"] == [{"topic": "X"}]


def test_diff_versions_tolerates_none_inputs_and_missing_top_level_keys():
    base = {"papers": None, "directions": None}
    target = {}
    diff = diff_versions(base, target)
    assert diff == {
        "added_papers": [],
        "removed_papers": [],
        "added_directions": [],
        "removed_directions": [],
        "added_relations": [],
        "removed_relations": [],
        "added_frontiers": [],
        "removed_frontiers": [],
    }


def test_version_store_list_returns_versions_sorted_by_created_at(tmp_path):
    store = VersionStore(tmp_path / "versions")
    # Out-of-order created_at values
    v_jan3 = AnalysisVersion(
        version_id="ver_0000000000000001",
        snapshot_id="snap1",
        config_hash="h1",
        model="m1",
        prompt_version="p1",
        created_at="2024-01-03T00:00:00+00:00",
    )
    v_jan1 = AnalysisVersion(
        version_id="ver_0000000000000002",
        snapshot_id="snap2",
        config_hash="h2",
        model="m1",
        prompt_version="p1",
        created_at="2024-01-01T00:00:00+00:00",
    )
    v_jan2 = AnalysisVersion(
        version_id="ver_0000000000000003",
        snapshot_id="snap3",
        config_hash="h3",
        model="m1",
        prompt_version="p1",
        created_at="2024-01-02T00:00:00+00:00",
    )
    # Save in shuffled order
    store.save(v_jan3, {"x": 3})
    store.save(v_jan1, {"x": 1})
    store.save(v_jan2, {"x": 2})

    listed = store.list()
    assert [v.version_id for v in listed] == [
        v_jan1.version_id,
        v_jan2.version_id,
        v_jan3.version_id,
    ]


def test_new_version_records_parent_version_id_and_note():
    parent = new_version(
        snapshot_id="snap1",
        config={"temperature": 0.0},
        model="m1",
        prompt_version="p1",
    )
    child = new_version(
        snapshot_id="snap1",
        config={"temperature": 0.0},
        model="m1",
        prompt_version="p1",
        parent=parent,
        note="human correction",
    )
    assert child.parent_version_id == parent.version_id
    assert child.note == "human correction"
    # Child is still deterministic and derives a fresh version_id (since created_at changes
    # but the inputs are otherwise identical to parent — version_id stays stable across calls).
    child_again = new_version(
        snapshot_id="snap1",
        config={"temperature": 0.0},
        model="m1",
        prompt_version="p1",
        parent=parent,
        note="human correction",
    )
    assert child.version_id == child_again.version_id