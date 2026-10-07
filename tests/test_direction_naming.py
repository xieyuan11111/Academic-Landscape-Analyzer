"""Tests for cluster-local research direction naming."""

from __future__ import annotations

from typing import Any


from academic_landscape.clustering import (
    ClusterCandidate,
    ClusterMember,
)
from academic_landscape.direction_naming import (
    name_directions,
)
from academic_landscape.semantic_items import SemanticItem

SNAPSHOT_ID = "snapshot_test"


class RecordingClient:
    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._responses = responses or [
            {"label": "Nitrogen removal", "description": "HNAD studies",
             "status": "emerging"}
        ]

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append((system_prompt, user_prompt))
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def _item(item_id: str, paper_id: str, name: str) -> SemanticItem:
    return SemanticItem(
        item_id=item_id,
        snapshot_id=SNAPSHOT_ID,
        paper_id=paper_id,
        item_type="concept",
        name=name,
        description=f"definition of {name}",
        evidence_ids=(f"{paper_id}::E1",),
    )


def _cluster(
    cluster_id: int,
    members: list[ClusterMember],
    item_ids: tuple[str, ...],
) -> ClusterCandidate:
    return ClusterCandidate(
        cluster_id=cluster_id, item_ids=item_ids, members=tuple(members)
    )


def _member(paper_id: str, count: int, eligible: bool) -> ClusterMember:
    return ClusterMember(
        paper_id=paper_id, item_count=count, cos_to_core=0.9 if eligible else 0.3,
        eligible=eligible,
    )


def _viable_cluster() -> tuple[ClusterCandidate, list[SemanticItem]]:
    items = [
        _item("i1", "paper_a", "HNAD"),
        _item("i2", "paper_a", "denitrification"),
        _item("i3", "paper_b", "free ammonia"),
    ]
    cluster = _cluster(
        3,
        [_member("paper_a", 2, True), _member("paper_b", 1, True)],
        ("i1", "i2", "i3"),
    )
    return cluster, items


def test_viable_cluster_is_named() -> None:
    cluster, items = _viable_cluster()
    client = RecordingClient()
    directions = name_directions(SNAPSHOT_ID, [cluster], items, client)
    assert len(directions) == 1
    direction = directions[0]
    assert direction.label == "Nitrogen removal"
    assert direction.description == "HNAD studies"
    assert direction.status == "emerging"


def test_direction_binds_eligible_papers_and_items() -> None:
    cluster, items = _viable_cluster()
    client = RecordingClient()
    direction = name_directions(SNAPSHOT_ID, [cluster], items, client)[0]
    assert direction.paper_ids == ("paper_a", "paper_b")
    assert direction.item_ids == ("i1", "i2", "i3")
    assert direction.cluster_id == 3


def test_direction_id_is_deterministic() -> None:
    cluster, items = _viable_cluster()
    first = name_directions(SNAPSHOT_ID, [cluster], items, RecordingClient())[0]
    second = name_directions(SNAPSHOT_ID, [cluster], items, RecordingClient())[0]
    assert first.direction_id == second.direction_id
    assert first.direction_id.startswith("dir_")


def test_non_viable_cluster_is_not_named() -> None:
    cluster = _cluster(
        5, [_member("paper_a", 2, True)], ("i9",)
    )
    client = RecordingClient()
    assert name_directions(SNAPSHOT_ID, [cluster], [_item("i9","paper_a","x")], client) == []
    assert client.calls == []


def test_model_only_sees_its_own_cluster_items() -> None:
    cluster, items = _viable_cluster()
    other = _item("i_other", "paper_z", "unrelated topic")
    client = RecordingClient()
    name_directions(SNAPSHOT_ID, [cluster], items + [other], client)
    _, user_prompt = client.calls[0]
    assert "HNAD" in user_prompt
    assert "free ammonia" in user_prompt
    assert "unrelated topic" not in user_prompt


def test_invalid_status_falls_back_to_candidate() -> None:
    client = RecordingClient(
        [{"label": "X", "description": "Y", "status": "world_famous"}]
    )
    cluster, items = _viable_cluster()
    direction = name_directions(SNAPSHOT_ID, [cluster], items, client)[0]
    assert direction.status == "candidate"


def test_missing_status_falls_back_to_candidate() -> None:
    client = RecordingClient([{"label": "X", "description": "Y"}])
    cluster, items = _viable_cluster()
    direction = name_directions(SNAPSHOT_ID, [cluster], items, client)[0]
    assert direction.status == "candidate"
