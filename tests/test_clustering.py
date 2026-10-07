"""Tests for embedding clustering with membership cleaning gate."""

from __future__ import annotations

import numpy as np

from academic_landscape.clustering import (
    ClusterCandidate,
    cluster_items,
)
from academic_landscape.semantic_items import SemanticItem

SNAPSHOT_ID = "snapshot_test"


def _item(item_id: str, paper_id: str, name: str = "x") -> SemanticItem:
    return SemanticItem(
        item_id=item_id,
        snapshot_id=SNAPSHOT_ID,
        paper_id=paper_id,
        item_type="concept",
        name=name,
        description="",
        evidence_ids=(),
    )


def _unit(x: float, y: float) -> np.ndarray:
    vector = np.asarray([x, y], dtype=float)
    return vector / np.linalg.norm(vector)


def _build_scenario() -> tuple[list[SemanticItem], np.ndarray, dict[str, np.ndarray]]:
    """P1/P2 solid members of cluster A; P3 stray (1 item in A, 4 in B);
    P4 one item in A and one in B, but injected whole-paper vector is close to A."""
    items: list[SemanticItem] = []
    vectors: list[np.ndarray] = []

    def add(item_id: str, paper_id: str, vector: np.ndarray) -> None:
        items.append(_item(item_id, paper_id, item_id))
        vectors.append(vector)

    # P1: three items near A=[1,0]; P2: two items near A
    for k in range(3):
        add(f"i_p1_{k}", "paper_p1", _unit(1.0, 0.02 * k))
    for k in range(2):
        add(f"i_p2_{k}", "paper_p2", _unit(1.0, -0.02 * k))
    # P3: one item near A (the stray), four items near B=[0,1]
    add("i_p3_a", "paper_p3", _unit(0.98, 0.1))
    for k in range(4):
        add(f"i_p3_b{k}", "paper_p3", _unit(0.02 * k, 1.0))
    # P4: one item near A, one item near B
    add("i_p4_a", "paper_p4", _unit(1.0, 0.05))
    add("i_p4_b", "paper_p4", _unit(0.05, 1.0))

    # injected paper vectors: P4 overall semantics stay close to A;
    # P3's overall vector is pulled towards B.
    paper_vectors = {
        "paper_p4": _unit(0.97, 0.24),
        "paper_p3": _unit(0.3, 0.95),
    }
    return items, np.asarray(vectors), paper_vectors


def _cluster_a(clusters: list[ClusterCandidate]) -> ClusterCandidate:
    for cluster in clusters:
        if any(item_id.startswith("i_p1") for item_id in cluster.item_ids):
            return cluster
    raise AssertionError("cluster A not found")


def test_clustering_produces_two_clusters() -> None:
    items, vectors, paper_vectors = _build_scenario()
    clusters = cluster_items(
        items, vectors, n_clusters=2, random_state=0, paper_vectors=paper_vectors
    )
    assert len(clusters) == 2
    assert sorted(cluster.cluster_id for cluster in clusters) == [0, 1]


def test_membership_counts_per_paper() -> None:
    items, vectors, paper_vectors = _build_scenario()
    cluster = _cluster_a(
        cluster_items(items, vectors, n_clusters=2, random_state=0,
                      paper_vectors=paper_vectors)
    )
    by_paper = {member.paper_id: member for member in cluster.members}
    assert by_paper["paper_p1"].item_count == 3
    assert by_paper["paper_p2"].item_count == 2
    assert by_paper["paper_p3"].item_count == 1
    assert by_paper["paper_p4"].item_count == 1


def test_multi_item_members_are_eligible() -> None:
    items, vectors, paper_vectors = _build_scenario()
    cluster = _cluster_a(
        cluster_items(items, vectors, n_clusters=2, random_state=0,
                      paper_vectors=paper_vectors)
    )
    by_paper = {member.paper_id: member for member in cluster.members}
    assert by_paper["paper_p1"].eligible is True
    assert by_paper["paper_p2"].eligible is True


def test_single_item_stray_is_cleaned() -> None:
    items, vectors, paper_vectors = _build_scenario()
    cluster = _cluster_a(
        cluster_items(items, vectors, n_clusters=2, random_state=0,
                      paper_vectors=paper_vectors)
    )
    p3 = next(m for m in cluster.members if m.paper_id == "paper_p3")
    assert p3.eligible is False
    assert p3.cos_to_core < 0.72


def test_single_item_but_close_paper_vector_is_eligible() -> None:
    items, vectors, paper_vectors = _build_scenario()
    cluster = _cluster_a(
        cluster_items(items, vectors, n_clusters=2, random_state=0,
                      paper_vectors=paper_vectors)
    )
    p4 = next(m for m in cluster.members if m.paper_id == "paper_p4")
    assert p4.eligible is True
    assert p4.cos_to_core >= 0.72


def test_cluster_viability_requires_two_eligible_papers() -> None:
    items, vectors, paper_vectors = _build_scenario()
    clusters = cluster_items(
        items, vectors, n_clusters=2, random_state=0, paper_vectors=paper_vectors
    )
    cluster_a = _cluster_a(clusters)
    assert cluster_a.is_viable is True
    assert set(cluster_a.eligible_paper_ids) == {"paper_p1", "paper_p2", "paper_p4"}
    # cluster B: P3 alone eligible (P4 single item there and far from B core)
    cluster_b = next(c for c in clusters if c.cluster_id != cluster_a.cluster_id)
    assert cluster_b.is_viable is False


def test_cluster_candidate_carries_item_ids() -> None:
    items, vectors, paper_vectors = _build_scenario()
    cluster = _cluster_a(
        cluster_items(items, vectors, n_clusters=2, random_state=0,
                      paper_vectors=paper_vectors)
    )
    assert len(cluster.item_ids) == 3 + 2 + 1 + 1


def test_paper_vectors_default_to_derivation_when_absent() -> None:
    items, vectors, _ = _build_scenario()
    # without injection: P4's two items split A/B; it is single-item in each cluster
    clusters = cluster_items(items, vectors, n_clusters=2, random_state=0)
    cluster_a = _cluster_a(clusters)
    p4 = next(m for m in cluster_a.members if m.paper_id == "paper_p4")
    assert p4.item_count == 1
