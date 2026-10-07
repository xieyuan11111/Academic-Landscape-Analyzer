"""Embed semantic items into candidate clusters and apply the membership cleaning gate.

Clustering groups item occurrences; a paper legitimately scatters items across several
clusters. Before a cluster can become a research-direction candidate we clean its
roster: a paper with >=2 items in the cluster is a core member, and a single-item
membership is retained only when the paper's whole-item vector is close to the
cluster core (single-item memberships are often cross-field strays). A viable
direction needs at least two eligible papers.

This module is data-side: it consumes already-computed vectors and produces only
candidate records, never academic conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import KMeans

from .semantic_items import SemanticItem

SIM_CLOSE_TO_CORE = 0.72


@dataclass(frozen=True)
class ClusterMember:
    paper_id: str
    item_count: int
    cos_to_core: float
    eligible: bool


@dataclass(frozen=True)
class ClusterCandidate:
    cluster_id: int
    item_ids: tuple[str, ...]
    members: tuple[ClusterMember, ...]

    @property
    def eligible_paper_ids(self) -> tuple[str, ...]:
        return tuple(member.paper_id for member in self.members if member.eligible)

    @property
    def is_viable(self) -> bool:
        return len(self.eligible_paper_ids) >= 2


def _normalized_mean(vectors: np.ndarray) -> np.ndarray:
    mean = np.asarray(vectors).mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


def cluster_items(
    items: Sequence[SemanticItem],
    item_vectors: np.ndarray,
    n_clusters: int,
    random_state: int = 0,
    cos_close_to_core: float = SIM_CLOSE_TO_CORE,
    paper_vectors: dict[str, np.ndarray] | None = None,
) -> list[ClusterCandidate]:
    matrix = np.asarray(item_vectors)
    labels = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=10
    ).fit_predict(matrix)

    # whole-paper vectors default to a normalized mean over ALL of the paper's
    # item vectors; explicit vectors (e.g. built from title/abstract/RQ) win.
    derived_paper_indices: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        derived_paper_indices.setdefault(item.paper_id, []).append(index)
    resolved_paper_vectors = {
        paper_id: _normalized_mean(matrix[indices])
        for paper_id, indices in derived_paper_indices.items()
    }
    if paper_vectors:
        for pid, vector in paper_vectors.items():
            resolved_paper_vectors[pid] = np.asarray(vector)

    candidates: list[ClusterCandidate] = []
    for cluster_id in range(n_clusters):
        member_indices = [
            index for index, label in enumerate(labels) if int(label) == cluster_id
        ]
        item_ids = tuple(items[index].item_id for index in member_indices)

        counts: dict[str, int] = {}
        for index in member_indices:
            paper_id = items[index].paper_id
            counts[paper_id] = counts.get(paper_id, 0) + 1

        multi_item_papers = [
            paper_id for paper_id, count in counts.items() if count >= 2
        ]
        if multi_item_papers:
            core = _normalized_mean(
                np.stack([resolved_paper_vectors[pid] for pid in multi_item_papers])
            )
        else:
            core = _normalized_mean(matrix[member_indices])

        members = []
        for paper_id, count in counts.items():
            cos_to_core = float(np.dot(resolved_paper_vectors[paper_id], core))
            eligible = count >= 2 or cos_to_core >= cos_close_to_core
            members.append(
                ClusterMember(
                    paper_id=paper_id,
                    item_count=count,
                    cos_to_core=round(cos_to_core, 3),
                    eligible=eligible,
                )
            )
        members.sort(key=lambda m: (-m.item_count, m.paper_id))
        candidates.append(
            ClusterCandidate(
                cluster_id=cluster_id,
                item_ids=item_ids,
                members=tuple(members),
            )
        )
    return candidates
