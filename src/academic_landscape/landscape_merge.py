"""Cross-direction frontier signal merge (deterministic code + optional name vectors).

The same frontier thread can surface inside several directions (papers legitimately
participate in multiple clusters). Signals are merged when their canonical thread
names normalize identically or their name vectors are close; the strongest tier,
the union of directions/evidence and the repeat count are retained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class MergedFrontier:
    statement: str
    tier: str
    direction_ids: tuple[str, ...]
    evidence: tuple[str, ...]
    repeat_count: int


_TIER_ORDER = {"candidate": 1, "emerging_frontier": 2}


def _normalize(name: str) -> str:
    lowered = name.lower().strip()
    lowered = re.sub(r"\([^)]*\)", "", lowered)
    return re.sub(r"[\s\-–—,.:;_]+", " ", lowered).strip()


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def merge_frontier_signals(
    signals: Sequence[dict[str, Any]],
    name_vectors: dict[str, np.ndarray] | None = None,
    semantic_threshold: float = 0.90,
) -> list[MergedFrontier]:
    count = len(signals)
    names = [str(signal.get("thread", "")) for signal in signals]
    normalized = [_normalize(name) for name in names]

    similarity: np.ndarray | None = None
    if name_vectors:
        keys = [
            f"{signal.get('direction_id', '')}::{names[index]}"
            for index, signal in enumerate(signals)
        ]
        resolved = [name_vectors.get(key) for key in keys]
        if all(vector is not None for vector in resolved):
            stacked = np.asarray(resolved, dtype=float)
            stacked = stacked / np.linalg.norm(stacked, axis=1, keepdims=True)
            similarity = stacked @ stacked.T

    dsu = _DSU(count)
    for i in range(count):
        for j in range(i + 1, count):
            same_name = bool(normalized[i]) and normalized[i] == normalized[j]
            close = similarity is not None and similarity[i][j] >= semantic_threshold
            if same_name or close:
                dsu.union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(dsu.find(index), []).append(index)

    merged: list[MergedFrontier] = []
    for member_indices in sorted(groups.values(), key=lambda g: min(g)):
        sub_names = [names[index] for index in member_indices]
        sub_norms = {normalized[index] for index in member_indices}
        if len(sub_norms) == 1:
            # identical thread names: keep that shared name
            statement = next(
                name for name in sub_names if _normalize(name) in sub_norms
            )
        else:
            # semantic merge of different wordings: keep the most explicit
            statement = max(sub_names, key=len)

        tier = max(
            (str(signals[index].get("tier", "candidate")) for index in member_indices),
            key=lambda value: _TIER_ORDER.get(value, 0),
        )
        direction_ids = tuple(
            sorted({str(signals[index].get("direction_id", "")) for index in member_indices})
        )
        evidence = tuple(
            sorted(
                {
                    evidence_id
                    for index in member_indices
                    for evidence_id in signals[index].get("evidence", [])
                }
            )
        )
        merged.append(
            MergedFrontier(
                statement=statement,
                tier=tier,
                direction_ids=direction_ids,
                evidence=evidence,
                repeat_count=len(member_indices),
            )
        )
    return merged
