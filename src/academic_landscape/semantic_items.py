"""Flatten validated single-paper extractions into corpus-level semantic items.

A semantic item is one concept / theoretical framework / method occurrence with a
deterministic, snapshot-scoped id and globally-scoped evidence references. These
items are the unit indexed for embedding-based clustering and retrieval; they are
derived records (rebuildable from extraction results), never authority data.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

_GROUP_MAP: tuple[tuple[str, str, str, str], ...] = (
    ("concepts", "concept", "name", "definition"),
    ("theoretical_frameworks", "framework", "name", "description"),
    ("methods", "method", "name", "description"),
)


def _normalize_for_id(name: str) -> str:
    lowered = name.lower().strip()
    lowered = re.sub(r"\([^)]*\)", "", lowered)  # acronym parentheses
    return re.sub(r"[\s\-–—,.:;_]+", " ", lowered).strip()


def deterministic_item_id(
    snapshot_id: str,
    paper_id: str,
    item_type: str,
    name: str,
) -> str:
    payload = "|".join(
        (snapshot_id, paper_id, item_type, _normalize_for_id(name))
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"item_{digest}"


@dataclass(frozen=True)
class SemanticItem:
    item_id: str
    snapshot_id: str
    paper_id: str
    item_type: str
    name: str
    description: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        # tolerate list input when reconstructing from a serialized dict
        if not isinstance(self.evidence_ids, tuple):
            object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "snapshot_id": self.snapshot_id,
            "paper_id": self.paper_id,
            "item_type": self.item_type,
            "name": self.name,
            "description": self.description,
            "evidence_ids": list(self.evidence_ids),
        }


def build_semantic_items(
    snapshot_id: str,
    extraction_results: Iterable[dict[str, Any]],
) -> list[SemanticItem]:
    seen: set[str] = set()
    items: list[SemanticItem] = []
    for extraction in extraction_results:
        paper_id = extraction["paper_id"]
        for group, item_type, name_key, desc_key in _GROUP_MAP:
            for entry in extraction.get(group, []):
                name = entry.get(name_key, "")
                item_id = deterministic_item_id(
                    snapshot_id, paper_id, item_type, name
                )
                if item_id in seen:
                    continue
                seen.add(item_id)
                evidence_ids = tuple(
                    f"{paper_id}::{local_id}"
                    for local_id in entry.get("evidence_ids", [])
                )
                items.append(
                    SemanticItem(
                        item_id=item_id,
                        snapshot_id=snapshot_id,
                        paper_id=paper_id,
                        item_type=item_type,
                        name=name,
                        description=entry.get(desc_key, ""),
                        evidence_ids=evidence_ids,
                    )
                )
    return items
