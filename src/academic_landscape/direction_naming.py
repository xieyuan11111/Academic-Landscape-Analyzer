"""Name viable clusters into research directions via bounded local LLM calls.

The model sees ONLY one cluster's semantic items and supplies the human-facing label,
description and maturity status. Cluster membership (which papers/items) is data-derived
and frozen before the call; only viable clusters (>=2 eligible papers) are named.
Invalid/unknown status values fall back to `candidate`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .clustering import ClusterCandidate
from .semantic_items import SemanticItem

VALID_STATUSES = {"mainstream", "emerging", "candidate", "frontier"}

_SYSTEM = """You name a research direction from one tight cluster of related academic concepts/methods.
Return JSON only. Using only the given items, produce a concise academic direction label
(<=8 words), a 1-2 sentence description, and a maturity status.
Status meanings: mainstream (well established), emerging (rising recently), candidate
(not yet mature), frontier (most novel exploration). Do not invent concepts outside the
given items."""


def build_direction_naming_prompt(
    cluster_items: Sequence[SemanticItem],
) -> tuple[str, str]:
    bullet_lines = "\n".join(
        f"- {item.name}: {item.description}" if item.description else f"- {item.name}"
        for item in cluster_items
    )
    user = f"""CLUSTER ITEMS:
{bullet_lines}

Return exactly:
{{"label": "research direction name", "description": "1-2 sentences",
  "status": "mainstream|emerging|candidate|frontier"}}"""
    return _SYSTEM, user


@dataclass(frozen=True)
class ResearchDirection:
    direction_id: str
    cluster_id: int
    label: str
    description: str
    status: str
    paper_ids: tuple[str, ...]
    item_ids: tuple[str, ...]


def deterministic_direction_id(
    snapshot_id: str,
    cluster_id: int,
    paper_ids: Sequence[str],
) -> str:
    payload = "|".join(
        (snapshot_id, str(cluster_id), ",".join(sorted(paper_ids)))
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"dir_{digest}"


class NamingClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


def name_directions(
    snapshot_id: str,
    clusters: Sequence[ClusterCandidate],
    items: Sequence[SemanticItem],
    client: NamingClient,
) -> list[ResearchDirection]:
    items_by_id = {item.item_id: item for item in items}
    directions: list[ResearchDirection] = []
    for cluster in clusters:
        if not cluster.is_viable:
            continue
        cluster_items = [
            items_by_id[item_id]
            for item_id in cluster.item_ids
            if item_id in items_by_id
        ]
        system_prompt, user_prompt = build_direction_naming_prompt(cluster_items)
        raw = client.complete_json(system_prompt, user_prompt)

        status = raw.get("status")
        if status not in VALID_STATUSES:
            status = "candidate"
        eligible_papers = cluster.eligible_paper_ids
        directions.append(
            ResearchDirection(
                direction_id=deterministic_direction_id(
                    snapshot_id, cluster.cluster_id, eligible_papers
                ),
                cluster_id=cluster.cluster_id,
                label=str(raw.get("label", "")).strip() or "unlabeled",
                description=str(raw.get("description", "")).strip(),
                status=status,
                paper_ids=eligible_papers,
                item_ids=cluster.item_ids,
            )
        )
    return directions
