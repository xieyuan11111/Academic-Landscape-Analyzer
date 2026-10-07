"""Adapt pipeline artifacts into the Neo4j projection payload (schema 0.1).

Paper-local evidence ids collide across papers (each extraction starts at E1), and
the projection validator requires globally unique ids. Papers are globalized with
`globalize_evidence_ids`; directions aggregate the resulting global ids and the
paper set is limited to papers that belong to a named direction.

Paper relations (FR-04) are globalized upstream by `relation_extraction`; the
adapter preserves their global evidence-id form so they line up with the
globalized paper evidence when projected.
"""

from __future__ import annotations

from typing import Any, Sequence

from .direction_naming import ResearchDirection
from .evidence_scope import globalize_evidence_ids
from .relation_extraction import PaperRelation


def _normalize_paper_relation(rel: PaperRelation | dict[str, Any]) -> dict[str, Any]:
    """Project a PaperRelation (or matching dict) into the graph_projector shape.

    The graph_projector ``_write_projection`` relations branch reads
    ``relation_id``, ``source_paper_id``, ``target_paper_id``,
    ``relation_type`` and ``evidence_ids``. ``evidence_ids`` is flattened from
    PaperRelation's split ``source_evidence_ids`` / ``target_evidence_ids``;
    PaperRelation already stores them in the global ``paper_id::local`` form,
    so they line up with the globalized paper evidence nodes.
    """
    if isinstance(rel, PaperRelation):
        return {
            "relation_id": rel.relation_id,
            "source_paper_id": rel.source_paper_id,
            "target_paper_id": rel.target_paper_id,
            "relation_type": rel.relation_type,
            "evidence_ids": [
                *rel.source_evidence_ids,
                *rel.target_evidence_ids,
            ],
            "statement": rel.statement,
            "difference_dimension": rel.difference_dimension,
            "comparability_note": rel.comparability_note,
            "uncertainty": rel.uncertainty,
            "snapshot_id": rel.snapshot_id,
        }
    return {
        "relation_id": rel.get("relation_id"),
        "source_paper_id": rel.get("source_paper_id"),
        "target_paper_id": rel.get("target_paper_id"),
        "relation_type": rel.get("relation_type"),
        "evidence_ids": [
            *(rel.get("source_evidence_ids") or []),
            *(rel.get("target_evidence_ids") or []),
        ],
        "statement": rel.get("statement", ""),
        "difference_dimension": rel.get("difference_dimension"),
        "comparability_note": rel.get("comparability_note", ""),
        "uncertainty": rel.get("uncertainty", ""),
        "snapshot_id": rel.get("snapshot_id", ""),
    }


def _dedupe_relations(
    relations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop duplicate relations by deterministic ``relation_id`` (last write wins)."""
    seen: dict[str, dict[str, Any]] = {}
    for rel in relations:
        rel_id = rel.get("relation_id")
        if rel_id is None:
            # No id: keep all (caller expects graph_projector to derive one).
            seen[f"_anon_{id(rel)}"] = rel
        else:
            seen[str(rel_id)] = rel
    return list(seen.values())


def build_projection_payload(
    snapshot_id: str,
    directions: Sequence[ResearchDirection],
    extractions: Sequence[dict[str, Any]],
    paper_relations: Sequence[PaperRelation | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    member_paper_ids = {
        paper_id
        for direction in directions
        for paper_id in direction.paper_ids
    }
    member_papers = [
        dict(extraction)
        for extraction in extractions
        if extraction.get("paper_id") in member_paper_ids
    ]
    papers = globalize_evidence_ids(member_papers)

    evidence_by_paper = {
        paper["paper_id"]: [
            evidence["evidence_id"] for evidence in paper.get("evidence", [])
        ]
        for paper in papers
    }

    mapped_directions = []
    for direction in directions:
        evidence_ids: set[str] = set()
        for paper_id in direction.paper_ids:
            evidence_ids.update(evidence_by_paper.get(paper_id, []))
        mapped_directions.append(
            {
                "direction_id": direction.direction_id,
                "label": direction.label,
                "status": direction.status,
                "paper_ids": list(direction.paper_ids),
                "evidence_ids": sorted(evidence_ids),
            }
        )

    relations: list[dict[str, Any]] = []
    if paper_relations:
        for rel in paper_relations:
            relations.append(_normalize_paper_relation(rel))
        relations = _dedupe_relations(relations)

    return {
        "schema_version": "0.1",
        "snapshot_id": snapshot_id,
        "papers": papers,
        "directions": mapped_directions,
        "relations": relations,
        "controversies": [],
        "frontier_candidates": [],
    }
