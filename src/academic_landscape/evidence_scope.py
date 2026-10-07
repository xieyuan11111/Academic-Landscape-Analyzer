"""Lift paper-local evidence IDs to corpus-global scope for cross-paper synthesis."""

from __future__ import annotations

import copy
from typing import Any


def globalize_evidence_ids(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a deep copy of ``papers`` with globally unique evidence IDs.

    Each paper scopes its evidence IDs as ``{paper_id}::{local_id}``, and every
    ``evidence_ids`` reference in the paper record is remapped to match. This is
    required before corpus-level synthesis: single-paper extraction numbers its
    evidence E1..En per paper, so the IDs collide once papers are combined.
    """
    out = copy.deepcopy(papers)
    for paper in out:
        if not isinstance(paper, dict):
            continue
        paper_id = paper.get("paper_id")
        evidence = paper.get("evidence")
        if not isinstance(evidence, list):
            continue
        mapping: dict[str, str] = {}
        for evidence_item in evidence:
            if not isinstance(evidence_item, dict):
                continue
            local_id = evidence_item.get("evidence_id")
            if isinstance(local_id, str) and local_id:
                mapping[local_id] = f"{paper_id}::{local_id}"
        for evidence_item in evidence:
            if isinstance(evidence_item, dict):
                local_id = evidence_item.get("evidence_id")
                if local_id in mapping:
                    evidence_item["evidence_id"] = mapping[local_id]
        _remap_references(paper, mapping)
    return out


def _remap_references(node: Any, mapping: dict[str, str]) -> None:
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key == "evidence_ids" and isinstance(value, list):
                node[key] = [mapping.get(item, item) for item in value]
            else:
                _remap_references(value, mapping)
    elif isinstance(node, list):
        for item in node:
            _remap_references(item, mapping)