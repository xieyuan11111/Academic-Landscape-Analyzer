"""Bounded within-direction comparison: commonalities, differences, controversies.

For each research direction the model receives ONLY that direction's papers
(year, key results, A-level claims and the exact evidence quotes it may cite).
Every returned statement must cite globally-scoped evidence ids; code validates
existence, ownership and difference dimensions. A controversy whose positions
all come from the same paper is not an academic controversy: it is recorded as
a within-paper tension for later review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .direction_naming import ResearchDirection

VALID_DIMENSIONS = {"object", "conditions", "method", "conclusion"}

_SYSTEM = """You compare papers that belong to ONE research direction. You receive each paper's
publication year and its key results/claims, each tagged with evidence IDs and exact quotes.
Produce a rigorous comparison as JSON with three arrays:
- commonalities: what the papers agree on or share (findings, mechanisms, assumptions)
- differences: how they differ (research object, system/conditions, method, or conclusion)
- controversies: only genuine contradictory positions held by DIFFERENT papers (empty array if none)
Rules:
- Use ONLY evidence IDs provided. Every item needs at least one evidence id.
- Do not invent quotes or findings. Distinguish a real contradiction from a mere difference
  in system or a tension reported inside one paper.
- Each controversy position must cite evidence owned by the paper taking that position.
Return JSON only."""


@dataclass(frozen=True)
class ComparedStatement:
    statement: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ComparedDifference:
    dimension: str
    statement: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class DirectionComparison:
    direction_id: str
    commonalities: tuple[ComparedStatement, ...]
    differences: tuple[ComparedDifference, ...]
    controversies: tuple[dict[str, Any], ...]
    within_paper_tensions: tuple[dict[str, Any], ...]
    validation_errors: tuple[str, ...]


def _build_paper_block(
    paper_id: str,
    extraction: dict[str, Any],
    year: int | None,
) -> tuple[str, set[str], dict[str, str]]:
    quote_by_local = {
        evidence["evidence_id"]: evidence["quote"]
        for evidence in extraction.get("evidence", [])
    }
    allowed = {f"{paper_id}::{local}" for local in quote_by_local}
    owners = {gid: paper_id for gid in allowed}

    lines = [f"### paper_id={paper_id} year={year}"]
    lines.append("Key results:")
    for result in extraction.get("results", []):
        tags = ", ".join(f"{paper_id}::{rid}" for rid in result.get("evidence_ids", []))
        lines.append(f"- ({tags}) {result['text']}")
    a_claims = [
        claim for claim in extraction.get("claims", [])
        if claim.get("evidence_type") == "A"
    ]
    if a_claims:
        lines.append("A-level claims:")
        for claim in a_claims:
            tags = ", ".join(
                f"{paper_id}::{rid}" for rid in claim.get("evidence_ids", [])
            )
            lines.append(f"- ({tags}) {claim['text']}")
    lines.append("Evidence quotes available:")
    for local, quote in quote_by_local.items():
        lines.append(f'- {paper_id}::{local}: "{quote}"')
    return "\n".join(lines), allowed, owners


class ComparisonClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


def _build_comparison_prompt(
    label: str,
    paper_list: str,
    paper_detail: str,
    retrieval_context: str = "",
) -> str:
    context = ""
    if retrieval_context:
        context = (
            "\n\nRETRIEVED DISCOVERY CONTEXT (not authoritative evidence; "
            "do not cite it in output):\n"
            f"{retrieval_context}"
        )
    return (
        f"RESEARCH DIRECTION: {label}\n"
        f"PAPERS:\n{paper_list}\n\n"
        f"PAPER DETAIL:\n{paper_detail}{context}\n\n"
        "Return exactly:\n"
        '{"commonalities": [{"statement": "...", "evidence": ["paper_id::E1"]}],'
        ' "differences": [{"dimension": "object|conditions|method|conclusion",'
        ' "statement": "...", "evidence": ["paper_id::E1"]}],'
        ' "controversies": [{"statement": "...", "positions":'
        ' [{"paper_id": "paper_id", "stance": "...", "evidence": ["paper_id::E1"]}]}]}'
    )


def compare_directions(
    directions: list[ResearchDirection],
    extractions: dict[str, dict[str, Any]],
    years: dict[str, int],
    client: ComparisonClient,
    retrieval_context_by_direction: dict[str, str] | None = None,
) -> list[DirectionComparison]:
    comparisons: list[DirectionComparison] = []
    for direction in directions:
        blocks = []
        allowed: set[str] = set()
        owners: dict[str, str] = {}
        present_papers: list[str] = []
        for paper_id in direction.paper_ids:
            extraction = extractions.get(paper_id)
            if extraction is None:
                continue
            present_papers.append(paper_id)
            block, paper_allowed, paper_owners = _build_paper_block(
                paper_id, extraction, years.get(paper_id)
            )
            blocks.append(block)
            allowed |= paper_allowed
            owners.update(paper_owners)

        member_list = "\n".join(f"- {pid}" for pid in present_papers)
        user_prompt = _build_comparison_prompt(
            direction.label,
            member_list,
            "\n\n".join(blocks),
            (retrieval_context_by_direction or {}).get(direction.direction_id, ""),
        )
        payload = client.complete_json(_SYSTEM, user_prompt)

        errors: list[str] = []
        commonalities = []
        for index, item in enumerate(payload.get("commonalities", [])):
            evidence = tuple(item.get("evidence", []))
            for gid in evidence:
                if gid not in allowed:
                    errors.append(f"commonality[{index}] unknown evidence {gid}")
            commonalities.append(
                ComparedStatement(
                    statement=str(item.get("statement", "")), evidence=evidence
                )
            )

        differences = []
        for index, item in enumerate(payload.get("differences", [])):
            dimension = item.get("dimension")
            if dimension not in VALID_DIMENSIONS:
                errors.append(
                    f"difference[{index}] invalid dimension {dimension!r}"
                )
            evidence = tuple(item.get("evidence", []))
            for gid in evidence:
                if gid not in allowed:
                    errors.append(f"difference[{index}] unknown evidence {gid}")
            differences.append(
                ComparedDifference(
                    dimension=str(dimension),
                    statement=str(item.get("statement", "")),
                    evidence=evidence,
                )
            )

        controversies = []
        tensions = []
        for controversy in payload.get("controversies", []):
            positions = controversy.get("positions", [])
            position_papers = {
                position.get("paper_id") for position in positions
            }
            for position_index, position in enumerate(positions):
                paper_id = position.get("paper_id")
                for gid in position.get("evidence", []):
                    if gid not in allowed:
                        errors.append(
                            f"controversy position[{position_index}] "
                            f"unknown evidence {gid}"
                        )
                    elif owners.get(gid) != paper_id:
                        errors.append(
                            f"controversy position[{position_index}] "
                            f"evidence {gid} not owned by {paper_id}"
                        )
            normalized = {
                "statement": controversy.get("statement", ""),
                "positions": positions,
            }
            if len(position_papers) >= 2:
                controversies.append(normalized)
            else:
                tensions.append(normalized)

        comparisons.append(
            DirectionComparison(
                direction_id=direction.direction_id,
                commonalities=tuple(commonalities),
                differences=tuple(differences),
                controversies=tuple(controversies),
                within_paper_tensions=tuple(tensions),
                validation_errors=tuple(errors),
            )
        )
    return comparisons
