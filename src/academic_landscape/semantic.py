"""Validation boundary for LLM-produced semantic analysis.

The LLM may propose directions, claims and relations, but it is not allowed to
invent source evidence. This module validates the proposal against the immutable
corpus snapshot before anything can become a report or graph projection.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .models import CorpusSnapshot, DocumentRecord
from .relation_types import VALID_RELATION_TYPES

_ALLOWED_EVIDENCE_TYPES = {"A", "B", "C", "D"}
# The relation vocabulary has one home: academic_landscape.relation_types.
# Re-declaring it here is what let the validator drift behind the extraction
# prompt and abort the first real MAG-50 run with `invalid_relation_type`.
_ALLOWED_RELATIONS = VALID_RELATION_TYPES


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    payload: dict[str, Any]
    errors: list[dict[str, Any]]


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **details}


def _normalise_for_match(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(str.maketrans({
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "‐": "-",
        "‑": "-",
        "–": "-",
        "—": "-",
    }))
    normalized = re.sub(r"-\s*(?:\r?\n)\s*", "", normalized)
    normalized = re.sub(r"(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])", "", normalized)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _quote_exists(page_text: str, quote: str) -> bool:
    if not quote or not page_text:
        return False
    return _normalise_for_match(quote) in _normalise_for_match(page_text)


def _required_list(payload: dict[str, Any], key: str, errors: list[dict[str, Any]]) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        errors.append(_error("invalid_collection", f"{key} must be a list", field=key))
        return []
    return value


def _validate_evidence(
    snapshot: CorpusSnapshot,
    papers: list[Any],
    errors: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, DocumentRecord]]:
    documents_by_artifact = {document.artifact_id: document for document in snapshot.documents}
    papers_by_id: dict[str, dict[str, Any]] = {}
    evidence_by_id: dict[str, dict[str, Any]] = {}
    evidence_owner: dict[str, str] = {}

    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            errors.append(_error("invalid_paper", "paper must be an object", index=index))
            continue
        paper_id = paper.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            errors.append(_error("missing_paper_id", "paper_id is required", index=index))
            continue
        if paper_id in papers_by_id:
            errors.append(_error("duplicate_paper_id", "paper_id is repeated", paper_id=paper_id))
            continue
        papers_by_id[paper_id] = paper
        if paper_id not in {document.paper_id for document in snapshot.documents}:
            errors.append(_error("unknown_paper_id", "paper_id is not in the corpus snapshot", paper_id=paper_id))

        evidence_items = paper.get("evidence", [])
        if not isinstance(evidence_items, list):
            errors.append(_error("invalid_collection", "paper evidence must be a list", paper_id=paper_id))
            continue
        for evidence_index, evidence in enumerate(evidence_items):
            if not isinstance(evidence, dict):
                errors.append(
                    _error(
                        "invalid_evidence",
                        "evidence must be an object",
                        paper_id=paper_id,
                        index=evidence_index,
                    )
                )
                continue
            evidence_id = evidence.get("evidence_id")
            artifact_id = evidence.get("artifact_id")
            page_number = evidence.get("page_number")
            quote = evidence.get("quote")
            if not isinstance(evidence_id, str) or not evidence_id:
                errors.append(_error("missing_evidence_id", "evidence_id is required", paper_id=paper_id))
                continue
            if evidence_id in evidence_by_id:
                errors.append(_error("duplicate_evidence_id", "evidence_id is repeated", evidence_id=evidence_id))
                continue
            evidence_by_id[evidence_id] = evidence
            evidence_owner[evidence_id] = paper_id

            document = documents_by_artifact.get(artifact_id)
            if document is None:
                errors.append(
                    _error(
                        "unknown_artifact_id",
                        "evidence artifact_id is not in the corpus snapshot",
                        evidence_id=evidence_id,
                        artifact_id=artifact_id,
                    )
                )
                continue
            if document.status != "parsed":
                errors.append(
                    _error(
                        "artifact_not_eligible",
                        "evidence must refer to a parsed text document",
                        evidence_id=evidence_id,
                        artifact_id=artifact_id,
                        status=document.status,
                    )
                )
            if document.paper_id != paper_id:
                errors.append(
                    _error(
                        "evidence_paper_mismatch",
                        "evidence artifact belongs to a different paper_id",
                        evidence_id=evidence_id,
                        paper_id=paper_id,
                        artifact_paper_id=document.paper_id,
                    )
                )
            if not isinstance(page_number, int) or isinstance(page_number, bool):
                errors.append(_error("invalid_page_number", "page_number must be an integer", evidence_id=evidence_id))
                continue
            if page_number < 1 or page_number > document.page_count:
                errors.append(
                    _error(
                        "page_out_of_range",
                        "evidence page_number is outside the parsed document",
                        evidence_id=evidence_id,
                        page_number=page_number,
                        page_count=document.page_count,
                    )
                )
                continue
            if not isinstance(quote, str) or not quote.strip():
                errors.append(_error("missing_quote", "evidence quote is required", evidence_id=evidence_id))
                continue
            page = document.pages[page_number - 1]
            if not _quote_exists(page.text, quote):
                errors.append(
                    _error(
                        "quote_not_found",
                        "evidence quote was not found on the referenced page",
                        evidence_id=evidence_id,
                        artifact_id=artifact_id,
                        page_number=page_number,
                    )
                )

    return evidence_by_id, documents_by_artifact


def _check_evidence_refs(
    value: Any,
    field_name: str,
    known_evidence: set[str],
    errors: list[dict[str, Any]],
    context: dict[str, Any],
    evidence_owner: dict[str, str] | None = None,
) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(_error("invalid_evidence_refs", f"{field_name} must be a list of strings", **context))
        return
    for evidence_id in value:
        if evidence_id not in known_evidence:
            errors.append(
                _error(
                    "unknown_evidence_id",
                    f"{field_name} references an unknown evidence_id",
                    evidence_id=evidence_id,
                    **context,
                )
            )
        elif evidence_owner is not None and context.get("paper_id"):
            owner = evidence_owner.get(evidence_id)
            if owner is not None and owner != context["paper_id"]:
                errors.append(
                    _error(
                        "evidence_owner_mismatch",
                        f"{field_name} references evidence owned by another paper",
                        evidence_id=evidence_id,
                        owner_paper_id=owner,
                        **context,
                    )
                )


def _validate_cross_paper_objects(
    payload: dict[str, Any],
    papers_by_id: dict[str, dict[str, Any]],
    known_evidence: set[str],
    evidence_owner: dict[str, str],
    errors: list[dict[str, Any]],
) -> None:
    paper_ids = set(papers_by_id)

    for paper_id, paper in papers_by_id.items():
        research_question = paper.get("research_question")
        if research_question is not None:
            if not isinstance(research_question, dict):
                errors.append(_error("invalid_research_question", "research_question must be an object", paper_id=paper_id))
            else:
                _check_evidence_refs(
                    research_question.get("evidence_ids"),
                    "research_question.evidence_ids",
                    known_evidence,
                    errors,
                    {"paper_id": paper_id},
                    evidence_owner=evidence_owner,
                )
        claims = paper.get("claims", [])
        if not isinstance(claims, list):
            errors.append(_error("invalid_collection", "paper claims must be a list", paper_id=paper_id))
            continue
        for claim in claims:
            if not isinstance(claim, dict):
                errors.append(_error("invalid_claim", "claim must be an object", paper_id=paper_id))
                continue
            evidence_type = claim.get("evidence_type")
            if evidence_type is not None and evidence_type not in _ALLOWED_EVIDENCE_TYPES:
                errors.append(_error("invalid_evidence_type", "claim evidence_type must be A, B, C or D", paper_id=paper_id))
            _check_evidence_refs(
                claim.get("evidence_ids"),
                "claim.evidence_ids",
                known_evidence,
                errors,
                {"paper_id": paper_id, "claim_id": claim.get("claim_id")},
                evidence_owner=evidence_owner,
            )

        extended_fields = (
            "concepts",
            "theoretical_frameworks",
            "methods",
            "data_samples",
            "contexts",
            "results",
            "limitations",
        )
        for field_name in extended_fields:
            items = paper.get(field_name, [])
            if not isinstance(items, list):
                errors.append(_error("invalid_collection", f"paper {field_name} must be a list", paper_id=paper_id))
                continue
            for item_index, item in enumerate(items):
                if not isinstance(item, dict):
                    errors.append(_error("invalid_item", f"{field_name} item must be an object", paper_id=paper_id, index=item_index))
                    continue
                _check_evidence_refs(
                    item.get("evidence_ids"),
                    f"{field_name}.evidence_ids",
                    known_evidence,
                    errors,
                    {"paper_id": paper_id, "field": field_name, "index": item_index},
                    evidence_owner=evidence_owner,
                )

    directions = payload.get("directions", [])
    seen_direction_ids: set[str] = set()
    if not isinstance(directions, list):
        errors.append(_error("invalid_collection", "directions must be a list", field="directions"))
    else:
        for direction in directions:
            if not isinstance(direction, dict):
                errors.append(_error("invalid_direction", "direction must be an object"))
                continue
            direction_id = direction.get("direction_id")
            if direction_id in seen_direction_ids:
                errors.append(_error("duplicate_direction_id", "direction_id is repeated", direction_id=direction_id))
            seen_direction_ids.add(direction_id)
            members = direction.get("paper_ids", [])
            if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
                errors.append(_error("invalid_direction_members", "direction paper_ids must be a list of strings", direction_id=direction_id))
            else:
                for paper_id in members:
                    if paper_id not in paper_ids:
                        errors.append(_error("unknown_paper_id", "direction references an unknown paper_id", direction_id=direction_id, paper_id=paper_id))
            _check_evidence_refs(direction.get("evidence_ids"), "direction.evidence_ids", known_evidence, errors, {"direction_id": direction_id})

    relations = payload.get("relations", [])
    if not isinstance(relations, list):
        errors.append(_error("invalid_collection", "relations must be a list", field="relations"))
    else:
        for relation in relations:
            if not isinstance(relation, dict):
                errors.append(_error("invalid_relation", "relation must be an object"))
                continue
            relation_type = relation.get("relation_type") or relation.get("type")
            if relation_type not in _ALLOWED_RELATIONS:
                errors.append(_error("invalid_relation_type", "relation type is not supported", relation_type=relation_type))
            for field in ("source_paper_id", "target_paper_id"):
                value = relation.get(field)
                if value is not None and value not in paper_ids:
                    errors.append(_error("unknown_paper_id", f"{field} is not in the corpus", field=field, paper_id=value))
            _check_evidence_refs(relation.get("evidence_ids"), "relation.evidence_ids", known_evidence, errors, {"relation_id": relation.get("relation_id")})

    for collection_name in ("controversies", "frontier_candidates"):
        collection = payload.get(collection_name, [])
        if not isinstance(collection, list):
            errors.append(_error("invalid_collection", f"{collection_name} must be a list", field=collection_name))
            continue
        for item in collection:
            if not isinstance(item, dict):
                errors.append(_error("invalid_item", f"{collection_name} item must be an object", field=collection_name))
                continue
            _check_evidence_refs(item.get("evidence_ids"), f"{collection_name}.evidence_ids", known_evidence, errors, {"item_id": item.get("id")})
            cited_papers = item.get("paper_ids", [])
            if isinstance(cited_papers, list):
                for paper_id in cited_papers:
                    if paper_id not in paper_ids:
                        errors.append(_error("unknown_paper_id", f"{collection_name} references an unknown paper_id", paper_id=paper_id))


def validate_analysis_payload(snapshot: CorpusSnapshot, payload: Any) -> ValidationResult:
    errors: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return ValidationResult(False, {}, [_error("invalid_payload", "analysis payload must be an object")])

    if payload.get("schema_version") != "0.1":
        errors.append(_error("unsupported_schema_version", "analysis payload schema_version must be 0.1"))
    if payload.get("snapshot_id") != snapshot.snapshot_id:
        errors.append(
            _error(
                "snapshot_mismatch",
                "analysis payload must target the exact corpus snapshot",
                expected=snapshot.snapshot_id,
                received=payload.get("snapshot_id"),
            )
        )

    papers = _required_list(payload, "papers", errors)
    evidence_by_id, _ = _validate_evidence(snapshot, papers, errors)
    evidence_owner = {
        evidence["evidence_id"]: paper["paper_id"]
        for paper in papers
        if isinstance(paper, dict) and isinstance(paper.get("paper_id"), str)
        for evidence in paper.get("evidence", [])
        if isinstance(evidence, dict) and isinstance(evidence.get("evidence_id"), str)
    }
    papers_by_id = {
        paper["paper_id"]: paper
        for paper in papers
        if isinstance(paper, dict) and isinstance(paper.get("paper_id"), str)
    }
    _validate_cross_paper_objects(payload, papers_by_id, set(evidence_by_id), evidence_owner, errors)
    return ValidationResult(valid=not errors, payload=payload, errors=errors)
