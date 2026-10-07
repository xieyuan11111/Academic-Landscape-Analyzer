"""Contract tests for the incremental Graphiti projection seam."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from academic_landscape.engine_dual_run import (
    AuthorityProjectionError,
    evidence_record_from_span,
    relation_from_authority,
)
from academic_landscape.graphiti_projection import ValidatedRelation
from academic_landscape.sqlite_store import EvidenceSpan


@dataclass
class _Relation:
    relation_id: str = "rel_001"
    source_paper_id: str = "paper_a"
    target_paper_id: str = "paper_b"
    relation_type: str = "supports"
    source_evidence_ids: list[str] | None = None
    target_evidence_ids: list[str] | None = None
    statement: str = "Paper A supports Paper B under validated conditions."
    difference_dimension: str | None = None
    comparability_note: str = "same endpoint"
    uncertainty: str = "limited sample"
    snapshot_id: str = "snapshot_1"

    def __post_init__(self) -> None:
        if self.source_evidence_ids is None:
            self.source_evidence_ids = ["paper_a::E1"]
        if self.target_evidence_ids is None:
            self.target_evidence_ids = ["paper_b::E2"]


class _Authority:
    def __init__(self, rows: dict[str, EvidenceSpan]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, str]] = []

    def get_evidence(self, evidence_id: str, snapshot_id: str) -> EvidenceSpan | None:
        self.calls.append((evidence_id, snapshot_id))
        return self.rows.get(evidence_id)


def _span(
    evidence_id: str = "paper_a::E1",
    *,
    paper_id: str = "paper_a",
    page_number: int = 3,
) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=evidence_id,
        snapshot_id="snapshot_1",
        paper_id=paper_id,
        artifact_id=f"artifact_{paper_id}",
        page_number=page_number,
        quote=f"Validated quote for {evidence_id}.",
        locator="Results",
        evidence_type="A",
    )


def test_evidence_record_contains_only_structured_authority_fields() -> None:
    record = evidence_record_from_span(_span())
    assert record == {
        "snapshot_id": "snapshot_1",
        "paper_id": "paper_a",
        "evidence_id": "paper_a::E1",
        "page_number": 3,
        "quote": "Validated quote for paper_a::E1.",
        "locator": "Results",
        "evidence_type": "A",
        "artifact_id": "artifact_paper_a",
        "namespace": "academic_landscape",
    }
    assert not {"raw_pdf", "pdf_path", "full_text", "raw_document"} & set(record)


def test_evidence_record_rejects_non_global_id() -> None:
    with pytest.raises(AuthorityProjectionError, match="globally scoped"):
        evidence_record_from_span(_span("E1"))


def test_relation_resolves_pages_from_authority_not_llm_claims() -> None:
    authority = _Authority({
        "paper_a::E1": _span("paper_a::E1", paper_id="paper_a", page_number=3),
        "paper_b::E2": _span("paper_b::E2", paper_id="paper_b", page_number=7),
    })
    relation = relation_from_authority(_Relation(), authority, valid_at="2026-01-01T00:00:00+00:00")
    assert isinstance(relation, ValidatedRelation)
    assert relation.evidence_ids == ["paper_a::E1", "paper_b::E2"]
    assert relation.page_numbers == [3, 7]
    assert authority.calls == [
        ("paper_a::E1", "snapshot_1"),
        ("paper_b::E2", "snapshot_1"),
    ]


def test_relation_rejects_absent_or_foreign_authority_evidence() -> None:
    authority = _Authority({"paper_a::E1": _span("paper_a::E1")})
    with pytest.raises(AuthorityProjectionError, match="not found"):
        relation_from_authority(_Relation(), authority, valid_at="2026-01-01T00:00:00+00:00")


def test_relation_rejects_unscoped_input_even_when_store_has_it() -> None:
    relation = _Relation(source_evidence_ids=["E1"])
    authority = _Authority({"E1": _span("E1")})
    with pytest.raises(AuthorityProjectionError, match="globally scoped"):
        relation_from_authority(relation, authority, valid_at="2026-01-01T00:00:00+00:00")


def test_relation_rejects_naive_time_before_graphiti_can_project() -> None:
    authority = _Authority({
        "paper_a::E1": _span("paper_a::E1", paper_id="paper_a"),
        "paper_b::E2": _span("paper_b::E2", paper_id="paper_b"),
    })
    with pytest.raises(AuthorityProjectionError, match="timezone-aware"):
        relation_from_authority(_Relation(), authority, valid_at="2026-01-01")
