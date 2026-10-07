"""Tests for adapting pipeline artifacts into the Neo4j projection payload."""

from __future__ import annotations

from academic_landscape.direction_naming import ResearchDirection
from academic_landscape.projection_adapter import build_projection_payload
from academic_landscape.relation_extraction import PaperRelation


def _direction() -> ResearchDirection:
    return ResearchDirection(
        direction_id="dir_abc",
        cluster_id=2,
        label="Nitrogen removal",
        description="desc",
        status="emerging",
        paper_ids=("paper_a", "paper_b"),
        item_ids=("i1", "i2"),
    )


def _extraction(paper_id: str, evidence_ids: list[str]) -> dict:
    return {
        "paper_id": paper_id,
        "evidence": [
            {"evidence_id": eid, "artifact_id": f"artifact_{paper_id}",
             "page_number": 1, "quote": f"quote {eid}", "locator": "page 1"}
            for eid in evidence_ids
        ],
    }


def _context() -> tuple[list[ResearchDirection], dict[str, dict]]:
    direction = _direction()
    extractions = {
        "paper_a": _extraction("paper_a", ["E1", "E2"]),
        "paper_b": _extraction("paper_b", ["E1"]),
    }
    return [direction], extractions


def test_payload_binds_schema_and_snapshot() -> None:
    directions, extractions = _context()
    payload = build_projection_payload(
        "snapshot_x", directions, list(extractions.values())
    )
    assert payload["schema_version"] == "0.1"
    assert payload["snapshot_id"] == "snapshot_x"


def test_payload_papers_passthrough() -> None:
    directions, extractions = _context()
    payload = build_projection_payload(
        "snapshot_x", directions, list(extractions.values())
    )
    assert len(payload["papers"]) == 2
    assert payload["papers"][0]["paper_id"] == "paper_a"


def test_paper_evidence_ids_are_globalized() -> None:
    directions, extractions = _context()
    payload = build_projection_payload(
        "snapshot_x", directions, list(extractions.values())
    )
    paper_a = next(p for p in payload["papers"] if p["paper_id"] == "paper_a")
    ids = {e["evidence_id"] for e in paper_a["evidence"]}
    assert ids == {"paper_a::E1", "paper_a::E2"}


def test_direction_mapped_with_global_evidence() -> None:
    directions, extractions = _context()
    payload = build_projection_payload(
        "snapshot_x", directions, list(extractions.values())
    )
    mapped = payload["directions"][0]
    assert mapped["direction_id"] == "dir_abc"
    assert mapped["label"] == "Nitrogen removal"
    assert mapped["status"] == "emerging"
    assert set(mapped["paper_ids"]) == {"paper_a", "paper_b"}
    # global evidence ids aggregated from member papers (no cross-paper collision)
    assert set(mapped["evidence_ids"]) == {
        "paper_a::E1", "paper_a::E2", "paper_b::E1",
    }


def test_relations_and_other_sections_present_empty() -> None:
    directions, extractions = _context()
    payload = build_projection_payload(
        "snapshot_x", directions, list(extractions.values())
    )
    assert payload["relations"] == []
    assert payload["controversies"] == []
    assert payload["frontier_candidates"] == []


def test_only_extractions_that_belong_are_included() -> None:
    direction = _direction()
    extractions = list(_context()[1].values())
    extractions.append(_extraction("paper_stray", ["E9"]))
    payload = build_projection_payload(
        "snapshot_x", [direction], extractions
    )
    paper_ids = {paper["paper_id"] for paper in payload["papers"]}
    assert paper_ids == {"paper_a", "paper_b"}
    assert "paper_stray" not in paper_ids


def _make_relation(
    relation_type: str = "supports",
    source_evidence_ids: list[str] | None = None,
    target_evidence_ids: list[str] | None = None,
) -> PaperRelation:
    return PaperRelation(
        relation_id="rel_abc123",
        source_paper_id="paper_a",
        target_paper_id="paper_b",
        relation_type=relation_type,
        statement="A supports B",
        source_evidence_ids=source_evidence_ids or ["paper_a::E1"],
        target_evidence_ids=target_evidence_ids or [],
        difference_dimension=None,
        snapshot_id="snapshot_x",
    )


def test_payload_paper_relations_default_is_backward_compatible() -> None:
    directions, extractions = _context()
    payload_default = build_projection_payload(
        "snapshot_x", directions, list(extractions.values())
    )
    payload_explicit_none = build_projection_payload(
        "snapshot_x",
        directions,
        list(extractions.values()),
        paper_relations=None,
    )
    assert payload_default == payload_explicit_none
    assert payload_default["relations"] == []


def test_payload_paper_relations_shape_matches_graph_projector() -> None:
    directions, extractions = _context()
    relation = _make_relation(
        relation_type="extends",
        source_evidence_ids=["paper_a::E1"],
    )
    payload = build_projection_payload(
        "snapshot_x",
        directions,
        list(extractions.values()),
        paper_relations=[relation],
    )

    assert len(payload["relations"]) == 1
    rel = payload["relations"][0]
    assert rel["relation_id"] == "rel_abc123"
    assert rel["source_paper_id"] == "paper_a"
    assert rel["target_paper_id"] == "paper_b"
    assert rel["relation_type"] == "extends"
    assert rel["evidence_ids"] == ["paper_a::E1"]


def test_payload_paper_relations_evidence_ids_match_papers_global_form() -> None:
    directions, extractions = _context()
    relation = _make_relation(
        relation_type="supports",
        source_evidence_ids=["paper_a::E1", "paper_a::E2"],
        target_evidence_ids=["paper_b::E1"],
    )
    payload = build_projection_payload(
        "snapshot_x",
        directions,
        list(extractions.values()),
        paper_relations=[relation],
    )

    paper_a = next(p for p in payload["papers"] if p["paper_id"] == "paper_a")
    paper_b = next(p for p in payload["papers"] if p["paper_id"] == "paper_b")
    paper_evidence_ids = (
        {e["evidence_id"] for e in paper_a["evidence"]}
        | {e["evidence_id"] for e in paper_b["evidence"]}
    )
    relation_evidence_ids = set(payload["relations"][0]["evidence_ids"])

    # Global form matches paper.evidence evidence_id form
    assert relation_evidence_ids == paper_evidence_ids
    # No evidence_id on the relation should be in the local (non-globalized) form
    for ev_id in payload["relations"][0]["evidence_ids"]:
        assert "::" in ev_id, f"relation evidence_id {ev_id!r} missing global prefix"


def test_payload_paper_relations_no_double_prefix() -> None:
    directions, extractions = _context()
    # Inputs already in global form (per PaperRelation contract)
    relation = _make_relation(
        source_evidence_ids=["paper_a::E1", "paper_b::E1"],
    )
    payload = build_projection_payload(
        "snapshot_x",
        directions,
        list(extractions.values()),
        paper_relations=[relation],
    )

    relation_ids = payload["relations"][0]["evidence_ids"]
    # No double-prefix like "paper_a::paper_a::E1"
    for ev_id in relation_ids:
        assert not ev_id.startswith("paper_a::paper_a::")
        assert not ev_id.startswith("paper_b::paper_b::")
    # Original global IDs preserved verbatim
    assert set(relation_ids) == {"paper_a::E1", "paper_b::E1"}


def test_payload_paper_relations_accepts_dict_form() -> None:
    directions, extractions = _context()
    relation_dict = {
        "relation_id": "rel_dict01",
        "source_paper_id": "paper_a",
        "target_paper_id": "paper_b",
        "relation_type": "qualifies",
        "statement": "A qualifies B",
        "source_evidence_ids": ["paper_a::E2"],
        "target_evidence_ids": [],
        "snapshot_id": "snapshot_x",
    }
    payload = build_projection_payload(
        "snapshot_x",
        directions,
        list(extractions.values()),
        paper_relations=[relation_dict],
    )

    assert len(payload["relations"]) == 1
    rel = payload["relations"][0]
    assert rel["relation_id"] == "rel_dict01"
    assert rel["relation_type"] == "qualifies"
    assert rel["evidence_ids"] == ["paper_a::E2"]


def test_payload_paper_relations_dedupes_by_relation_id() -> None:
    directions, extractions = _context()
    rel = _make_relation()
    duplicate = _make_relation()
    payload = build_projection_payload(
        "snapshot_x",
        directions,
        list(extractions.values()),
        paper_relations=[rel, duplicate],
    )
    # relation_id is deterministic, so duplicates collapse
    relation_ids = [r["relation_id"] for r in payload["relations"]]
    assert relation_ids == ["rel_abc123"]
