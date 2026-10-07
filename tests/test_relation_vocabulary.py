"""Invariant guards for the FR-04 relation vocabulary.

Why this file exists
--------------------
The first real MAG-50 production run aborted at graph projection with
``invalid_relation_type``: ``build_relation_prompt`` offers the model 12
relation types, but the payload validator accepted only 9. The projector's
existing tests all called the internal ``_write_projection`` and therefore
skipped payload validation entirely, so 311 green unit tests never caught it.

These tests exercise the *public* path (``validate_analysis_payload`` and
``Neo4jProjector.project``) for every relation type, and pin every consumer of
the vocabulary to one shared source of truth.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import fitz
import pytest

from academic_landscape.graph_projector import _RELATION_TYPES, Neo4jProjector
from academic_landscape.graph_projector import GraphProjectionError
from academic_landscape.ingest import build_snapshot
from academic_landscape.prompts import build_corpus_prompt
from academic_landscape.relation_extraction import (
    RELATION_PROMPT_VOCABULARY,
    VALID_RELATION_TYPES,
    build_relation_prompt,
)
from academic_landscape import semantic
from academic_landscape.timeline_events import _NON_EVOLUTION_RELATIONS

_spec = importlib.util.spec_from_file_location(
    "gp_shared",
    Path(__file__).parent / "test_graph_projector.py",
)
gp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gp)

RELATION_TYPES = sorted(VALID_RELATION_TYPES)


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def two_paper_snapshot(tmp_path: Path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    make_pdf(first_dir / "a.pdf", "Paper A evaluates method A.")
    make_pdf(second_dir / "b.pdf", "Paper B compares against method A.")
    snapshot = build_snapshot(tmp_path)
    return snapshot


def relation_payload(snapshot, relation_type: str) -> dict[str, Any]:
    first, second = snapshot.included_documents
    return {
        "schema_version": "0.1",
        "snapshot_id": snapshot.snapshot_id,
        "papers": [
            {
                "paper_id": first.paper_id,
                "claims": [],
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "artifact_id": first.artifact_id,
                        "page_number": 1,
                        "quote": "Paper A evaluates method A",
                    }
                ],
            },
            {
                "paper_id": second.paper_id,
                "claims": [],
                "evidence": [
                    {
                        "evidence_id": "E2",
                        "artifact_id": second.artifact_id,
                        "page_number": 1,
                        "quote": "Paper B compares against method A",
                    }
                ],
            },
        ],
        "directions": [],
        "relations": [
            {
                "relation_id": "R1",
                "source_paper_id": first.paper_id,
                "target_paper_id": second.paper_id,
                "relation_type": relation_type,
                "evidence_ids": ["E2"],
            }
        ],
        "controversies": [],
        "frontier_candidates": [],
    }


# --------------------------------------------------------------------------
# Vocabulary invariants
# --------------------------------------------------------------------------


def test_vocabulary_has_the_twelve_fr04_relation_types() -> None:
    assert len(VALID_RELATION_TYPES) == 12, sorted(VALID_RELATION_TYPES)
    assert {
        "revises",
        "same_name_different_meaning",
        "surface_conflict",
    } <= VALID_RELATION_TYPES


def test_validation_vocabulary_equals_extraction_vocabulary() -> None:
    assert set(semantic._ALLOWED_RELATIONS) == set(VALID_RELATION_TYPES), (
        "payload validator and relation extractor disagree on the vocabulary; "
        f"validator-only={sorted(set(semantic._ALLOWED_RELATIONS) - set(VALID_RELATION_TYPES))} "
        f"extractor-only={sorted(set(VALID_RELATION_TYPES) - set(semantic._ALLOWED_RELATIONS))}"
    )


def test_graph_edge_labels_cover_the_whole_vocabulary() -> None:
    assert set(_RELATION_TYPES) == set(VALID_RELATION_TYPES)
    labels = list(_RELATION_TYPES.values())
    assert all(label.strip() for label in labels)
    assert len(set(labels)) == len(labels), "edge labels must be unique"


def test_non_evolution_relation_set_is_a_subset_of_the_vocabulary() -> None:
    assert set(_NON_EVOLUTION_RELATIONS) <= set(VALID_RELATION_TYPES)


def test_every_relation_type_has_a_decided_temporal_behaviour() -> None:
    """The event map and the non-event set must partition the whole vocabulary.

    A relation type in neither would be silently dropped from the temporal
    track; adding a 13th type must force an explicit decision here.
    """
    from academic_landscape.timeline_events import RELATION_TO_EVENT

    event_types = set(RELATION_TO_EVENT)
    non_events = set(_NON_EVOLUTION_RELATIONS)

    assert not event_types & non_events, sorted(event_types & non_events)
    unclassified = set(VALID_RELATION_TYPES) - event_types - non_events
    assert not unclassified, f"unclassified for the timeline: {sorted(unclassified)}"


def test_gate_requirement_sets_stay_inside_the_vocabulary() -> None:
    from academic_landscape.relation_extraction import (
        RELATIONS_REQUIRING_DIFFERENCE_DIMENSION,
        RELATIONS_REQUIRING_TARGET_EVIDENCE,
    )

    dual_evidence = set(RELATIONS_REQUIRING_TARGET_EVIDENCE)
    needs_dimension = set(RELATIONS_REQUIRING_DIFFERENCE_DIMENSION)
    assert dual_evidence <= set(VALID_RELATION_TYPES)
    assert needs_dimension <= set(VALID_RELATION_TYPES)
    assert not dual_evidence & needs_dimension


def test_every_consumer_shares_the_canonical_object() -> None:
    """Equality is not enough: consumers must hold the *same* object.

    A future re-declaration of an identical copy would satisfy an equality
    assertion while reintroducing exactly the drift that caused the outage.
    """
    from academic_landscape import relation_extraction, relation_types

    assert relation_extraction.VALID_RELATION_TYPES is relation_types.VALID_RELATION_TYPES
    assert semantic._ALLOWED_RELATIONS is relation_types.VALID_RELATION_TYPES
    assert _RELATION_TYPES is relation_types.RELATION_EDGE_LABELS
    assert RELATION_PROMPT_VOCABULARY == "|".join(
        relation_types.RELATION_TYPES_IN_PROMPT_ORDER
    )


def test_relation_prompt_offers_every_relation_type() -> None:
    _system, user_prompt = build_relation_prompt("direction label", "PAPER DETAIL")
    for relation_type in RELATION_TYPES:
        assert relation_type in user_prompt, f"{relation_type} missing from relation prompt"


def test_legacy_corpus_prompt_offers_every_relation_type(tmp_path: Path) -> None:
    snapshot = two_paper_snapshot(tmp_path)
    system, user = build_corpus_prompt(snapshot, [])
    text = f"{system}\n{user}"
    for relation_type in RELATION_TYPES:
        assert relation_type in text, f"{relation_type} missing from corpus prompt"


# --------------------------------------------------------------------------
# Behavioral regression: the public path must accept every allowed type
# --------------------------------------------------------------------------


def test_payload_validator_accepts_every_relation_type(tmp_path: Path) -> None:
    snapshot = two_paper_snapshot(tmp_path)
    rejected: dict[str, list[Any]] = {}
    for relation_type in RELATION_TYPES:
        payload = relation_payload(snapshot, relation_type)
        result = semantic.validate_analysis_payload(snapshot, payload)
        bad = [
            error
            for error in result.errors
            if error.get("code") == "invalid_relation_type"
        ]
        if bad:
            rejected[relation_type] = bad
    assert not rejected, f"validator rejected allowed relation types: {sorted(rejected)}"


def test_projector_public_path_accepts_every_relation_type(tmp_path: Path) -> None:
    """Reproduces the real MAG-50 crash: project() each type on the public path."""
    snapshot = two_paper_snapshot(tmp_path)
    failed: dict[str, str] = {}
    for relation_type in RELATION_TYPES:
        payload = relation_payload(snapshot, relation_type)
        projector = Neo4jProjector(
            uri="bolt://127.0.0.1:7687",
            auth=("neo4j", "test-only"),
            namespace="academic_landscape_test",
            driver=gp.FakeDriver(),
        )
        try:
            projector.project(snapshot, payload)
        except GraphProjectionError as exc:
            failed[relation_type] = str(exc)
    assert not failed, f"project() rejected allowed relation types: {failed}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
