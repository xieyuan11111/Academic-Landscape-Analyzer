from pathlib import Path

import fitz
import pytest

from academic_landscape.graph_projector import GraphProjectionError, Neo4jProjector
from academic_landscape.ingest import build_snapshot


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class FakeTx:
    def __init__(self):
        self.calls = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return None


class FakeSession:
    def __init__(self, tx):
        self.tx = tx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute_write(self, callback, *args):
        return callback(self.tx, *args)


class FakeDriver:
    def __init__(self):
        self.tx = FakeTx()
        self.closed = False

    def session(self, *, database):
        assert database == "neo4j"
        return FakeSession(self.tx)

    def close(self):
        self.closed = True


def valid_payload(snapshot, document):
    return {
        "schema_version": "0.1",
        "snapshot_id": snapshot.snapshot_id,
        "papers": [
            {
                "paper_id": document.paper_id,
                "claims": [
                    {
                        "claim_id": "C1",
                        "text": "Method A is evaluated.",
                        "evidence_type": "B",
                        "evidence_ids": ["E1"],
                    }
                ],
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "artifact_id": document.artifact_id,
                        "page_number": 1,
                        "quote": "The paper evaluates method A",
                        "locator": "page 1",
                    }
                ],
            }
        ],
        "directions": [
            {
                "direction_id": "D1",
                "label": "method evaluation",
                "paper_ids": [document.paper_id],
                "evidence_ids": ["E1"],
                "status": "candidate",
            }
        ],
        "relations": [],
        "controversies": [],
        "frontier_candidates": [],
    }


def test_projector_writes_snapshot_scoped_nodes_and_is_idempotent_at_query_level(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The paper evaluates method A.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]
    payload = valid_payload(snapshot, document)
    driver = FakeDriver()
    projector = Neo4jProjector(
        uri="bolt://127.0.0.1:7687",
        auth=("neo4j", "test-only"),
        namespace="academic_landscape_test",
        driver=driver,
    )

    result = projector.project(snapshot, payload)
    projector.project(snapshot, payload)

    assert result.snapshot_id == snapshot.snapshot_id
    assert result.node_count >= 4
    assert result.relationship_count >= 3
    assert all("academic_landscape_test" in params.values() for _, params in driver.tx.calls if params)
    assert sum("MERGE" in query for query, _ in driver.tx.calls) > 0
    assert any("EVIDENCED_BY" in query for query, _ in driver.tx.calls)
    projector.close()
    assert driver.closed is True


def test_projector_writes_allowed_cross_paper_relation_with_evidence(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    make_pdf(first_dir / "a.pdf", "Paper A evaluates method A.")
    make_pdf(second_dir / "b.pdf", "Paper B challenges method A.")
    snapshot = build_snapshot(tmp_path)
    documents = snapshot.included_documents
    first, second = documents
    payload = {
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
                        "quote": "Paper B challenges method A",
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
                "relation_type": "challenges",
                "evidence_ids": ["E2"],
            }
        ],
        "controversies": [],
        "frontier_candidates": [],
    }
    driver = FakeDriver()
    projector = Neo4jProjector(
        uri="bolt://127.0.0.1:7687",
        auth=("neo4j", "test-only"),
        namespace="academic_landscape_test",
        driver=driver,
    )

    result = projector.project(snapshot, payload)

    assert result.relationship_count >= 3
    relation_queries = [query for query, _ in driver.tx.calls if "CHALLENGES" in query]
    assert relation_queries
    assert any(params.get("relation_id") == "R1" for query, params in driver.tx.calls if "CHALLENGES" in query)


def test_projector_rejects_relation_without_evidence(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The paper evaluates method A.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]
    payload = valid_payload(snapshot, document)
    payload["relations"] = [
        {
            "relation_id": "R1",
            "source_paper_id": document.paper_id,
            "target_paper_id": document.paper_id,
            "relation_type": "supports",
            "evidence_ids": [],
        }
    ]
    projector = Neo4jProjector(
        uri="bolt://127.0.0.1:7687",
        auth=("neo4j", "test-only"),
        namespace="academic_landscape_test",
        driver=FakeDriver(),
    )

    with pytest.raises(GraphProjectionError, match="relation_without_evidence"):
        projector.project(snapshot, payload)


def _build_two_paper_relation_payload(
    snapshot,
    relation_id: str,
    relation_type: str,
) -> dict:
    documents = snapshot.included_documents
    first, second = documents
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
                        "quote": "Paper B revises method A",
                    }
                ],
            },
        ],
        "directions": [],
        "relations": [
            {
                "relation_id": relation_id,
                "source_paper_id": first.paper_id,
                "target_paper_id": second.paper_id,
                "relation_type": relation_type,
                "evidence_ids": ["E2"],
            }
        ],
        "controversies": [],
        "frontier_candidates": [],
    }


def test_projector_writes_revises_relation_with_evidence(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    make_pdf(first_dir / "a.pdf", "Paper A evaluates method A.")
    make_pdf(second_dir / "b.pdf", "Paper B revises method A.")
    snapshot = build_snapshot(tmp_path)
    payload = _build_two_paper_relation_payload(
        snapshot, relation_id="R_REVISES", relation_type="revises"
    )
    driver = FakeDriver()
    projector = Neo4jProjector(
        uri="bolt://127.0.0.1:7687",
        auth=("neo4j", "test-only"),
        namespace="academic_landscape_test",
        driver=driver,
    )

    with driver.session(database="neo4j") as session:
        result = session.execute_write(
            projector._write_projection, snapshot, payload
        )

    assert result.relationship_count >= 3
    relation_queries = [query for query, _ in driver.tx.calls if "REVISES" in query]
    assert relation_queries, "expected REVISES edge in Cypher"
    assert any(
        params.get("relation_id") == "R_REVISES"
        for query, params in driver.tx.calls
        if "REVISES" in query
    )


def test_projector_writes_same_name_different_meaning_relation(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    make_pdf(first_dir / "a.pdf", "Paper A evaluates method A.")
    make_pdf(second_dir / "b.pdf", "Paper B redefines method A.")
    snapshot = build_snapshot(tmp_path)
    payload = _build_two_paper_relation_payload(
        snapshot,
        relation_id="R_SNDM",
        relation_type="same_name_different_meaning",
    )
    driver = FakeDriver()
    projector = Neo4jProjector(
        uri="bolt://127.0.0.1:7687",
        auth=("neo4j", "test-only"),
        namespace="academic_landscape_test",
        driver=driver,
    )

    with driver.session(database="neo4j") as session:
        session.execute_write(projector._write_projection, snapshot, payload)

    relation_queries = [
        query for query, _ in driver.tx.calls if "SAME_NAME_DIFFERENT_MEANING" in query
    ]
    assert relation_queries, "expected SAME_NAME_DIFFERENT_MEANING edge in Cypher"
    assert any(
        params.get("relation_id") == "R_SNDM"
        for query, params in driver.tx.calls
        if "SAME_NAME_DIFFERENT_MEANING" in query
    )


def test_projector_writes_surface_conflict_relation(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    make_pdf(first_dir / "a.pdf", "Paper A evaluates method A.")
    make_pdf(second_dir / "b.pdf", "Paper B disagrees superficially.")
    snapshot = build_snapshot(tmp_path)
    payload = _build_two_paper_relation_payload(
        snapshot,
        relation_id="R_SC",
        relation_type="surface_conflict",
    )
    driver = FakeDriver()
    projector = Neo4jProjector(
        uri="bolt://127.0.0.1:7687",
        auth=("neo4j", "test-only"),
        namespace="academic_landscape_test",
        driver=driver,
    )

    with driver.session(database="neo4j") as session:
        session.execute_write(projector._write_projection, snapshot, payload)

    relation_queries = [
        query for query, _ in driver.tx.calls if "SURFACE_CONFLICT" in query
    ]
    assert relation_queries, "expected SURFACE_CONFLICT edge in Cypher"
    assert any(
        params.get("relation_id") == "R_SC"
        for query, params in driver.tx.calls
        if "SURFACE_CONFLICT" in query
    )
