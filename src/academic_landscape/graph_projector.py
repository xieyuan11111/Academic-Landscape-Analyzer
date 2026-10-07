"""Minimal, evidence-bound Neo4j projection for validated analysis payloads."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from neo4j import GraphDatabase

from .models import CorpusSnapshot, DocumentRecord
from .relation_types import RELATION_EDGE_LABELS
from .semantic import validate_analysis_payload


class GraphProjectionError(RuntimeError):
    """Raised when a graph projection cannot be safely committed."""


@dataclass(frozen=True)
class GraphProjectionResult:
    snapshot_id: str
    node_count: int
    relationship_count: int
    analysis_version: str
    projection_version: str


_RELATION_TYPES = RELATION_EDGE_LABELS


class Neo4jProjector:
    """Project validated records into an isolated, snapshot-scoped Neo4j graph."""

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str] | None,
        namespace: str,
        *,
        database: str = "neo4j",
        analysis_version: str = "analysis-v0.1",
        projection_version: str = "projection-v0.1",
        driver: Any | None = None,
    ) -> None:
        if not uri.strip():
            raise ValueError("uri must not be empty")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{2,63}", namespace):
            raise ValueError("namespace must be 3-64 alphanumeric characters/underscores")
        self.uri = uri
        self.auth = auth
        self.namespace = namespace
        self.database = database
        self.analysis_version = analysis_version
        self.projection_version = projection_version
        if driver is None:
            driver = GraphDatabase.driver(uri, auth=auth)
        self._owns_driver = True
        self._driver = driver

    def _scoped_id(self, snapshot_id: str, kind: str, value: str) -> str:
        return f"{self.namespace}|{snapshot_id}|{kind}|{value}"

    def _edge_id(self, snapshot_id: str, kind: str, value: str) -> str:
        raw = f"{self.namespace}\0{snapshot_id}\0{kind}\0{value}"
        return f"edge_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"

    def _base_properties(self, snapshot_id: str) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "snapshot_id": snapshot_id,
            "analysis_version": self.analysis_version,
            "projection_version": self.projection_version,
        }

    @staticmethod
    def _merge_node(tx: Any, label: str, node_id: str, properties: Mapping[str, Any]) -> None:
        tx.run(
            f"""
            MERGE (n:AcademicLandscapeNode:{label} {{node_id: $node_id}})
            SET n += $properties
            """,
            namespace=properties["namespace"],
            node_id=node_id,
            properties=dict(properties),
        )

    @staticmethod
    def _merge_edge(
        tx: Any,
        relationship_type: str,
        edge_id: str,
        source_id: str,
        target_id: str,
        properties: Mapping[str, Any],
    ) -> None:
        tx.run(
            f"""
            MATCH (source:AcademicLandscapeNode {{node_id: $source_id}})
            MATCH (target:AcademicLandscapeNode {{node_id: $target_id}})
            MERGE (source)-[r:{relationship_type} {{edge_id: $edge_id}}]->(target)
            SET r += $properties
            """,
            namespace=properties["namespace"],
            edge_id=edge_id,
            relation_id=properties.get("relation_id"),
            source_id=source_id,
            target_id=target_id,
            properties=dict(properties),
        )

    @staticmethod
    def _documents_by_paper(snapshot: CorpusSnapshot) -> dict[str, tuple[DocumentRecord, ...]]:
        result: dict[str, list[DocumentRecord]] = {}
        for document in snapshot.documents:
            result.setdefault(document.paper_id, []).append(document)
        return {paper_id: tuple(documents) for paper_id, documents in result.items()}

    def _write_projection(self, tx: Any, snapshot: CorpusSnapshot, payload: dict[str, Any]) -> GraphProjectionResult:
        snapshot_id = snapshot.snapshot_id
        base = self._base_properties(snapshot_id)
        documents_by_paper = self._documents_by_paper(snapshot)
        documents_by_artifact = {document.artifact_id: document for document in snapshot.documents}
        node_ids: set[str] = set()
        edge_ids: set[str] = set()
        evidence_by_id: dict[str, dict[str, Any]] = {}

        for paper in payload["papers"]:
            paper_id = paper["paper_id"]
            paper_node_id = self._scoped_id(snapshot_id, "paper", paper_id)
            documents = documents_by_paper.get(paper_id, ())
            document = documents[0] if documents else None
            paper_properties = {
                **base,
                "node_id": paper_node_id,
                "paper_id": paper_id,
                "relative_path": document.relative_path if document else None,
                "filename": document.filename if document else None,
                "status": document.status if document else "unknown",
            }
            self._merge_node(tx, "PaperWork", paper_node_id, paper_properties)
            node_ids.add(paper_node_id)

            for document in documents:
                artifact_node_id = self._scoped_id(snapshot_id, "artifact", document.artifact_id)
                artifact_properties = {
                    **base,
                    "node_id": artifact_node_id,
                    "artifact_id": document.artifact_id,
                    "paper_id": paper_id,
                    "sha256": document.sha256,
                    "filename": document.filename,
                }
                self._merge_node(tx, "Artifact", artifact_node_id, artifact_properties)
                node_ids.add(artifact_node_id)
                edge_id = self._edge_id(snapshot_id, "has_artifact", f"{paper_id}:{document.artifact_id}")
                self._merge_edge(tx, "HAS_ARTIFACT", edge_id, paper_node_id, artifact_node_id, base)
                edge_ids.add(edge_id)

            for evidence in paper.get("evidence", []):
                evidence_id = evidence["evidence_id"]
                evidence_by_id[evidence_id] = evidence
                document = documents_by_artifact.get(evidence["artifact_id"])
                evidence_node_id = self._scoped_id(snapshot_id, "evidence", evidence_id)
                evidence_properties = {
                    **base,
                    "node_id": evidence_node_id,
                    "evidence_id": evidence_id,
                    "paper_id": paper_id,
                    "artifact_id": evidence["artifact_id"],
                    "page_number": evidence["page_number"],
                    "quote": evidence["quote"],
                    "locator": evidence.get("locator", ""),
                }
                self._merge_node(tx, "EvidenceSpan", evidence_node_id, evidence_properties)
                node_ids.add(evidence_node_id)
                edge_id = self._edge_id(snapshot_id, "has_evidence", f"{paper_id}:{evidence_id}")
                self._merge_edge(tx, "HAS_EVIDENCE", edge_id, paper_node_id, evidence_node_id, base)
                edge_ids.add(edge_id)

            for claim in paper.get("claims", []):
                evidence_ids = claim.get("evidence_ids", [])
                if not evidence_ids:
                    continue
                claim_key = f"{paper_id}:{claim['claim_id']}"
                claim_node_id = self._scoped_id(snapshot_id, "claim", claim_key)
                claim_properties = {
                    **base,
                    "node_id": claim_node_id,
                    "claim_id": claim["claim_id"],
                    "paper_id": paper_id,
                    "text": claim["text"],
                    "evidence_type": claim["evidence_type"],
                    "evidence_ids": evidence_ids,
                }
                self._merge_node(tx, "Assertion", claim_node_id, claim_properties)
                node_ids.add(claim_node_id)
                edge_id = self._edge_id(snapshot_id, "asserts", claim_key)
                self._merge_edge(tx, "ASSERTS", edge_id, paper_node_id, claim_node_id, base)
                edge_ids.add(edge_id)
                for evidence_id in evidence_ids:
                    if evidence_id not in evidence_by_id:
                        raise GraphProjectionError(f"unknown_evidence_id: {evidence_id}")
                    evidence_node_id = self._scoped_id(snapshot_id, "evidence", evidence_id)
                    edge_id = self._edge_id(snapshot_id, "evidenced_by", f"{claim_key}:{evidence_id}")
                    edge_properties = {**base, "evidence_ids": [evidence_id]}
                    self._merge_edge(tx, "EVIDENCED_BY", edge_id, claim_node_id, evidence_node_id, edge_properties)
                    edge_ids.add(edge_id)

        for direction in payload.get("directions", []):
            evidence_ids = direction.get("evidence_ids", [])
            if not evidence_ids:
                continue
            direction_id = direction["direction_id"]
            direction_node_id = self._scoped_id(snapshot_id, "direction", direction_id)
            direction_properties = {
                **base,
                "node_id": direction_node_id,
                "direction_id": direction_id,
                "label": direction["label"],
                "status": direction.get("status", "candidate"),
                "evidence_ids": evidence_ids,
            }
            self._merge_node(tx, "ResearchDirection", direction_node_id, direction_properties)
            node_ids.add(direction_node_id)
            for paper_id in direction.get("paper_ids", []):
                paper_node_id = self._scoped_id(snapshot_id, "paper", paper_id)
                edge_id = self._edge_id(snapshot_id, "belongs_to_direction", f"{paper_id}:{direction_id}")
                self._merge_edge(
                    tx,
                    "BELONGS_TO_DIRECTION",
                    edge_id,
                    paper_node_id,
                    direction_node_id,
                    {**base, "evidence_ids": evidence_ids},
                )
                edge_ids.add(edge_id)

        for relation in payload.get("relations", []):
            evidence_ids = relation.get("evidence_ids", [])
            if not evidence_ids:
                raise GraphProjectionError("relation_without_evidence")
            relation_type = relation.get("relation_type") or relation.get("type")
            relationship_type = _RELATION_TYPES.get(relation_type)
            if relationship_type is None:
                raise GraphProjectionError(f"unsupported_relation_type: {relation_type}")
            relation_id = relation.get("relation_id") or relation.get("id")
            if not relation_id:
                relation_id = self._edge_id(
                    snapshot_id,
                    "relation",
                    ":".join(
                        [
                            relation["source_paper_id"],
                            relation["target_paper_id"],
                            relation_type,
                            ",".join(sorted(evidence_ids)),
                        ]
                    ),
                )
            source_id = self._scoped_id(snapshot_id, "paper", relation["source_paper_id"])
            target_id = self._scoped_id(snapshot_id, "paper", relation["target_paper_id"])
            edge_id = self._edge_id(snapshot_id, "relation", str(relation_id))
            relation_properties = {
                **base,
                "relation_id": relation_id,
                "relation_type": relation_type,
                "evidence_ids": evidence_ids,
            }
            self._merge_edge(tx, relationship_type, edge_id, source_id, target_id, relation_properties)
            edge_ids.add(edge_id)

        return GraphProjectionResult(
            snapshot_id=snapshot_id,
            node_count=len(node_ids),
            relationship_count=len(edge_ids),
            analysis_version=self.analysis_version,
            projection_version=self.projection_version,
        )

    def project(self, snapshot: CorpusSnapshot, payload: Any) -> GraphProjectionResult:
        validation = validate_analysis_payload(snapshot, payload)
        if not validation.valid:
            codes = ", ".join(sorted({error["code"] for error in validation.errors}))
            raise GraphProjectionError(f"analysis validation failed: {codes}")
        with self._driver.session(database=self.database) as session:
            return session.execute_write(self._write_projection, snapshot, validation.payload)

    def count_snapshot(self, snapshot_id: str) -> dict[str, int]:
        query = """
        MATCH (n:AcademicLandscapeNode {namespace: $namespace, snapshot_id: $snapshot_id})
        WITH count(n) AS nodes
        CALL () {
            WITH $namespace AS namespace, $snapshot_id AS snapshot_id
            MATCH ()-[r]->()
            WHERE r.namespace = namespace AND r.snapshot_id = snapshot_id
            RETURN count(r) AS relationships
        }
        RETURN nodes, relationships
        """
        with self._driver.session(database=self.database) as session:
            record = session.run(query, namespace=self.namespace, snapshot_id=snapshot_id).single()
        if record is None:
            return {"nodes": 0, "relationships": 0}
        return {"nodes": int(record["nodes"]), "relationships": int(record["relationships"])}

    def related_papers(
        self,
        snapshot_id: str,
        paper_ids: list[str] | tuple[str, ...] | set[str],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        if not paper_ids:
            return []
        if limit <= 0:
            raise ValueError("limit must be positive")
        query = """
        MATCH (source:AcademicLandscapeNode {namespace: $namespace, snapshot_id: $snapshot_id})
              -[r]->
              (target:AcademicLandscapeNode {namespace: $namespace, snapshot_id: $snapshot_id})
        WHERE type(r) IN $relation_types
          AND (source.paper_id IN $paper_ids OR target.paper_id IN $paper_ids)
          AND r.evidence_ids IS NOT NULL
          AND size(r.evidence_ids) > 0
        RETURN r.relation_id AS relation_id,
               r.relation_type AS relation_type,
               source.paper_id AS source_paper_id,
               target.paper_id AS target_paper_id,
               r.evidence_ids AS evidence_ids
        ORDER BY r.relation_id
        LIMIT $limit
        """
        with self._driver.session(database=self.database) as session:
            rows = session.run(
                query,
                namespace=self.namespace,
                snapshot_id=snapshot_id,
                paper_ids=sorted(set(paper_ids)),
                relation_types=sorted(set(_RELATION_TYPES.values())),
                limit=limit,
            )
            return [dict(row) for row in rows]

    def delete_snapshot(self, snapshot_id: str) -> None:
        query = """
        MATCH (n:AcademicLandscapeNode {namespace: $namespace, snapshot_id: $snapshot_id})
        DETACH DELETE n
        """
        with self._driver.session(database=self.database) as session:
            session.execute_write(
                lambda tx: tx.run(query, namespace=self.namespace, snapshot_id=snapshot_id)
            )

    def close(self) -> None:
        if self._owns_driver:
            self._driver.close()

    def __enter__(self) -> "Neo4jProjector":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
