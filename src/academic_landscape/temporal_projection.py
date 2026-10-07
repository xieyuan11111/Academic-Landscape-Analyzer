"""Temporal projection of EvolutionEvents into an isolated Neo4j view.

Naming / scope clarification (read this first)
----------------------------------------------

This module is **this project's own, hand-written temporal projection layer**.
It *implements* Graphiti-style bi-temporal semantics — ``valid_at`` /
``invalid_at`` bounds, ``EVOLVES_TO`` temporal edges and time-cutoff queries —
but it is **not** the open-source ``getzep/graphiti`` library: nothing here
calls that package's API. Everything below is plain Cypher issued through the
**official Neo4j Python driver**.

The real ``graphiti-core`` package (pinned 0.30.2, optional ``[graphiti]`` extra)
is installed and used elsewhere — see ``graphiti_projection.py`` (dependency-lazy
direct-bulk writer) and ``graphiti_bulk_runtime.py``. This module and that one
are deliberately separate: the Graphiti path writes through
``add_nodes_and_edges_bulk`` so prevalidated evidence is never re-resolved by
LLM, while this module is plain deterministic Cypher. They are compared
against each other by ``runtime/evaluate_temporal_parity.py``.

Historical note: this file used to be named ``graphiti_adapter.py``, with
classes ``GraphitiProjector`` / ``GraphitiProjectionResult`` /
``GraphitiProjectionError``. Those names were misleading (they suggested a
dependency on the getzep library that never existed) and were renamed on
2026-09-24.

This is the **temporal track** of the two-track architecture (see
``docs/EVOLUTION_EVENT_CONTRACT_v0.1.md`` §1 and
``docs/GRAPHITI_INTEGRATION_DESIGN_v0.1.md`` §4). It receives a sequence of
already-validated ``EvolutionEvent`` objects and projects them into the
**project Neo4j** under a caller-chosen ``namespace`` and ``snapshot_id``.

Distinct from ``graph_projector``:

* Owns its own node label ``EvolutionEventNode`` (does not collide with
  ``AcademicLandscapeNode``). Both labels can coexist on the same
  Neo4j database without polluting each other.
* Provides a time-cutoff read path (``search_history``) that follows the
  ``valid_at <= t AND (invalid_at == "" or "open" or invalid_at > t)``
  semantics defined in contract §8.
* Builds ``EVOLVES_TO`` temporal edges between consecutive events on the
  same subject (ordered by ``year`` then ``event_id``). These edges are
  the carrier of "sorted by time" and are 100 % deterministic code —
  no LLM involvement.

Fail-soft boundary
------------------

This module is a *replaceable* projection layer. The adapter **does not**
swallow projection errors: a Neo4j failure (connection, constraint,
malformed Cypher) propagates to the caller, who decides whether to wrap
the call in ``try/except``. The M4/M5 main flow is expected to remain
operational even when this module is unavailable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from neo4j import GraphDatabase

from .timeline_events import EvolutionEvent


class TemporalProjectionError(RuntimeError):
    """Raised when a temporal projection cannot be safely committed.

    Examples: ``event_without_evidence``, ``unknown_evidence_id``.
    Errors are *not* swallowed — callers decide fail-soft behavior.
    """


@dataclass(frozen=True)
class TemporalProjectionResult:
    snapshot_id: str
    event_count: int
    edge_count: int
    analysis_version: str
    projection_version: str


# Node label is intentionally distinct from ``AcademicLandscapeNode`` so
# the temporal track never bleeds into the M4 snapshot projection.
_NODE_LABEL = "EvolutionEventNode"
_EDGE_LABEL = "EVOLVES_TO"

# Namespace shape: 3-64 chars, must start with a letter, only
# alphanumeric + underscore. Mirrors ``graph_projector.Neo4jProjector``.
_NAMESPACE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,63}")


class TemporalProjector:
    """Project a sequence of ``EvolutionEvent`` objects into Neo4j.

    The projector is the projection-layer boundary for the temporal track.
    It is *not* responsible for any of the following (delegated upstream):

    * validating event vocabulary, evidence ownership, year consistency;
    * merging duplicates across batches (handled by MERGE on ``event_id``);
    * LLM-side classification of event types.

    What it *does* enforce at the adapter boundary:

    * ``evidence_ids`` must be non-empty (refuses ``event_without_evidence``);
    * ``namespace`` shape must match the documented regex;
    * every node carries ``namespace`` + ``snapshot_id`` so deletion is
      snapshot-scoped and never bleeds into a foreign snapshot.
    """

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str] | None,
        namespace: str,
        *,
        database: str = "neo4j",
        analysis_version: str = "analysis-v0.1",
        projection_version: str = "temporal-projection-v0.1",
        driver: Any | None = None,
    ) -> None:
        if not uri.strip():
            raise ValueError("uri must not be empty")
        if not _NAMESPACE_RE.fullmatch(namespace):
            raise ValueError(
                "namespace must be 3-64 alphanumeric characters/underscores, "
                "starting with a letter"
            )
        self.uri = uri
        self.auth = auth
        self.namespace = namespace
        self.database = database
        self.analysis_version = analysis_version
        self.projection_version = projection_version
        if driver is None:
            driver = GraphDatabase.driver(uri, auth=auth)
        # Match graph_projector: always ``True`` (the driver is closed on
        # ``close()`` regardless of whether the caller injected one).
        self._owns_driver = True
        self._driver = driver

    # ------------------------------------------------------------------
    # Internal helpers — scoped ids, base property dicts, Cypher runners
    # ------------------------------------------------------------------

    def _scoped_event_id(self, snapshot_id: str, event_id: str) -> str:
        """Globally-unique node id, namespaced + snapshot-scoped."""
        return f"{self.namespace}|{snapshot_id}|evo|{event_id}"

    def _scoped_edge_id(
        self, snapshot_id: str, src_event_id: str, dst_event_id: str
    ) -> str:
        """Globally-unique edge id, namespaced + snapshot-scoped."""
        return (
            f"{self.namespace}|{snapshot_id}|evo_chain|"
            f"{src_event_id}->{dst_event_id}"
        )

    def _base_properties(self, snapshot_id: str) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "snapshot_id": snapshot_id,
            "analysis_version": self.analysis_version,
            "projection_version": self.projection_version,
        }

    def _event_properties(
        self, snapshot_id: str, event: EvolutionEvent
    ) -> dict[str, Any]:
        """Per-event property dict for the node ``SET n += $properties``."""
        base = self._base_properties(snapshot_id)
        return {
            **base,
            "node_id": self._scoped_event_id(snapshot_id, event.event_id),
            "event_id": event.event_id,
            "event_type": event.event_type,
            "subject": event.subject,
            "paper_id": event.paper_id,
            "source_paper_id": event.source_paper_id,
            "year": int(event.year),
            "valid_at": event.valid_at,
            "invalid_at": event.invalid_at,
            "evidence_ids": list(event.evidence_ids),
            "statement": event.statement,
            "difference_dimension": event.difference_dimension,
        }

    @staticmethod
    def _merge_event_node(
        tx: Any, node_id: str, properties: Mapping[str, Any]
    ) -> None:
        """MERGE-by-node_id upsert of a single event node."""
        tx.run(
            f"""
            MERGE (n:{_NODE_LABEL} {{node_id: $node_id}})
            SET n += $properties
            """,
            namespace=properties["namespace"],
            node_id=node_id,
            properties=dict(properties),
        )

    @staticmethod
    def _merge_evolves_to(
        tx: Any,
        edge_id: str,
        source_id: str,
        target_id: str,
        properties: Mapping[str, Any],
    ) -> None:
        """MERGE-by-edge_id upsert of an EVOLVES_TO temporal edge."""
        tx.run(
            f"""
            MATCH (a:{_NODE_LABEL} {{node_id: $source_id}})
            MATCH (b:{_NODE_LABEL} {{node_id: $target_id}})
            MERGE (a)-[r:{_EDGE_LABEL} {{edge_id: $edge_id}}]->(b)
            SET r += $properties
            """,
            namespace=properties["namespace"],
            snapshot_id=properties["snapshot_id"],
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            properties=dict(properties),
        )

    # ------------------------------------------------------------------
    # Chain construction — deterministic, year-ascending per subject
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chain(
        events: Sequence[EvolutionEvent],
    ) -> list[tuple[EvolutionEvent, EvolutionEvent, int]]:
        """Return ``(a, b, gap_years)`` triples for adjacent events.

        Order: ``(year ASC, event_id ASC)`` per subject. ``event_id`` is
        the deterministic tiebreaker so re-runs are stable.
        """
        by_subject: dict[str, list[EvolutionEvent]] = {}
        for event in events:
            by_subject.setdefault(event.subject, []).append(event)
        for items in by_subject.values():
            items.sort(key=lambda e: (int(e.year), e.event_id))
        pairs: list[tuple[EvolutionEvent, EvolutionEvent, int]] = []
        for items in by_subject.values():
            for prev, curr in zip(items, items[1:]):
                pairs.append((prev, curr, int(curr.year) - int(prev.year)))
        return pairs

    # ------------------------------------------------------------------
    # Write path — projection inside one Neo4j transaction
    # ------------------------------------------------------------------

    def _write_projection(
        self,
        tx: Any,
        snapshot_id: str,
        events: list[EvolutionEvent],
    ) -> TemporalProjectionResult:
        seen_ids: set[str] = set()
        unique_count = 0
        for event in events:
            # Fail-closed: an evidence-less event is not allowed to land
            # in the temporal projection (would orphan a "deterministic
            # fact edge" with no EvidenceSpan to back it).
            if not event.evidence_ids:
                raise TemporalProjectionError(
                    f"event_without_evidence: {event.event_id}"
                )
            if event.event_id in seen_ids:
                # Within-batch dedupe — caller may pass the same event_id
                # more than once; we count it once and write the node once.
                continue
            seen_ids.add(event.event_id)
            unique_count += 1
            properties = self._event_properties(snapshot_id, event)
            self._merge_event_node(tx, properties["node_id"], properties)

        edge_ids: set[str] = set()
        for prev, curr, gap in self._build_chain(events):
            src = self._scoped_event_id(snapshot_id, prev.event_id)
            dst = self._scoped_event_id(snapshot_id, curr.event_id)
            edge_id = self._scoped_edge_id(snapshot_id, prev.event_id, curr.event_id)
            edge_properties: dict[str, Any] = {
                **self._base_properties(snapshot_id),
                "edge_id": edge_id,
                "subject": prev.subject,
                "gap_years": gap,
            }
            self._merge_evolves_to(tx, edge_id, src, dst, edge_properties)
            edge_ids.add(edge_id)

        return TemporalProjectionResult(
            snapshot_id=snapshot_id,
            event_count=unique_count,
            edge_count=len(edge_ids),
            analysis_version=self.analysis_version,
            projection_version=self.projection_version,
        )

    def project_events(
        self, snapshot_id: str, events: Sequence[EvolutionEvent]
    ) -> TemporalProjectionResult:
        """Project a sequence of ``EvolutionEvent`` into Neo4j.

        Idempotent on ``event_id``: re-running the same events produces
        the same graph (MERGE-by-node_id + MERGE-by-edge_id). The
        returned ``event_count`` is the count of *unique* events in the
        input batch (intra-batch duplicates are silently deduped).
        """
        with self._driver.session(database=self.database) as session:
            return session.execute_write(
                self._write_projection, snapshot_id, list(events)
            )

    # ------------------------------------------------------------------
    # Read path — temporal cutoff queries
    # ------------------------------------------------------------------

    _RETURN_COLUMNS = (
        "n.event_id AS event_id, "
        "n.event_type AS event_type, "
        "n.subject AS subject, "
        "n.paper_id AS paper_id, "
        "n.source_paper_id AS source_paper_id, "
        "n.year AS year, "
        "n.valid_at AS valid_at, "
        "n.invalid_at AS invalid_at, "
        "n.evidence_ids AS evidence_ids, "
        "n.statement AS statement, "
        "n.difference_dimension AS difference_dimension"
    )

    def search_history(
        self,
        snapshot_id: str,
        *,
        valid_at: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return events that are *valid at time t*, sorted by year ASC.

        Cutoff semantics (contract §8):
            ``valid_at <= t AND (invalid_at == "" or invalid_at == "open"
            or invalid_at > t)``

        ``valid_at=None`` returns all events in the snapshot (still
        sorted by year ASC).
        """
        if limit <= 0:
            raise ValueError("limit must be positive")

        base_match = (
            f"MATCH (n:{_NODE_LABEL} {{namespace: $namespace, "
            f"snapshot_id: $snapshot_id}})"
        )
        order_limit = (
            "ORDER BY n.year ASC, n.event_id ASC LIMIT $limit"
        )

        if valid_at is None:
            query = f"""
            {base_match}
            RETURN {self._RETURN_COLUMNS}
            {order_limit}
            """
            params: dict[str, Any] = {
                "namespace": self.namespace,
                "snapshot_id": snapshot_id,
                "limit": limit,
            }
        else:
            query = f"""
            {base_match}
            WHERE date(CASE WHEN n.valid_at = ''
                            THEN toString(n.year) + '-01-01'
                            ELSE n.valid_at END) <= date($valid_at)
              AND (n.invalid_at = ''
                   OR n.invalid_at = 'open'
                   OR date(n.invalid_at) > date($valid_at))
            RETURN {self._RETURN_COLUMNS}
            {order_limit}
            """
            params = {
                "namespace": self.namespace,
                "snapshot_id": snapshot_id,
                "valid_at": valid_at,
                "limit": limit,
            }

        with self._driver.session(database=self.database) as session:
            rows = session.run(query, **params)
            return [dict(record) for record in rows]

    # ------------------------------------------------------------------
    # Counts and snapshot-scoped deletion
    # ------------------------------------------------------------------

    def count_events(self, snapshot_id: str) -> dict[str, int]:
        """Return ``{"nodes": ..., "relationships": ...}`` for the snapshot."""
        query = f"""
        MATCH (n:{_NODE_LABEL} {{namespace: $namespace, snapshot_id: $snapshot_id}})
        WITH count(n) AS nodes
        CALL () {{
            WITH $namespace AS namespace, $snapshot_id AS snapshot_id
            MATCH ()-[r:{_EDGE_LABEL}]->()
            WHERE r.namespace = namespace AND r.snapshot_id = snapshot_id
            RETURN count(r) AS relationships
        }}
        RETURN nodes, relationships
        """
        with self._driver.session(database=self.database) as session:
            record = session.run(
                query, namespace=self.namespace, snapshot_id=snapshot_id
            ).single()
        if record is None:
            return {"nodes": 0, "relationships": 0}
        return {
            "nodes": int(record["nodes"]),
            "relationships": int(record["relationships"]),
        }

    def delete_snapshot(self, snapshot_id: str) -> None:
        """Detach-delete every node carrying this namespace + snapshot_id.

        Safe to call repeatedly: when nothing matches, ``DETACH DELETE``
        is a no-op. Other snapshots in the same database are untouched.
        """
        query = f"""
        MATCH (n:{_NODE_LABEL} {{namespace: $namespace, snapshot_id: $snapshot_id}})
        DETACH DELETE n
        """
        with self._driver.session(database=self.database) as session:
            session.execute_write(
                lambda tx: tx.run(
                    query, namespace=self.namespace, snapshot_id=snapshot_id
                )
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._owns_driver:
            self._driver.close()

    def __enter__(self) -> "TemporalProjector":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()