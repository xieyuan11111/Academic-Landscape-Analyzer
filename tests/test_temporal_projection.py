"""Tests for our own temporal projection layer (EvolutionEvents → Neo4j).

Note on naming: ``temporal_projection`` is this project's **self-written**
temporal projection layer. It implements Graphiti-*style* bi-temporal
semantics, but the ``getzep/graphiti`` library is not a dependency and is not
used here. The module was previously called ``graphiti_adapter`` (renamed
2026-09-24) — that name wrongly suggested a getzep dependency.

Mirrors the FakeDriver / FakeSession / FakeTx pattern from
tests/test_graph_projector.py — never touches a real Neo4j.

TDD order: tests written first (RED), then implementation (GREEN).
The tests will skip gracefully when the parallel ``timeline_events``
module is not yet implemented.
"""

from __future__ import annotations

from typing import Any

import pytest

# Skip the entire module if the parallel ``timeline_events`` module has not
# landed yet (parallel agent dependency). When the dependency shows up,
# importing temporal_projection / EvolutionEvent becomes mandatory — and the
# tests will run normally.
try:
    from academic_landscape.timeline_events import EvolutionEvent  # noqa: E402
except ImportError:  # pragma: no cover - parallel dev dependency
    pytest.skip(
        "timeline_events not yet implemented (parallel agent)",
        allow_module_level=True,
    )

# When the test module is collected before the implementation lands, we
# also gracefully skip on missing projection module.
try:
    from academic_landscape.temporal_projection import (  # noqa: E402
        TemporalProjectionError,
        TemporalProjectionResult,
        TemporalProjector,
    )
except ImportError:  # pragma: no cover - TDD RED phase
    pytest.skip(
        "temporal_projection not yet implemented (TDD RED)",
        allow_module_level=True,
    )


SNAPSHOT_ID = "snapshot_gti_test"
NAMESPACE = "academic_landscape_gti_test"


# ---------------------------------------------------------------------------
# Fake driver / session / result — mirrors tests/test_graph_projector.py
# ---------------------------------------------------------------------------


class FakeTx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, **params: Any) -> None:
        self.calls.append((query, params))
        return None


class FakeResult:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = list(rows or [])

    def single(self) -> dict[str, Any] | None:
        return self._rows[0] if self._rows else None

    def __iter__(self):  # noqa: D401
        return iter(self._rows)


class FakeSession:
    """A session that exposes both write (tx-callback) and read (run) hooks.

    The first ``session.run(...)`` call consumes the next list from
    ``read_results`` (queue-style); subsequent calls return empty results.
    All ``session.run(...)`` calls are recorded for assertions.
    """

    def __init__(
        self,
        tx: FakeTx,
        read_results: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.tx = tx
        self._read_results = list(read_results or [])
        self.run_calls: list[tuple[str, dict[str, Any]]] = []

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def execute_write(self, callback, *args):
        return callback(self.tx, *args)

    def run(self, query: str, **params: Any) -> FakeResult:
        self.run_calls.append((query, params))
        if self._read_results:
            return FakeResult(self._read_results.pop(0))
        return FakeResult([])


class FakeDriver:
    def __init__(
        self,
        *,
        read_results: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.tx = FakeTx()
        self._read_results = read_results
        self.closed = False
        self.sessions: list[FakeSession] = []

    def session(self, *, database: str) -> FakeSession:
        assert database == "neo4j"
        session = FakeSession(self.tx, read_results=self._read_results)
        self.sessions.append(session)
        return session

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(
    *,
    event_id: str,
    event_type: str,
    subject: str,
    paper_id: str,
    year: int,
    evidence_ids: tuple[str, ...] = ("paper_a::E1",),
    source_paper_id: str = "",
    valid_at: str = "",
    invalid_at: str = "",
    snapshot_id: str = SNAPSHOT_ID,
    statement: str = "",
    difference_dimension: str | None = None,
) -> EvolutionEvent:
    return EvolutionEvent(
        event_id=event_id,
        event_type=event_type,
        subject=subject,
        paper_id=paper_id,
        source_paper_id=source_paper_id,
        year=year,
        evidence_ids=evidence_ids,
        statement=statement,
        difference_dimension=difference_dimension,
        valid_at=valid_at,
        invalid_at=invalid_at,
        snapshot_id=snapshot_id,
    )


def _projector(driver: FakeDriver | None = None) -> TemporalProjector:
    return TemporalProjector(
        uri="bolt://127.0.0.1:17687",
        auth=("neo4j", "test-only"),
        namespace=NAMESPACE,
        driver=driver or FakeDriver(),
    )


# ---------------------------------------------------------------------------
# 1. project_events writes event nodes + EVOLVES_TO temporal edges
# ---------------------------------------------------------------------------


def test_project_events_writes_event_nodes_and_evolves_to_edges() -> None:
    driver = FakeDriver()
    projector = _projector(driver)
    events = [
        _event(
            event_id="evo_a",
            event_type="explicit_proposal",
            subject="concept_x",
            paper_id="paper_a",
            year=2020,
        ),
        _event(
            event_id="evo_b",
            event_type="object_extension",
            subject="concept_x",
            paper_id="paper_b",
            year=2022,
        ),
    ]

    result = projector.project_events(SNAPSHOT_ID, events)

    assert isinstance(result, TemporalProjectionResult)
    assert result.snapshot_id == SNAPSHOT_ID
    assert result.event_count == 2
    assert result.edge_count == 1
    # Node MERGE present.
    node_queries = [q for q, _ in driver.tx.calls if "EvolutionEventNode" in q and "MERGE" in q]
    assert node_queries, "expected EvolutionEventNode MERGE in Cypher"
    # EVOLVES_TO MERGE/CREATE present.
    edge_queries = [q for q, _ in driver.tx.calls if "EVOLVES_TO" in q and "MERGE" in q]
    assert edge_queries, "expected EVOLVES_TO MERGE in Cypher"
    # namespace + snapshot_id propagated on every call.
    for _, params in driver.tx.calls:
        assert params.get("namespace") == NAMESPACE
    # All writes scoped to the right snapshot.
    for _, params in driver.tx.calls:
        assert params.get("snapshot_id") == SNAPSHOT_ID or "snapshot_id" in params.get(
            "properties", {}
        )


# ---------------------------------------------------------------------------
# 2. Idempotent: re-projecting the same events does not duplicate
# ---------------------------------------------------------------------------


def test_project_events_is_idempotent_under_reprojection() -> None:
    driver = FakeDriver()
    projector = _projector(driver)
    events = [
        _event(
            event_id="evo_a",
            event_type="explicit_proposal",
            subject="concept_x",
            paper_id="paper_a",
            year=2020,
        ),
        _event(
            event_id="evo_b",
            event_type="object_extension",
            subject="concept_x",
            paper_id="paper_b",
            year=2022,
        ),
    ]

    first = projector.project_events(SNAPSHOT_ID, events)
    calls_after_first = len(driver.tx.calls)
    second = projector.project_events(SNAPSHOT_ID, events)

    # Counts are identical (event_count + edge_count unchanged).
    assert first.event_count == second.event_count
    assert first.edge_count == second.edge_count
    # MERGE used throughout — no CREATE on nodes/edges.
    assert all("MERGE" in q for q, _ in driver.tx.calls)
    # Second projection does still touch the DB (so dedupe is via MERGE,
    # not skipped re-execution).
    assert len(driver.tx.calls) > calls_after_first


# ---------------------------------------------------------------------------
# 3. event_without_evidence is rejected (fail-closed at adapter boundary)
# ---------------------------------------------------------------------------


def test_project_events_rejects_event_without_evidence() -> None:
    projector = _projector()
    bad = _event(
        event_id="evo_no_ev",
        event_type="explicit_proposal",
        subject="concept_x",
        paper_id="paper_a",
        year=2020,
        evidence_ids=(),
    )

    with pytest.raises(TemporalProjectionError, match="event_without_evidence"):
        projector.project_events(SNAPSHOT_ID, [bad])


# ---------------------------------------------------------------------------
# 4. search_history cutoff semantics (valid_at <= t AND not invalidated)
# ---------------------------------------------------------------------------


def test_search_history_cutoff_filters_by_valid_at_and_invalid_at() -> None:
    """The Cypher issued to Neo4j must express the cutoff clause verbatim."""
    rows = [
        {
            "event_id": "evo_a",
            "event_type": "explicit_proposal",
            "subject": "concept_x",
            "paper_id": "paper_a",
            "source_paper_id": "",
            "year": 2020,
            "valid_at": "2020-01-01",
            "invalid_at": "",
            "evidence_ids": ["paper_a::E1"],
            "statement": "",
            "difference_dimension": None,
        },
    ]
    driver = FakeDriver(read_results=[rows])
    projector = _projector(driver)

    out = projector.search_history(SNAPSHOT_ID, valid_at="2021-06-01", limit=10)

    assert out == rows
    # Find the first read call recorded by any FakeSession.run().
    read_query, read_params = _find_read_call(driver)
    assert "valid_at" in read_query
    assert "invalid_at" in read_query
    assert read_params.get("valid_at") == "2021-06-01"
    assert read_params.get("namespace") == NAMESPACE
    assert read_params.get("snapshot_id") == SNAPSHOT_ID
    assert read_params.get("limit") == 10
    # Ordering is by year ASC.
    assert "ORDER BY" in read_query
    assert "year" in read_query


def test_search_history_empty_valid_at_uses_event_year_fallback() -> None:
    """The stored empty value must match filter_by_cutoff's year fallback."""
    driver = FakeDriver(read_results=[[]])
    projector = _projector(driver)
    projector.search_history(SNAPSHOT_ID, valid_at="2021-06-01", limit=10)
    read_query, _ = _find_read_call(driver)
    assert "toString(n.year) + '-01-01'" in read_query
    assert "date(" in read_query


def test_search_history_without_cutoff_returns_all_events_year_asc() -> None:
    driver = FakeDriver(read_results=[[]])
    projector = _projector(driver)

    projector.search_history(SNAPSHOT_ID, valid_at=None, limit=25)

    read_query, read_params = _find_read_call(driver)
    # No cutoff filter when valid_at is None.
    assert "$valid_at" not in read_query
    # Still sorted by year ascending.
    assert "ORDER BY" in read_query and "year" in read_query
    assert read_params.get("limit") == 25


def _find_read_call(driver: FakeDriver) -> tuple[str, dict[str, Any]]:
    """Return the first (query, params) recorded by any FakeSession.run()."""
    for session in driver.sessions:
        if session.run_calls:
            return session.run_calls[0]
    raise AssertionError("no read call recorded")


# ---------------------------------------------------------------------------
# 5. delete_snapshot only deletes the specified snapshot (namespace+id filter)
# ---------------------------------------------------------------------------


def test_delete_snapshot_is_scoped_to_namespace_and_snapshot() -> None:
    driver = FakeDriver()
    projector = _projector(driver)

    projector.delete_snapshot(SNAPSHOT_ID)

    delete_call = next((c for c in driver.tx.calls if "DETACH DELETE" in c[0]), None)
    assert delete_call is not None, "expected DETACH DELETE in Cypher"
    _, dparams = delete_call
    assert dparams.get("namespace") == NAMESPACE
    assert dparams.get("snapshot_id") == SNAPSHOT_ID
    # Also confirms EvolutionEventNode label isolation.
    assert "EvolutionEventNode" in delete_call[0]
    assert "AcademicLandscapeNode" not in delete_call[0]


# ---------------------------------------------------------------------------
# 6. namespace with invalid shape is rejected at construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_namespace",
    [
        "",  # empty
        "ab",  # too short (< 3)
        "1abc",  # starts with digit
        "a-b",  # dash
        "a" * 65,  # too long
    ],
)
def test_invalid_namespace_shape_is_rejected(bad_namespace: str) -> None:
    with pytest.raises(ValueError):
        TemporalProjector(
            uri="bolt://127.0.0.1:17687",
            auth=("neo4j", "test-only"),
            namespace=bad_namespace,
            driver=FakeDriver(),
        )


# ---------------------------------------------------------------------------
# 7. Same-subject events form a year-ascending chain via EVOLVES_TO
# ---------------------------------------------------------------------------


def test_same_subject_events_form_year_ascending_chain() -> None:
    """Three events on subject S in years 2020, 2022, 2024 must produce
    exactly two EVOLVES_TO edges in chronological order, regardless of the
    order they were supplied in."""
    driver = FakeDriver()
    projector = _projector(driver)

    # Supply in scrambled order on purpose.
    e_2024 = _event(
        event_id="evo_2024",
        event_type="synthesis_or_alternative",
        subject="concept_x",
        paper_id="paper_d",
        year=2024,
    )
    e_2020 = _event(
        event_id="evo_2020",
        event_type="concept_origin",
        subject="concept_x",
        paper_id="paper_a",
        year=2020,
    )
    e_2022 = _event(
        event_id="evo_2022",
        event_type="object_extension",
        subject="concept_x",
        paper_id="paper_b",
        year=2022,
    )

    result = projector.project_events(SNAPSHOT_ID, [e_2024, e_2020, e_2022])

    assert result.event_count == 3
    assert result.edge_count == 2

    edge_calls = [c for c in driver.tx.calls if "EVOLVES_TO" in c[0]]
    assert len(edge_calls) == 2

    # The chain must go 2020 -> 2022 -> 2024. Collect source/target ids
    # and confirm the order.
    src_targets: list[tuple[str, str]] = []
    for query, params in edge_calls:
        src_targets.append((params["source_id"], params["target_id"]))

    expected_chain = [
        (
            f"{NAMESPACE}|{SNAPSHOT_ID}|evo|evo_2020",
            f"{NAMESPACE}|{SNAPSHOT_ID}|evo|evo_2022",
        ),
        (
            f"{NAMESPACE}|{SNAPSHOT_ID}|evo|evo_2022",
            f"{NAMESPACE}|{SNAPSHOT_ID}|evo|evo_2024",
        ),
    ]
    assert src_targets == expected_chain

    # gap_years property on each edge matches the year deltas (2 and 2).
    gaps = [p.get("properties", {}).get("gap_years") for _, p in edge_calls]
    assert gaps == [2, 2]
    # subject echoed on the edge properties.
    subjects = [p.get("properties", {}).get("subject") for _, p in edge_calls]
    assert all(s == "concept_x" for s in subjects)


# ---------------------------------------------------------------------------
# 8. EvolutionEventNode label isolation from AcademicLandscapeNode
# ---------------------------------------------------------------------------


def test_projection_does_not_touch_academic_landscape_node_label() -> None:
    """GTI events must use their own label (``EvolutionEventNode``) so
    they coexist with the M4 ``AcademicLandscapeNode`` namespace without
    polluting it. Both layers share the same Neo4j database."""
    driver = FakeDriver()
    projector = _projector(driver)
    events = [
        _event(
            event_id="evo_a",
            event_type="explicit_proposal",
            subject="s",
            paper_id="paper_a",
            year=2020,
        ),
    ]

    projector.project_events(SNAPSHOT_ID, events)

    # Every node Cypher must reference EvolutionEventNode …
    node_queries = [
        q for q, _ in driver.tx.calls
        if "MERGE" in q and "node_id" in q
    ]
    assert node_queries
    assert all("EvolutionEventNode" in q for q in node_queries)
    # … and never AcademicLandscapeNode.
    assert not any("AcademicLandscapeNode" in q for q in node_queries)


# ---------------------------------------------------------------------------
# Bonus: count_events returns nodes + relationships counts from FakeDriver
# ---------------------------------------------------------------------------


def test_count_events_returns_nodes_and_relationships() -> None:
    rows = [{"nodes": 3, "relationships": 2}]
    driver = FakeDriver(read_results=[rows])
    projector = _projector(driver)

    counts = projector.count_events(SNAPSHOT_ID)

    assert counts == {"nodes": 3, "relationships": 2}