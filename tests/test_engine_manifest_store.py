"""Tests for :mod:`academic_landscape.engine_manifest_store`.

Authority invariant: the store is a rebuildable projection ledger, not the
authority layer.  These tests verify:

- durable roundtrip across DB close/reopen
- APPLIED idempotency (same projection_id + same input_hash)
- FAILED retry record visibility
- snapshot isolation (fail-closed retrieval)
- malicious / unscoped evidence rejected at the API boundary
- the Graphiti ``ProvenanceManifest`` protocol

No live services needed — stdlib sqlite only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from academic_landscape.engine_manifest_store import (
    EngineManifestStore,
    InputHashConflictError,
    ManifestStoreError,
    SnapshotMismatchError,
    SQLiteProvenanceManifest,
)
from academic_landscape.graphiti_projection import (
    ManifestEntry,
    ProvenanceManifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "manifest.db"


@pytest.fixture
def store(store_path: Path) -> EngineManifestStore:
    s = EngineManifestStore(store_path)
    yield s
    s.close()


def _make_graphiti_entry(
    *,
    snapshot_id: str = "snap-2026-09-25-A",
    projection_id: str = "proj_g_abc123def456",
    input_hash: str = "hash-xyz",
    relation_id: str = "rel-007",
    authority_reinterpretation_risk: bool = True,
    projected_at: str = "2026-09-25T12:00:00+00:00",
) -> ManifestEntry:
    return ManifestEntry(
        projection_id=projection_id,
        input_hash=input_hash,
        relation_id=relation_id,
        snapshot_id=snapshot_id,
        projected_at=projected_at,
        authority_reinterpretation_risk=authority_reinterpretation_risk,
    )


# ---------------------------------------------------------------------------
# Schema / migration
# ---------------------------------------------------------------------------

class TestSchemaAndMigration:
    def test_creates_db_file(self, store_path: Path) -> None:
        assert not store_path.exists()
        s = EngineManifestStore(store_path)
        s.close()
        assert store_path.exists()

    def test_migration_is_idempotent(self, store_path: Path) -> None:
        # Open, close, reopen — schema must remain identical.
        s1 = EngineManifestStore(store_path)
        s1.close()
        s2 = EngineManifestStore(store_path)
        try:
            cur = s2._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='engine_manifest'"
            )
            assert cur.fetchone() is not None
        finally:
            s2.close()

    def test_wal_mode_enabled(self, store: EngineManifestStore) -> None:
        cur = store._conn.execute("PRAGMA journal_mode")
        assert cur.fetchone()[0].lower() == "wal"

    def test_busy_timeout_set(self, store: EngineManifestStore) -> None:
        cur = store._conn.execute("PRAGMA busy_timeout")
        assert cur.fetchone()[0] == 5000

    def test_parent_directory_created(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "manifest.db"
        s = EngineManifestStore(nested)
        s.close()
        assert nested.exists()


# ---------------------------------------------------------------------------
# Durable roundtrip — close DB, reopen, expect exact equality
# ---------------------------------------------------------------------------

class TestDurableRoundtrip:
    def test_graphiti_roundtrip_survives_reopen(
        self, store_path: Path, store: EngineManifestStore
    ) -> None:
        entry = _make_graphiti_entry()
        assert store.write_graphiti(entry) is True
        store.close()

        reopened = EngineManifestStore(store_path)
        try:
            got = reopened.get_graphiti_entry(entry.snapshot_id, entry.projection_id)
            assert got is not None
            assert got.projection_id == entry.projection_id
            assert got.input_hash == entry.input_hash
            assert got.relation_id == entry.relation_id
            assert got.snapshot_id == entry.snapshot_id
            assert got.authority_reinterpretation_risk is True
        finally:
            reopened.close()

class TestAppliedIdempotency:
    def test_graphiti_re_record_returns_false(
        self, store: EngineManifestStore
    ) -> None:
        entry = _make_graphiti_entry()
        assert store.write_graphiti(entry) is True
        # Same id+hash → idempotent no-op (matches in-memory ProvenanceManifest).
        assert store.write_graphiti(entry) is False

class TestFailedRetryVisibility:
    def test_graphiti_conflict_on_different_input_hash(
        self, store: EngineManifestStore
    ) -> None:
        ok = _make_graphiti_entry(input_hash="hash-A")
        store.write_graphiti(ok)
        bad = _make_graphiti_entry(
            projection_id=ok.projection_id,
            input_hash="hash-B",
        )
        with pytest.raises(InputHashConflictError):
            store.write_graphiti(bad)


# ---------------------------------------------------------------------------
# Snapshot isolation — fail-closed retrieval by snapshot
# ---------------------------------------------------------------------------

class TestSnapshotIsolation:
    def test_get_across_snapshot_raises(self, store: EngineManifestStore) -> None:
        snap_a = _make_graphiti_entry(snapshot_id="snap-A", projection_id="proj_isolation_a")
        store.write_graphiti(snap_a)
        with pytest.raises(SnapshotMismatchError):
            store.get_graphiti_entry("snap-B", snap_a.projection_id)

    def test_get_missing_under_correct_snapshot_returns_none(
        self, store: EngineManifestStore
    ) -> None:
        got = store.get_graphiti_entry("snap-A", "proj_does_not_exist")
        assert got is None

    def test_list_for_snapshot_is_scoped(
        self, store: EngineManifestStore
    ) -> None:
        snap_a1 = _make_graphiti_entry(
            snapshot_id="snap-A", projection_id="proj_scope_a1", input_hash="h-a1"
        )
        snap_a2 = _make_graphiti_entry(
            snapshot_id="snap-A", projection_id="proj_scope_a2", input_hash="h-a2"
        )
        snap_b = _make_graphiti_entry(
            snapshot_id="snap-B", projection_id="proj_scope_b", input_hash="h-b"
        )
        store.write_graphiti(snap_a1)
        store.write_graphiti(snap_a2)
        store.write_graphiti(snap_b)

        rows_a = store.list_for_snapshot("snap-A")
        rows_b = store.list_for_snapshot("snap-B")
        assert len(rows_a) == 2
        assert len(rows_b) == 1
        assert {r["projection_id"] for r in rows_a} == {
            "proj_scope_a1", "proj_scope_a2"
        }
        assert {r["projection_id"] for r in rows_b} == {"proj_scope_b"}

    def test_graphiti_manifest_rejects_foreign_snapshot(
        self, store: EngineManifestStore
    ) -> None:
        prov = SQLiteProvenanceManifest(store, snapshot_id="snap-A")
        foreign = _make_graphiti_entry(snapshot_id="snap-B")
        with pytest.raises(SnapshotMismatchError):
            prov.record(foreign)


# ---------------------------------------------------------------------------
# Protocol compliance — both adapters match the upstream protocol surface
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_sqlite_manifest_satisfies_provenance_manifest_protocol(self) -> None:
        store = EngineManifestStore(Path("protocol_test2.db"))
        try:
            prov = SQLiteProvenanceManifest(store, snapshot_id="snap-1")
            for method in ("get", "record", "has", "entries"):
                assert hasattr(prov, method), f"missing method {method}"
                assert callable(getattr(prov, method))
        finally:
            store.close()
            Path("protocol_test2.db").unlink(missing_ok=True)

    def test_sqlite_manifest_record_returns_bool(
        self, store: EngineManifestStore
    ) -> None:
        prov = SQLiteProvenanceManifest(store, snapshot_id="snap-1")
        entry = _make_graphiti_entry(snapshot_id="snap-1")
        result_first = prov.record(entry)
        result_second = prov.record(entry)
        assert result_first is True
        assert result_second is False

    def test_sqlite_manifest_matches_in_memory_semantics(
        self, store: EngineManifestStore
    ) -> None:
        # SQLite-backed adapter must produce identical record()/get() results
        # as the in-memory ProvenanceManifest for the same sequence of ops.
        inmem = ProvenanceManifest()
        durable = SQLiteProvenanceManifest(store, snapshot_id="snap-X")

        e1 = _make_graphiti_entry(
            projection_id="proj_x_1111111111",
            input_hash="hash-1",
            relation_id="rel-1",
            snapshot_id="snap-X",
        )
        e2 = _make_graphiti_entry(
            projection_id="proj_x_2222222222",
            input_hash="hash-2",
            relation_id="rel-2",
            snapshot_id="snap-X",
        )
        # Re-record e1 idempotently
        assert durable.record(e1) == inmem.record(e1) is True
        assert durable.record(e2) == inmem.record(e2) is True
        assert durable.record(e1) == inmem.record(e1) is False

        for pid in ("proj_x_1111111111", "proj_x_2222222222"):
            d_entry = durable.get(pid)
            m_entry = inmem.get(pid)
            assert d_entry is not None
            assert m_entry is not None
            assert d_entry.projection_id == m_entry.projection_id
            assert d_entry.input_hash == m_entry.input_hash
            assert d_entry.relation_id == m_entry.relation_id
            assert d_entry.snapshot_id == m_entry.snapshot_id


# ---------------------------------------------------------------------------
# Malicious / unscoped evidence rejected
# ---------------------------------------------------------------------------

class TestMaliciousInputRejected:
    def test_non_entry_payload_rejected(
        self, store: EngineManifestStore
    ) -> None:
        with pytest.raises(ManifestStoreError):
            store.write_graphiti("not a ManifestEntry")


# ---------------------------------------------------------------------------
# Sink/Manifest full lifecycle
# ---------------------------------------------------------------------------

class TestSinkLifecycle:
    def test_graphiti_record_duplicate(self, store: EngineManifestStore) -> None:
        prov = SQLiteProvenanceManifest(store, snapshot_id="snap-G")
        e = _make_graphiti_entry(snapshot_id="snap-G")
        assert prov.record(e) is True
        # Same id+hash → False (matches ProvenanceManifest.record).
        assert prov.record(e) is False
        assert prov.has(e.projection_id) is True


# ---------------------------------------------------------------------------
# Circular-import safety — module imports cleanly even if its adapters are
# imported by the projection modules later.
# ---------------------------------------------------------------------------

class TestCircularImportSafety:
    def test_module_imports_without_circular_error(self) -> None:
        """Import order must not produce a partially-initialised module.

        The bare imports above used to be the whole test, so it passed
        unconditionally.  A circular import surfaces as ``ImportError`` or as
        a module whose attributes are missing at use time, so both are
        asserted: the modules import, and every attribute this store resolves
        lazily is actually bound.
        """
        import importlib

        import academic_landscape.engine_manifest_store as mod

        # Importing the projection module that this store references lazily
        # must not re-trigger the store's import.
        importlib.import_module("academic_landscape.graphiti_projection")
        reimported = importlib.import_module("academic_landscape.engine_manifest_store")

        # Re-import must yield the SAME module object (a reload cycle would
        # hand back a second instance with its own class identities).
        assert reimported is mod

        # The class the tests actually use must be the one on the module.
        assert isinstance(mod.EngineManifestStore, type)
        assert mod.EngineManifestStore.__name__ == "EngineManifestStore"

        # Symbols the projection adapters reach for at call time.
        for attr in ("EngineManifestStore", "SQLiteProvenanceManifest",
                     "ManifestStoreError", "SnapshotMismatchError",
                     "InputHashConflictError", "InvalidEvidenceScopeError"):
            assert hasattr(mod, attr), f"engine_manifest_store lost {attr}"
            assert getattr(mod, attr) is not None

    def test_projection_modules_import_manifest_store_lazily(self) -> None:
        """The reverse edge must stay lazy: importing a projection adapter
        may not drag the store in before it is needed.

        This is the half of the cycle that a bare import test cannot see —
        ``sys.modules`` is the only place the eager/lazy distinction shows.
        """
        import json
        import subprocess
        import sys

        code = (
            "import json, sys;"
            "import academic_landscape.graphiti_projection;"
            "b = 'academic_landscape.engine_manifest_store' in sys.modules;"
            "print(json.dumps({'graphiti': b}))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        flags = json.loads(proc.stdout.strip())
        # The adapter must leave the store unimported until a call needs it.
        assert flags["graphiti"] is False, (
            "graphiti_projection eagerly imported engine_manifest_store"
        )
