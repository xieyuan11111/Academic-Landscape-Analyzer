from pathlib import Path

import fitz

from academic_landscape.chunking import chunk_snapshot
from academic_landscape.embedding import EmbeddingVector, QdrantEmbeddingStore
from academic_landscape.ingest import build_snapshot

import pytest


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def _cleanup(store: QdrantEmbeddingStore, snapshot_id: str) -> None:
    """Best-effort teardown: never let cleanup failure mask the test result.

    The recurring orphan-collection problem came from cleanup running only on
    the happy path; a failure inside the test skipped the finally block in some
    cases and left directories behind.
    """
    try:
        store.delete_collection(snapshot_id)
    except Exception:  # noqa: BLE001 - teardown must not mask the test outcome
        pass
    finally:
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass


def test_qdrant_store_uses_extended_timeout_for_large_rebuilds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Long rebuild operations must not inherit the client's short HTTP default."""
    observed: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

    monkeypatch.setattr("academic_landscape.embedding.QdrantClient", _FakeClient)

    QdrantEmbeddingStore(
        url="http://127.0.0.1:6333",
        collection_prefix="unit",
        model_name="test-model",
        dimension=3,
    )

    assert observed == {"url": "http://127.0.0.1:6333", "timeout": 120.0}


def test_local_qdrant_store_commits_each_upsert_batch_once(
    tmp_path: Path,
) -> None:
    from academic_landscape.chunking import Chunk
    from academic_landscape.models import CorpusSnapshot

    local_path = tmp_path / "qdrant-local"
    store = QdrantEmbeddingStore(
        url="http://127.0.0.1:6333",
        collection_prefix="unit",
        model_name="test-model",
        dimension=2,
        local_path=str(local_path),
    )
    snapshot = CorpusSnapshot(
        snapshot_id="snapshot_batch",
        created_at="2026-09-26T00:00:00+00:00",
        input_root=str(tmp_path),
        documents=(),
    )
    chunks = [
        Chunk(
            chunk_id=f"chunk_{index}",
            snapshot_id=snapshot.snapshot_id,
            paper_id="paper_a",
            artifact_id="artifact_a",
            page_number=1,
            text=f"text {index}",
            text_hash=__import__("hashlib").sha256(f"text {index}".encode("utf-8")).hexdigest(),
            chunk_type="page",
            ordinal=index,
        )
        for index in range(3)
    ]
    vectors = [
        EmbeddingVector(chunk_id=chunk.chunk_id, values=(float(index), float(index + 1)))
        for index, chunk in enumerate(chunks)
    ]

    try:
        store.upsert(snapshot, chunks, vectors)
        assert store.count(snapshot.snapshot_id) == 3
        store.close()
        reopened = QdrantEmbeddingStore(
            url="http://127.0.0.1:6333",
            collection_prefix="unit",
            model_name="test-model",
            dimension=2,
            local_path=str(local_path),
        )
        try:
            assert reopened.count(snapshot.snapshot_id) == 3
            reopened.upsert(snapshot, chunks, vectors)
            assert reopened.count(snapshot.snapshot_id) == 3
        finally:
            reopened.close()
    finally:
        store.close()


@pytest.mark.live
def test_embedding_store_validates_dimension_and_upserts_idempotently(
    tmp_path: Path, qdrant_url: str, qdrant_collection_prefix: str
) -> None:
    make_pdf(tmp_path / "paper.pdf", "A source chunk for vector storage.")
    snapshot = build_snapshot(tmp_path)
    chunks = chunk_snapshot(snapshot)
    store = QdrantEmbeddingStore(
        url=qdrant_url,
        collection_prefix=qdrant_collection_prefix,
        model_name="test-model",
        dimension=3,
    )
    try:
        store.recreate_collection(snapshot.snapshot_id)
        vector = EmbeddingVector(
            chunk_id=chunks[0].chunk_id,
            values=(0.1, 0.2, 0.3),
        )
        store.upsert(snapshot, chunks, [vector])
        store.upsert(snapshot, chunks, [vector])

        hits = store.search(snapshot.snapshot_id, (0.1, 0.2, 0.3), limit=5)
        assert len(hits) == 1
        assert hits[0].chunk_id == chunks[0].chunk_id
        assert hits[0].payload["snapshot_id"] == snapshot.snapshot_id
        assert hits[0].payload["artifact_id"] == chunks[0].artifact_id
    finally:
        _cleanup(store, snapshot.snapshot_id)


@pytest.mark.live
def test_embedding_store_rejects_wrong_dimension(
    tmp_path: Path, qdrant_url: str, qdrant_collection_prefix: str
) -> None:
    make_pdf(tmp_path / "paper.pdf", "A source chunk.")
    snapshot = build_snapshot(tmp_path)
    chunks = chunk_snapshot(snapshot)
    store = QdrantEmbeddingStore(
        url=qdrant_url,
        collection_prefix=qdrant_collection_prefix,
        model_name="test-model",
        dimension=3,
    )
    try:
        store.recreate_collection(snapshot.snapshot_id)
        with pytest.raises(ValueError, match="dimension"):
            store.upsert(
                snapshot,
                chunks,
                [EmbeddingVector(chunk_id=chunks[0].chunk_id, values=(0.1, 0.2))],
            )
    finally:
        _cleanup(store, snapshot.snapshot_id)
