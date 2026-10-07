from pathlib import Path

import fitz
import pytest

from academic_landscape.chunking import chunk_snapshot
from academic_landscape.embedding import (
    EmbeddingVector,
    OpenAIEmbeddingProvider,
    QdrantEmbeddingStore,
    embed_chunks,
)
from academic_landscape.ingest import build_snapshot


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class FakeEmbeddings:
    def create(self, *, model, input, encoding_format):
        return type(
            "Response",
            (),
            {
                "data": [
                    type("Item", (), {"index": index, "embedding": [float(index + 1), 0.0]})
                    for index, _ in enumerate(input)
                ]
            },
        )()


def test_openai_embedding_provider_uses_sdk_shape_and_preserves_order() -> None:
    provider = OpenAIEmbeddingProvider(
        model_name="embedding-test",
        client=type("Client", (), {"embeddings": FakeEmbeddings()})(),
    )

    vectors = provider.embed(["first", "second"])

    assert vectors == [(1.0, 0.0), (2.0, 0.0)]
    assert provider.dimension == 2


def test_embed_chunks_keeps_chunk_ids_aligned(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "A source chunk.")
    snapshot = build_snapshot(tmp_path)
    chunks = chunk_snapshot(snapshot)
    provider = OpenAIEmbeddingProvider(
        model_name="embedding-test",
        client=type("Client", (), {"embeddings": FakeEmbeddings()})(),
    )

    vectors = embed_chunks(provider, chunks)

    assert [vector.chunk_id for vector in vectors] == [chunk.chunk_id for chunk in chunks]


@pytest.mark.live
def test_qdrant_search_is_scoped_to_snapshot(
    tmp_path: Path, qdrant_url: str, qdrant_collection_prefix: str
) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    make_pdf(first_dir / "paper.pdf", "First snapshot source.")
    make_pdf(second_dir / "paper.pdf", "Second snapshot source.")
    first = build_snapshot(first_dir)
    second = build_snapshot(second_dir)
    first_chunk = chunk_snapshot(first)[0]
    second_chunk = chunk_snapshot(second)[0]
    store = QdrantEmbeddingStore(
        url=qdrant_url,
        collection_prefix=qdrant_collection_prefix,
        model_name="scope-test",
        dimension=2,
    )
    try:
        store.recreate_collection(first.snapshot_id)
        store.recreate_collection(second.snapshot_id)
        store.upsert(
            first,
            [first_chunk],
            [EmbeddingVector(chunk_id=first_chunk.chunk_id, values=(1.0, 0.0))],
        )
        store.upsert(
            second,
            [second_chunk],
            [EmbeddingVector(chunk_id=second_chunk.chunk_id, values=(1.0, 0.0))],
        )

        first_hits = store.search(first.snapshot_id, (1.0, 0.0), limit=10)
        second_hits = store.search(second.snapshot_id, (1.0, 0.0), limit=10)

        assert [hit.chunk_id for hit in first_hits] == [first_chunk.chunk_id]
        assert [hit.chunk_id for hit in second_hits] == [second_chunk.chunk_id]
    finally:
        # Best-effort teardown: cleanup failure must not mask the assertion.
        for snapshot_id in (first.snapshot_id, second.snapshot_id):
            try:
                store.delete_collection(snapshot_id)
            except Exception:  # noqa: BLE001
                pass
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass
