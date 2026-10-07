"""Tests for the M5 RetrievalIndex facade (population + retrieval)."""

from __future__ import annotations

from typing import Any

import numpy as np

from academic_landscape.models import CorpusSnapshot, DocumentRecord, PageText
from academic_landscape.retrieval_index import RetrievalIndex


class FakeEmbedding:
    dimension = 3

    def encode(self, texts: list[str]) -> np.ndarray:
        out = []
        for i, _ in enumerate(texts):
            vec = np.asarray([1.0, 0.01 * i, 0.0], dtype=float)
            out.append(vec / np.linalg.norm(vec))
        return np.asarray(out) if out else np.zeros((0, 3))


class FakeEvidenceStore:
    def __init__(self) -> None:
        self.saved: list[str] = []

    def save_snapshot(self, snapshot: Any) -> None:
        self.saved.append(snapshot.snapshot_id)

    def search_pages(self, snapshot_id: str, query: str, limit: int = 10) -> list[Any]:
        from academic_landscape.sqlite_store import PageSearchHit

        return [
            PageSearchHit(
                snapshot_id=snapshot_id,
                paper_id="paper_a",
                artifact_id="artifact_a",
                page_number=1,
                text=query,
                rank=1.0,
            )
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserted: list[int] = []
        self.searches: list[int] = []

    def recreate_collection(self, snapshot_id: str) -> None:
        self.upserted = []

    def upsert(self, snapshot: Any, chunks: Any, vectors: Any) -> None:
        self.upserted.append(len(chunks))

    def search(self, snapshot_id: str, query_vector: Any, limit: int = 10) -> list[Any]:
        self.searches.append(len(query_vector))
        return []


class FakeProjector:
    def related_papers(self, snapshot_id: str, paper_ids: Any, limit: int = 10) -> list[Any]:
        return []


def _snapshot() -> CorpusSnapshot:
    doc = DocumentRecord(
        artifact_id="artifact_a",
        paper_id="paper_a",
        relative_path="a.pdf",
        filename="a.pdf",
        sha256="sha",
        size_bytes=10,
        status="parsed",
        pages=(PageText(1, "alpha beta gamma delta epsilon zeta eta theta"),),
    )
    return CorpusSnapshot(
        snapshot_id="snapshot_x",
        created_at="2026-09-23T00:00:00+00:00",
        input_root="E:/input",
        documents=(doc,),
    )


def _index() -> tuple[RetrievalIndex, FakeEvidenceStore, FakeVectorStore]:
    evidence_store = FakeEvidenceStore()
    vector_store = FakeVectorStore()
    index = RetrievalIndex(
        embedding_backend=FakeEmbedding(),
        evidence_store=evidence_store,
        vector_store=vector_store,
        graph_projector=FakeProjector(),
    )
    return index, evidence_store, vector_store


def test_populate_writes_lexical_and_vector_stores() -> None:
    index, evidence_store, vector_store = _index()
    index.populate(_snapshot())
    assert evidence_store.saved == ["snapshot_x"]
    assert vector_store.upserted  # at least one batch upsert


def test_populate_returns_chunk_and_vector_counts() -> None:
    index, _, _ = _index()
    summary = index.populate(_snapshot())
    assert summary["chunks"] >= 1
    assert summary["snapshot_id"] == "snapshot_x"


def test_populate_splits_large_vector_batches() -> None:
    index, _, vector_store = _index()
    summary = index.populate(
        _snapshot(), max_chars=10, overlap_chars=2, vector_batch_size=2
    )
    assert summary["chunks"] >= 1
    assert len(vector_store.upserted) >= 2


def test_search_returns_evidence_bundle() -> None:
    index, _, _ = _index()
    snapshot = _snapshot()
    index.populate(snapshot)
    bundle = index.search(snapshot, query="alpha beta", limit=5)
    assert bundle.snapshot_id == "snapshot_x"
    assert bundle.lexical_hits is not None
    assert bundle.vector_hits is not None


def test_search_encodes_query_to_backend_dimension() -> None:
    index, _, vector_store = _index()
    snapshot = _snapshot()
    index.populate(snapshot)
    index.search(snapshot, query="alpha", limit=3)
    assert vector_store.searches  # search reached the vector store
    assert vector_store.searches[0] == 3
