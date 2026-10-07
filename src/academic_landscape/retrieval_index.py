"""M5 retrieval facade: populate lexical/vector/graph stores and run hybrid search.

Wraps the chunk builder, embedding backend, SQLite FTS store, Qdrant store and
Neo4j projector behind one object so the pipeline can populate a retrieval index
once and issue snapshot-scoped hybrid searches. Stores are projections; the
authority layer remains the source of truth.
"""

from __future__ import annotations

from typing import Any

from .chunking import chunk_snapshot
from .embedding import EmbeddingVector
from .models import CorpusSnapshot
from .retrieval import EvidenceBundle, HybridRetriever


class RetrievalIndex:
    def __init__(
        self,
        embedding_backend: Any,
        evidence_store: Any,
        vector_store: Any,
        graph_projector: Any,
    ) -> None:
        self.embedding_backend = embedding_backend
        self.evidence_store = evidence_store
        self.vector_store = vector_store
        self.graph_projector = graph_projector
        self._retriever = HybridRetriever(
            evidence_store=evidence_store,
            vector_store=vector_store,
            graph_projector=graph_projector,
        )

    def populate(
        self,
        snapshot: CorpusSnapshot,
        *,
        max_chars: int = 1800,
        overlap_chars: int = 180,
        vector_batch_size: int = 64,
    ) -> dict[str, int]:
        if vector_batch_size <= 0:
            raise ValueError("vector_batch_size must be positive")
        chunks = chunk_snapshot(
            snapshot, max_chars=max_chars, overlap_chars=overlap_chars
        )
        # lexical projection (pages/FTS)
        self.evidence_store.save_snapshot(snapshot)

        # vector projection
        if chunks:
            chunk_texts = [chunk.text for chunk in chunks]
            encoded = self.embedding_backend.encode(chunk_texts)
            vectors = [
                EmbeddingVector(
                    chunk_id=chunk.chunk_id,
                    values=tuple(float(value) for value in encoded[index]),
                )
                for index, chunk in enumerate(chunks)
            ]
            self.vector_store.recreate_collection(snapshot.snapshot_id)
            for start in range(0, len(chunks), vector_batch_size):
                end = start + vector_batch_size
                self.vector_store.upsert(
                    snapshot,
                    chunks[start:end],
                    vectors[start:end],
                )
        return {
            "snapshot_id": snapshot.snapshot_id,
            "chunks": len(chunks),
        }

    def search(
        self,
        snapshot: CorpusSnapshot,
        query: str,
        *,
        limit: int = 10,
    ) -> EvidenceBundle:
        encoded_query = self.embedding_backend.encode([query])[0]
        return self._retriever.search(
            snapshot,
            lexical_query=query,
            query_vector=[float(value) for value in encoded_query],
            limit=limit,
        )
