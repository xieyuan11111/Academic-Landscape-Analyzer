"""Snapshot-scoped hybrid retrieval with source provenance preserved."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .embedding import QdrantEmbeddingStore, QdrantHit
from .graph_projector import Neo4jProjector
from .models import CorpusSnapshot
from .sqlite_store import PageSearchHit, SQLiteEvidenceStore


@dataclass(frozen=True)
class EvidenceBundle:
    snapshot_id: str
    lexical_hits: tuple[PageSearchHit, ...]
    vector_hits: tuple[QdrantHit, ...]
    graph_relations: tuple[dict[str, Any], ...]
    source_ids: set[str]


class HybridRetriever:
    def __init__(
        self,
        evidence_store: SQLiteEvidenceStore,
        vector_store: QdrantEmbeddingStore,
        graph_projector: Neo4jProjector,
    ) -> None:
        self.evidence_store = evidence_store
        self.vector_store = vector_store
        self.graph_projector = graph_projector

    def search(
        self,
        snapshot: CorpusSnapshot,
        *,
        lexical_query: str,
        query_vector: Sequence[float],
        limit: int = 10,
    ) -> EvidenceBundle:
        if limit <= 0:
            raise ValueError("limit must be positive")
        lexical_hits = self.evidence_store.search_pages(
            snapshot.snapshot_id,
            lexical_query,
            limit=limit,
        )
        vector_hits = self.vector_store.search(
            snapshot.snapshot_id,
            query_vector,
            limit=limit,
        )
        paper_ids = {
            hit.paper_id for hit in lexical_hits
        }
        paper_ids.update(
            str(hit.payload["paper_id"])
            for hit in vector_hits
            if hit.payload.get("snapshot_id") == snapshot.snapshot_id
            and hit.payload.get("paper_id")
        )
        try:
            graph_relations = self.graph_projector.related_papers(
                snapshot.snapshot_id,
                sorted(paper_ids),
                limit=limit,
            )
        except Exception:  # graph is an optional navigation projection
            graph_relations = []
        source_ids: set[str] = set(paper_ids)
        source_ids.update(hit.artifact_id for hit in lexical_hits)
        source_ids.update(
            str(hit.payload["artifact_id"])
            for hit in vector_hits
            if hit.payload.get("snapshot_id") == snapshot.snapshot_id
            and hit.payload.get("artifact_id")
        )
        return EvidenceBundle(
            snapshot_id=snapshot.snapshot_id,
            lexical_hits=tuple(lexical_hits),
            vector_hits=tuple(vector_hits),
            graph_relations=tuple(graph_relations),
            source_ids=source_ids,
        )
