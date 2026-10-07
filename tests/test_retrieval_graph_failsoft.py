"""Tests that M5 lexical/vector retrieval survives a temporary graph outage."""

from __future__ import annotations

from types import SimpleNamespace

from academic_landscape.retrieval import HybridRetriever


class Lexical:
    def search_pages(self, snapshot_id, query, limit=10):
        return []


class Vector:
    def search(self, snapshot_id, query_vector, limit=10):
        return []


class DownGraph:
    def related_papers(self, snapshot_id, paper_ids, limit=10):
        raise RuntimeError("neo4j temporarily unavailable")


def test_graph_outage_returns_bundle_without_graph_relations() -> None:
    snapshot = SimpleNamespace(snapshot_id="snapshot_x")
    retriever = HybridRetriever(Lexical(), Vector(), DownGraph())
    bundle = retriever.search(
        snapshot, lexical_query="query", query_vector=[0.1, 0.2], limit=3
    )
    assert bundle.snapshot_id == "snapshot_x"
    assert bundle.graph_relations == ()
