from pathlib import Path

import fitz

from academic_landscape.embedding import QdrantHit
from academic_landscape.ingest import build_snapshot
from academic_landscape.retrieval import HybridRetriever
from academic_landscape.sqlite_store import SQLiteEvidenceStore


def make_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


class FakeVectorStore:
    def __init__(self, hit):
        self.hit = hit
        self.calls = []

    def search(self, snapshot_id, query_vector, *, limit):
        self.calls.append((snapshot_id, tuple(query_vector), limit))
        return [self.hit]


class FakeGraphProjector:
    def __init__(self, relation):
        self.relation = relation
        self.calls = []

    def related_papers(self, snapshot_id, paper_ids, *, limit):
        self.calls.append((snapshot_id, tuple(paper_ids), limit))
        return [self.relation]


def test_hybrid_retriever_joins_lexical_vector_and_graph_results_with_source_ids(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "Method alpha is evaluated on a real dataset.")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]
    store = SQLiteEvidenceStore(tmp_path / "evidence.sqlite3")
    store.save_snapshot(snapshot)
    vector = FakeVectorStore(
        QdrantHit(
            chunk_id="chunk_1",
            score=0.9,
            payload={
                "snapshot_id": snapshot.snapshot_id,
                "paper_id": document.paper_id,
                "artifact_id": document.artifact_id,
                "page_number": 1,
                "chunk_type": "page",
                "ordinal": 0,
                "text_hash": "hash",
                "text": "Method alpha is evaluated on a real dataset.",
            },
        )
    )
    graph = FakeGraphProjector(
        {
            "relation_id": "R1",
            "relation_type": "challenges",
            "source_paper_id": document.paper_id,
            "target_paper_id": document.paper_id,
            "evidence_ids": ["E1"],
        }
    )

    bundle = HybridRetriever(store, vector, graph).search(
        snapshot,
        lexical_query="alpha",
        query_vector=(1.0, 0.0),
        limit=5,
    )

    assert bundle.snapshot_id == snapshot.snapshot_id
    assert len(bundle.lexical_hits) == 1
    assert len(bundle.vector_hits) == 1
    assert len(bundle.graph_relations) == 1
    assert bundle.source_ids == {document.paper_id, document.artifact_id}
    assert vector.calls == [(snapshot.snapshot_id, (1.0, 0.0), 5)]
    assert graph.calls == [(snapshot.snapshot_id, (document.paper_id,), 5)]
    store.close()
