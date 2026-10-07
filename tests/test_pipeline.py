"""End-to-end pipeline integration test with fakes (no network, no real model)."""

from __future__ import annotations

from typing import Any

import numpy as np

from academic_landscape.models import CorpusSnapshot, DocumentRecord, PageText
from academic_landscape.pipeline import LandscapePipeline
from academic_landscape.landscape_report import render_landscape_report


class FakeEmbeddingBackend:
    """Everything sits near one direction [1,0] so one cluster is viable."""

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for index, _ in enumerate(texts):
            angle = 0.01 * (index % 3)
            vectors.append(
                np.asarray(
                    [np.cos(angle), np.sin(angle), 0.0], dtype=float
                )
            )
        return np.asarray(vectors)


class ScriptedLLM:
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if "name a research direction" in system_prompt:
            return {
                "label": "Direction A",
                "description": "a coherent direction",
                "status": "emerging",
            }
        if "compare papers" in system_prompt:
            return {
                "commonalities": [
                    {"statement": "shared mechanism",
                     "evidence": ["paper_a::E1", "paper_b::E1"]}
                ],
                "differences": [
                    {"dimension": "method", "statement": "method gap",
                     "evidence": ["paper_a::E2"]}
                ],
                "controversies": [
                    {"statement": "a real split",
                     "positions": [
                         {"paper_id": "paper_a", "stance": "view one",
                          "evidence": ["paper_a::E3"]},
                         {"paper_id": "paper_b", "stance": "view two",
                          "evidence": ["paper_b::E3"]},
                     ]}
                ],
            }
        if "reconstruct the developmental stages" in system_prompt:
            return {
                "material_assessment": "sparse",
                "stages": [
                    {"label": "Early", "year_start": 2020, "year_end": 2020,
                     "change_dimension": "method", "summary": "early period",
                     "key_paper_ids": ["paper_a"],
                     "evidence": ["paper_a::E1"], "confidence": "medium"},
                    {"label": "Later", "year_start": 2024, "year_end": 2024,
                     "change_dimension": "object", "summary": "later period",
                     "key_paper_ids": ["paper_b"],
                     "evidence": ["paper_b::E1"], "confidence": "low"},
                ],
                "turning_points": [
                    {"year": 2024, "paper_id": "paper_b",
                     "statement": "object shift", "evidence": ["paper_b::E2"]}
                ],
                "notes": "",
            }
        raise AssertionError("unexpected LLM call")


def _document(paper_id: str, filename: str, author: str) -> DocumentRecord:
    return DocumentRecord(
        artifact_id=f"artifact_{paper_id}",
        paper_id=paper_id,
        relative_path=filename,
        filename=filename,
        sha256=f"sha_{paper_id}",
        size_bytes=10,
        status="parsed",
        metadata={"author": author},
        pages=(PageText(1, f"body text from {paper_id} with terms"),),
    )


def _extraction(paper_id: str) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "research_question": {"text": f"RQ {paper_id}", "evidence_ids": ["E1"]},
        "concepts": [
            {"name": f"{paper_id} concept one", "definition": "def one",
             "evidence_ids": ["E1"]},
            {"name": f"{paper_id} concept two", "definition": "def two",
             "evidence_ids": ["E2"]},
        ],
        "theoretical_frameworks": [],
        "methods": [],
        "data_samples": [
            {"description": f"field samples from {paper_id}",
             "evidence_ids": ["E1"]}
        ],
        "contexts": [],
        "results": [
            {"text": f"finding from {paper_id}", "evidence_ids": ["E1"]}
        ],
        "limitations": [
            {"text": f"future work note from {paper_id}", "evidence_ids": []}
        ],
        "claims": [
            {"claim_id": "C1", "text": f"author claim {paper_id}",
             "evidence_type": "A", "evidence_ids": ["E1"]}
        ],
        "evidence": [
            {"evidence_id": "E1", "artifact_id": f"artifact_{paper_id}",
             "page_number": 1, "quote": f"quote one {paper_id}",
             "locator": "page 1"},
            {"evidence_id": "E2", "artifact_id": f"artifact_{paper_id}",
             "page_number": 1, "quote": f"quote two {paper_id}",
             "locator": "page 1"},
            {"evidence_id": "E3", "artifact_id": f"artifact_{paper_id}",
             "page_number": 2, "quote": f"quote three {paper_id}",
             "locator": "page 2"},
        ],
    }


def _snapshot() -> CorpusSnapshot:
    return CorpusSnapshot(
        snapshot_id="snapshot_int",
        created_at="2026-09-22T00:00:00+00:00",
        input_root="E:/input",
        documents=(
            _document("paper_a", "paper_a_study_2020.pdf", "Wei Zhang"),
            _document("paper_b", "paper_b_study_2024.pdf", "Jun Zuo"),
        ),
    )


def test_pipeline_runs_full_chain() -> None:
    snapshot = _snapshot()
    extractions = {
        "paper_a": _extraction("paper_a"),
        "paper_b": _extraction("paper_b"),
    }
    pipeline = LandscapePipeline(
        client=ScriptedLLM(),
        embedding_backend=FakeEmbeddingBackend(),
        current_year=2026,
        n_clusters=1,
    )
    result = pipeline.run(snapshot, extractions)

    assert result.snapshot_id == "snapshot_int"
    assert len(result.directions) == 1
    assert result.directions[0].label == "Direction A"
    assert len(result.comparisons) == 1
    assert len(result.reconstructions) == 1
    assert result.family_report is not None
    assert result.frontier_report is not None

    # name vectors are passed to the merge: fake embeddings are mutually close,
    # so every non-established thread in the single direction merges into one
    assert len(result.merged_frontiers) == 1

    rendered = render_landscape_report(result)
    for section in ("9.1", "9.7", "9.11"):
        assert section in rendered


def test_pipeline_infers_years() -> None:
    snapshot = _snapshot()
    extractions = {
        "paper_a": _extraction("paper_a"),
        "paper_b": _extraction("paper_b"),
    }
    pipeline = LandscapePipeline(
        client=ScriptedLLM(),
        embedding_backend=FakeEmbeddingBackend(),
        current_year=2026,
        n_clusters=1,
    )
    result = pipeline.run(snapshot, extractions)
    assert result.corpus.year_min == 2020
    assert result.corpus.year_max == 2024
    assert result.corpus.paper_count == 2
