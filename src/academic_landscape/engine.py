"""Two-stage, fail-closed semantic analysis orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .models import CorpusSnapshot
from .paper_extraction import PaperExtractionService
from .prompts import build_corpus_prompt
from .semantic import validate_analysis_payload


class JsonModelClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


class AnalysisError(RuntimeError):
    """Raised when an AI response cannot be safely bound to the corpus."""


class AnalysisEngine:
    def __init__(self, client: JsonModelClient, cache_path: str | Path | None = None) -> None:
        self.client = client
        self.paper_extractor = PaperExtractionService(client, cache_path=cache_path)

    def _validate_paper(self, snapshot: CorpusSnapshot, paper: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": "0.1",
            "snapshot_id": snapshot.snapshot_id,
            "papers": [paper],
            "directions": [],
            "relations": [],
            "controversies": [],
            "frontier_candidates": [],
        }
        validation = validate_analysis_payload(snapshot, payload)
        if not validation.valid:
            codes = ", ".join(sorted({error["code"] for error in validation.errors}))
            paper_id = paper.get("paper_id", "unknown") if isinstance(paper, dict) else "unknown"
            raise AnalysisError(f"paper {paper_id} validation failed: {codes}")
        return validation.payload["papers"][0]

    def run(self, snapshot: CorpusSnapshot) -> dict[str, Any]:
        if not snapshot.included_documents:
            raise AnalysisError("no parsed text documents are eligible for semantic analysis")

        papers: list[dict[str, Any]] = []
        for document in snapshot.included_documents:
            try:
                paper = self.paper_extractor.extract(snapshot, document)
            except ValueError as error:
                raise AnalysisError(
                    f"paper {document.paper_id} extraction failed: {error}"
                ) from error
            papers.append(paper.model_dump(mode="json"))

        system_prompt, user_prompt = build_corpus_prompt(snapshot, papers)
        corpus_response = self.client.complete_json(system_prompt, user_prompt)
        if not isinstance(corpus_response, dict):
            raise AnalysisError("corpus response was not a JSON object")

        full_payload = {
            "schema_version": "0.1",
            "snapshot_id": snapshot.snapshot_id,
            "papers": papers,
            "directions": corpus_response.get("directions", []),
            "relations": corpus_response.get("relations", []),
            "controversies": corpus_response.get("controversies", []),
            "frontier_candidates": corpus_response.get("frontier_candidates", []),
        }
        validation = validate_analysis_payload(snapshot, full_payload)
        if not validation.valid:
            codes = ", ".join(sorted({error["code"] for error in validation.errors}))
            raise AnalysisError(f"corpus synthesis validation failed: {codes}")
        return validation.payload
