"""End-to-end knowledge-base-driven landscape pipeline.

Chains every validated stage without a long-context call:
semantic items -> local vectors -> candidate clusters (cleaning gate)
-> bounded direction naming -> bounded comparison -> concept-thread stage
reconstruction -> evidence families -> popularity/innovation/frontier scoring
-> cross-direction merge -> assembled LandscapeResult.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from .citations import author_year_map
from .clustering import cluster_items
from .cross_paper import compare_directions
from .debate_cards import DebatePosition, build_debate_card
from .direction_naming import name_directions
from .evidence_family import analyze_evidence_families, split_author_list
from .evidence_tier import classify_evidence_tier
from .frontier_analysis import build_frontier_report
from .landscape_merge import merge_frontier_signals
from .landscape_report import CorpusFacts, LandscapeResult
from .models import CorpusSnapshot
from .projection_adapter import build_projection_payload
from .publication_years import infer_publication_year
from .relation_extraction import (
    RELATIONS_REQUIRING_TARGET_EVIDENCE,
    extract_paper_relations,
)
from .semantic_items import build_semantic_items
from .source_cards import SourceRef, build_source_card, build_source_refs
from .stage_reconstruction import reconstruct_stages
from .timeline_events import normalize_events
from .versioning import new_version


def _retrieval_context(index: Any, snapshot: CorpusSnapshot, query: str, limit: int = 5) -> str:
    """Render bounded discovery-only context; never promotes it to evidence."""
    if index is None:
        return ""
    bundle = index.search(snapshot, query, limit=limit)
    if bundle is None:
        return ""
    lines: list[str] = []
    seen: set[str] = set()
    for hit in list(bundle.lexical_hits) + list(bundle.vector_hits):
        if hasattr(hit, "text"):
            text = hit.text
            key = f"lex:{hit.artifact_id}:{hit.page_number}:{text[:80]}"
            source = f"{hit.paper_id} p{hit.page_number}"
        else:
            payload = getattr(hit, "payload", {})
            text = str(payload.get("text", ""))
            key = f"vec:{getattr(hit, 'chunk_id', '')}"
            source = f"{payload.get('paper_id', '?')} p{payload.get('page_number', '?')}"
        if text and key not in seen:
            seen.add(key)
            lines.append(f"[{source}] {text[:800]}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def default_cluster_count(item_count: int) -> int:
    return max(2, min(12, round(math.sqrt(item_count / 2))))

def _as_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


class LandscapePipeline:
    def __init__(
        self,
        client: Any,
        embedding_backend: Any,
        current_year: int = 2026,
        n_clusters: int | None = None,
        graph_projector: Any = None,
        retrieval_index: Any = None,
        relation_cache_path: Any = None,
    ) -> None:
        self.client = client
        self.embedding_backend = embedding_backend
        self.current_year = current_year
        self.n_clusters = n_clusters
        self.graph_projector = graph_projector
        self.retrieval_index = retrieval_index
        self.relation_cache_path = relation_cache_path

    def run(
        self,
        snapshot: CorpusSnapshot,
        extractions: dict[str, Any],
    ) -> LandscapeResult:
        snapshot_id = snapshot.snapshot_id

        if self.retrieval_index is not None:
            self.retrieval_index.populate(snapshot)

        extraction_by_id = {
            paper_id: _as_dict(extraction)
            for paper_id, extraction in extractions.items()
        }

        years: dict[str, int] = {}
        for document in snapshot.documents:
            guess = infer_publication_year(document, self.current_year)
            years[document.paper_id] = guess.year

        items = build_semantic_items(snapshot_id, list(extraction_by_id.values()))

        item_texts = [
            f"{item.name}: {item.description}"
            if item.description else item.name
            for item in items
        ]
        item_vectors = (
            self.embedding_backend.encode(item_texts) if items else np.zeros((0, 0))
        )
        name_vectors = (
            self.embedding_backend.encode([item.name for item in items])
            if items else np.zeros((0, 0))
        )

        cluster_count = self.n_clusters or default_cluster_count(len(items))
        clusters = cluster_items(
            items, item_vectors, n_clusters=cluster_count
        )
        directions = name_directions(
            snapshot_id, clusters, items, self.client
        )
        retrieval_context_by_direction = {
            direction.direction_id: _retrieval_context(
                self.retrieval_index,
                snapshot,
                f"{direction.label}. {direction.description}",
            )
            for direction in directions
        }
        comparisons = compare_directions(
            directions, extraction_by_id, years, self.client,
            retrieval_context_by_direction=retrieval_context_by_direction,
        )
        reconstructions = [
            reconstruct_stages(
                direction, items, name_vectors, years, self.client,
                retrieval_context=retrieval_context_by_direction.get(
                    direction.direction_id, ""
                ),
            )
            for direction in directions
        ]
        relation_results = extract_paper_relations(
            self.client,
            directions,
            extraction_by_id,
            years,
            retrieval_context_by_direction=retrieval_context_by_direction,
            snapshot_id=snapshot_id,
            cache_path=self.relation_cache_path,
        )

        def _global_evidence_id(pid: str, eid: str) -> str:
            return eid if "::" in eid else f"{pid}::{eid}"

        generated_at = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        doc_by_pid = {document.paper_id: document for document in snapshot.documents}

        def _ref(paper_id: str, evidence_ids: Sequence[str]) -> list[SourceRef]:
            extraction = extraction_by_id.get(paper_id, {})
            return build_source_refs(
                paper_id=paper_id,
                evidence_ids=[
                    _global_evidence_id(paper_id, evidence_id)
                    for evidence_id in evidence_ids
                ],
                extraction=extraction,
                document=doc_by_pid.get(paper_id),
                year=years.get(paper_id),
                verified_at=generated_at,
            )

        # Evidence families are computed FIRST: the tier classification needs
        # the independence signal (shared dataset / shared team) before the
        # source cards are built.
        authors_by_paper: dict[str, list[str]] = {}
        data_samples_by_paper: dict[str, list[str]] = {}
        results_by_paper: dict[str, list[str]] = {}
        for paper_id, extraction in extraction_by_id.items():
            document = doc_by_pid.get(paper_id)
            author = (
                document.metadata.get("author", "") if document is not None else ""
            )
            authors_by_paper[paper_id] = split_author_list(author)
            data_samples_by_paper[paper_id] = [
                sample["description"]
                for sample in extraction.get("data_samples", [])
            ]
            results_by_paper[paper_id] = [
                result["text"] for result in extraction.get("results", [])
            ]
        family_report = analyze_evidence_families(
            authors_by_paper,
            data_samples_by_paper,
            results_by_paper,
        )

        def _independent(paper_a: str, paper_b: str) -> bool:
            """Two papers are independent only if no dataset/team overlaps."""
            family_a = family_report.family_index.get(paper_a)
            family_b = family_report.family_index.get(paper_b)
            if family_a is None or family_b is None:
                return True
            return family_a != family_b

        review_like = set(family_report.review_like_papers)

        source_cards = []
        for relation_result in relation_results:
            for relation in relation_result.relations:
                judgment_type = (
                    "A" if relation.relation_type in RELATIONS_REQUIRING_TARGET_EVIDENCE
                    else "B"
                )
                independent = _independent(
                    relation.source_paper_id, relation.target_paper_id
                )
                # The supporting bucket holds only the source paper's evidence.
                # One paper cannot be "多项相对独立的研究" (§10.2), so count what
                # that bucket actually carries: distinct source papers.
                supporting_papers = len({relation.source_paper_id})
                tier, tier_reason = classify_evidence_tier(
                    relation.relation_type,
                    source_independent=independent,
                    supporting_papers=supporting_papers,
                    source_is_review=(
                        relation.source_paper_id in review_like
                        or relation.target_paper_id in review_like
                    ),
                )
                try:
                    source_cards.append(build_source_card(
                        snapshot_id=snapshot_id,
                        stable_key=relation.relation_id,
                        judgment=(
                            f"{relation.source_paper_id} --{relation.relation_type}--> "
                            f"{relation.target_paper_id}: {relation.statement}"
                        ),
                        judgment_type=judgment_type,
                        supporting=tuple(
                            _ref(relation.source_paper_id, relation.source_evidence_ids)
                        ),
                        opposing=tuple(
                            _ref(relation.target_paper_id, relation.target_evidence_ids)
                        ),
                        time_attribute="",
                        applicability=relation.comparability_note,
                        uncertainty=relation.uncertainty,
                        tier=tier,
                        tier_reason=tier_reason,
                    ))
                except ValueError:
                    continue

        debate_cards = []
        for comparison in comparisons:
            for controversy in comparison.controversies:
                positions = [
                    DebatePosition(
                        paper_id=str(position.get("paper_id", "")),
                        stance=str(position.get("stance", "")),
                        evidence=tuple(_ref(
                            str(position.get("paper_id", "")),
                            list(position.get("evidence", [])),
                        )),
                    )
                    for position in controversy.get("positions", [])
                ]
                try:
                    debate_cards.append(build_debate_card(
                        snapshot_id=snapshot_id,
                        question=str(controversy.get("statement", "")),
                        direction_ids=(comparison.direction_id,),
                        positions=tuple(positions),
                    ))
                except ValueError:
                    continue

        direction_papers = {
            direction.direction_id: direction.paper_ids
            for direction in directions
        }
        threads_by_direction = {
            direction.direction_id: reconstruction.threads
            for direction, reconstruction in zip(directions, reconstructions)
        }
        frontier_report = build_frontier_report(
            direction_papers=direction_papers,
            years_by_id=years,
            threads_by_direction=threads_by_direction,
            family_index=family_report.family_index,
            current_year=self.current_year,
        )

        merge_inputs: list[dict[str, Any]] = []
        for entry in frontier_report.frontier_threads:
            reconstruction = next(
                candidate
                for candidate in reconstructions
                if candidate.direction_id == entry["direction_id"]
            )
            thread = next(
                (
                    candidate
                    for candidate in reconstruction.threads
                    if candidate.canonical_name == entry["thread"]
                ),
                None,
            )
            enriched = dict(entry)
            enriched["evidence"] = (
                list(thread.evidence_ids) if thread is not None else []
            )
            merge_inputs.append(enriched)

        # name vectors keyed exactly as the merge expects (`direction_id::name`)
        # so the same thread expressed with slightly different names across
        # directions merges semantically rather than staying duplicated.
        unique_pairs = sorted(
            {
                (str(entry.get("direction_id", "")), str(entry.get("thread", "")))
                for entry in merge_inputs
            }
        )
        merge_name_vectors: dict[str, np.ndarray] = {}
        if unique_pairs:
            encoded_names = self.embedding_backend.encode(
                [thread_name for _, thread_name in unique_pairs]
            )
            merge_name_vectors = {
                f"{direction_id}::{thread_name}": vector
                for (direction_id, thread_name), vector in zip(
                    unique_pairs, encoded_names
                )
            }
        merged_frontiers = merge_frontier_signals(
            merge_inputs, name_vectors=merge_name_vectors
        )

        future_work = {
            paper_id: [
                item["text"] for item in extraction.get("limitations", [])
            ]
            for paper_id, extraction in extraction_by_id.items()
        }

        parse_failures = tuple(
            document.filename
            for document in snapshot.documents
            if document.status not in ("parsed", "duplicate")
        )
        duplicates = tuple(
            document.filename
            for document in snapshot.documents
            if document.status == "duplicate"
        )
        known_years = [year for year in years.values() if year]
        facts = CorpusFacts(
            document_count=len(snapshot.documents),
            parsed_count=len(snapshot.included_documents),
            paper_count=len(snapshot.unique_paper_ids),
            year_min=min(known_years) if known_years else 0,
            year_max=max(known_years) if known_years else 0,
            parse_failures=parse_failures,
            duplicates=duplicates,
        )

        timeline_bundle = normalize_events(
            relation_results,
            reconstructions,
            merged_frontiers,
            extraction_by_id,
            years,
            snapshot_id,
            analysis_version="",
        )


        graph_projection = None
        graph_projection_error: str | None = None
        if self.graph_projector is not None:
            projection_payload = build_projection_payload(
                snapshot_id, directions, list(extraction_by_id.values()),
                paper_relations=[
                    relation
                    for relation_result in relation_results
                    for relation in relation_result.relations
                ],
            )
            try:
                graph_projection = self.graph_projector.project(
                    snapshot, projection_payload
                )
            except Exception as exc:
                # Fail soft. The analysis above is backed by the authority layer
                # (PDF/evidence/SQLite); the graph is a rebuildable projection and
                # must never be able to discard a completed run or its report.
                graph_projection_error = f"{type(exc).__name__}: {exc}"

        return LandscapeResult(
            snapshot_id=snapshot_id,
            generated_at=generated_at,
            corpus=facts,
            citations=author_year_map(doc_by_pid, years),
            directions=tuple(directions),
            comparisons=tuple(comparisons),
            reconstructions=tuple(reconstructions),
            family_report=family_report,
            frontier_report=frontier_report,
            merged_frontiers=tuple(merged_frontiers),
            future_work=future_work,
            graph_projection=graph_projection,
            graph_projection_error=graph_projection_error,
            relation_results=tuple(relation_results),
            analysis_version=new_version(
                snapshot_id=snapshot_id,
                config={
                    "n_clusters": self.n_clusters or default_cluster_count(len(items)),
                    "current_year": self.current_year,
                },
                model=getattr(self.client, "model", "unknown"),
                prompt_version="paper-v0.4-hint-anchor",
            ),
            debate_cards=tuple(debate_cards),
            source_cards=tuple(source_cards),
            timeline_bundle=timeline_bundle,
        )
