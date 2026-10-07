"""FR-05 stage reconstruction via concept-thread timelines and bounded LLM induction.

Paper-level similarity cannot segment stages (validated probes). Instead occurrences
are merged, on their NAMES, into concept threads (exact normalized name or name-vector
cosine), producing an evidence-bound timeline. The model receives ONE direction's
timeline and proposes a small number of stages; code validates chronology, roster
membership, evidence and the propagation gate (a stage carried by one paper is an
unpropagated thread, not an established stage).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from .direction_naming import ResearchDirection
from .llm_client import is_retryable_completion
from .semantic_items import SemanticItem

VALID_CHANGE_DIMENSIONS = {
    "question", "theory", "method", "data", "object", "context", "conclusion"
}

_SYSTEM = """You reconstruct the developmental stages of ONE research direction from an evidence-bound
timeline of concept/method threads. Each thread shows the years it appears and how many
distinct papers carry it. Propose a SMALL number of stages (2-4; never one stage per year)
and the turning points between them.
Rules:
- Use only the given papers and evidence ids. Never invent a paper.
- Boundaries must be justified by what actually changed: question, theory, method, data,
  object, context or conclusion.
- The first stage starts at the earliest year IN THIS CORPUS. Phrase it as "the earliest
  verifiable work in this corpus", never "historically first".
- If material is too sparse or work looks continuous, set material_assessment to sparse or
  insufficient and return fewer/no stages.
Return JSON only."""


def _normalize_thread_name(name: str) -> str:
    lowered = name.lower().strip()
    lowered = re.sub(r"\([^)]*\)", "", lowered)
    return re.sub(r"[\s\-–—,.:;_]+", " ", lowered).strip()


@dataclass(frozen=True)
class ConceptThread:
    thread_id: str
    canonical_name: str
    variants: tuple[str, ...]
    paper_ids: tuple[str, ...]
    years: tuple[int, ...]
    earliest_year: int
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Stage:
    label: str
    year_start: int
    year_end: int
    change_dimension: str
    summary: str
    key_paper_ids: tuple[str, ...]
    evidence: tuple[str, ...]
    confidence: str
    propagated: bool


@dataclass(frozen=True)
class StageReconstruction:
    direction_id: str
    material_assessment: str
    stages: tuple[Stage, ...]
    turning_points: tuple[dict[str, Any], ...]
    notes: str
    threads: tuple[ConceptThread, ...]
    validation_errors: tuple[str, ...]
    ignored_evidence_ids: tuple[str, ...]
    # Records bounded-retry attempts and the thread-chunked fallback when the
    # full-timeline answer does not fit the completion ceiling. Callers must be
    # able to see that the stages came from chunks rather than one answer.
    extraction_trace: tuple[dict[str, Any], ...] = ()


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_concept_threads(
    items: Sequence[SemanticItem],
    name_vectors: np.ndarray,
    years_by_id: dict[str, int],
    cos_threshold: float = 0.90,
) -> list[ConceptThread]:
    matrix = np.asarray(name_vectors)
    similarity = matrix @ matrix.T
    normalized = [_normalize_thread_name(item.name) for item in items]

    dsu = _DSU(len(items))
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            same_name = bool(normalized[a]) and normalized[a] == normalized[b]
            if same_name or similarity[a][b] >= cos_threshold:
                dsu.union(a, b)

    groups: dict[int, list[int]] = {}
    for index in range(len(items)):
        groups.setdefault(dsu.find(index), []).append(index)

    threads: list[ConceptThread] = []
    for group_index, members in enumerate(sorted(groups.values(), key=lambda g: min(g))):
        names = [items[index].name for index in members]
        canonical = min(set(names), key=lambda x: (len(x), names.index(x)))
        paper_ids = sorted({items[index].paper_id for index in members})
        years = sorted(
            {
                years_by_id.get(items[index].paper_id)
                for index in members
                if years_by_id.get(items[index].paper_id) is not None
            }
        )
        evidence = tuple(
            sorted(
                {
                    evidence_id
                    for index in members
                    for evidence_id in items[index].evidence_ids
                }
            )
        )
        threads.append(
            ConceptThread(
                thread_id=f"thread_{group_index:02d}",
                canonical_name=canonical,
                variants=tuple(sorted(set(names))),
                paper_ids=tuple(paper_ids),
                years=tuple(years),
                earliest_year=years[0] if years else 0,
                evidence_ids=evidence,
            )
        )
    threads.sort(key=lambda thread: (thread.earliest_year, thread.canonical_name))
    return threads


def _build_timeline_text(threads: Sequence[ConceptThread]) -> str:
    lines: list[str] = []
    for thread in threads:
        lines.append(
            f"- {thread.earliest_year} {thread.canonical_name} "
            f"[papers={len(thread.paper_ids)}, years={list(thread.years)}]"
        )
        if thread.evidence_ids:
            lines.append(
                f"    evidence: {', '.join(thread.evidence_ids)}"
            )
    return "\n".join(lines)


def _build_stage_prompt(
    label: str,
    papers: str,
    timeline_text: str,
    retrieval_context: str = "",
) -> str:
    context = ""
    if retrieval_context:
        context = (
            "\n\nRETRIEVED DISCOVERY CONTEXT (not authoritative evidence; "
            "do not cite it in output):\n"
            f"{retrieval_context}"
        )
    return (
        f"RESEARCH DIRECTION: {label}\n"
        f"PAPERS: {papers}\n\n"
        f"THREAD TIMELINE:\n{timeline_text}{context}\n\n"
        "Use ONLY the evidence ids listed under each thread; do not guess other ids. "
        "The id numbering may be non-contiguous, so never infer an unlisted id.\n\n"
        'Return exactly: {"material_assessment": "sufficient|sparse|insufficient",'
        ' "stages": [{"label": "...", "year_start": 2020, "year_end": 2021,'
        ' "change_dimension": "question|theory|method|data|object|context|conclusion",'
        ' "summary": "1-3 sentences", "key_paper_ids": ["paper_id"],'
        ' "evidence": ["paper_id::E1"], "confidence": "high|medium|low"}],'
        ' "turning_points": [{"year": 2024, "paper_id": "paper_id",'
        ' "statement": "...", "evidence": ["paper_id::E1"]}], "notes": ""}'
    )


class StageInductionClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


def reconstruct_stages(
    direction: ResearchDirection,
    items: Sequence[SemanticItem],
    name_vectors: np.ndarray,
    years_by_id: dict[str, int],
    client: StageInductionClient,
    cos_threshold: float = 0.90,
    retrieval_context: str = "",
) -> StageReconstruction:
    roster = set(direction.paper_ids)
    aligned = [
        (item, index)
        for index, item in enumerate(items)
        if item.paper_id in roster
    ]
    direction_items = [pair[0] for pair in aligned]
    direction_name_vectors = np.asarray(
        [np.asarray(name_vectors)[pair[1]] for pair in aligned]
    )

    threads = build_concept_threads(
        direction_items,
        direction_name_vectors,
        years_by_id,
        cos_threshold=cos_threshold,
    )
    allowed_evidence = {
        evidence_id
        for thread in threads
        for evidence_id in thread.evidence_ids
    }

    timeline_text = _build_timeline_text(threads)
    user_prompt = _build_stage_prompt(
        direction.label,
        ", ".join(sorted(roster)),
        timeline_text,
        retrieval_context,
    )

    payload, trace, chunk_errors = _call_with_fallback(
        client, threads, direction, roster, retrieval_context, user_prompt
    )

    errors: list[str] = list(chunk_errors)
    ignored_evidence: list[str] = []
    stages: list[Stage] = []
    previous_end: int | None = None
    for index, raw_stage in enumerate(payload.get("stages", [])):
        year_start = raw_stage.get("year_start")
        year_end = raw_stage.get("year_end")
        if not isinstance(year_start, int) or not isinstance(year_end, int):
            errors.append(f"stage[{index}] bad year range")
        else:
            if year_start > year_end:
                errors.append(f"stage[{index}] bad year range")
            if previous_end is not None and year_start < previous_end:
                errors.append(f"stage[{index}] stages not in chronological order")
            previous_end = year_end

        dimension = raw_stage.get("change_dimension")
        if dimension not in VALID_CHANGE_DIMENSIONS:
            errors.append(f"stage[{index}] invalid change dimension {dimension!r}")

        key_paper_ids = tuple(raw_stage.get("key_paper_ids", []))
        valid_evidence = tuple(
            evidence_id
            for evidence_id in raw_stage.get("evidence", [])
            if evidence_id in allowed_evidence
        )
        for paper_id in key_paper_ids:
            if paper_id not in roster:
                errors.append(f"stage[{index}] paper not in roster: {paper_id}")
        for evidence_id in raw_stage.get("evidence", []):
            if evidence_id not in allowed_evidence:
                ignored_evidence.append(evidence_id)

        stages.append(
            Stage(
                label=str(raw_stage.get("label", "")),
                year_start=year_start if isinstance(year_start, int) else 0,
                year_end=year_end if isinstance(year_end, int) else 0,
                change_dimension=str(dimension),
                summary=str(raw_stage.get("summary", "")),
                key_paper_ids=key_paper_ids,
                evidence=valid_evidence,
                confidence=str(raw_stage.get("confidence", "")),
                propagated=len(set(key_paper_ids)) >= 2,
            )
        )

    turning_points: list[dict[str, Any]] = []
    for index, raw_tp in enumerate(payload.get("turning_points", [])):
        paper_id = raw_tp.get("paper_id")
        if paper_id not in roster:
            errors.append(f"turning_point[{index}] paper not in roster: {paper_id}")
        tp_evidence = []
        for evidence_id in raw_tp.get("evidence", []):
            if evidence_id in allowed_evidence:
                tp_evidence.append(evidence_id)
            else:
                ignored_evidence.append(evidence_id)
        sanitized = dict(raw_tp)
        sanitized["evidence"] = tp_evidence
        turning_points.append(sanitized)

    return StageReconstruction(
        direction_id=direction.direction_id,
        material_assessment=str(payload.get("material_assessment", "sparse")),
        stages=tuple(stages),
        turning_points=tuple(turning_points),
        notes=str(payload.get("notes", "")),
        threads=tuple(threads),
        validation_errors=tuple(errors),
        ignored_evidence_ids=tuple(ignored_evidence),
        extraction_trace=tuple(trace),
    )


# ---------------------------------------------------------------------------
# Truncation resilience
# ---------------------------------------------------------------------------

_STAGE_MAX_ATTEMPTS = 3
_STAGE_CHUNK_SIZE = 2


def iter_thread_chunks(
    threads: Sequence[ConceptThread], chunk_size: int = _STAGE_CHUNK_SIZE
) -> list[list[ConceptThread]]:
    """Deterministic chunks of threads covering every thread exactly once."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    chunks: list[list[ConceptThread]] = []
    ordered = list(threads)
    for start in range(0, len(ordered), chunk_size):
        chunks.append(ordered[start : start + chunk_size])
    return chunks


def _merge_chunk_payloads(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fold per-chunk stage answers into one payload.

    Stages and turning points are concatenated in chunk order (which follows
    the thread timeline, so the result stays chronological), notes are joined,
    and material_assessment takes the weakest verdict seen.
    """
    rank = {"sufficient": 2, "sparse": 1, "insufficient": 0}
    stages: list[Any] = []
    turning_points: list[Any] = []
    notes: list[str] = []
    weakest = "sufficient"
    for payload in payloads:
        stages.extend(payload.get("stages", []))
        turning_points.extend(payload.get("turning_points", []))
        if payload.get("notes"):
            notes.append(str(payload["notes"]))
        verdict = str(payload.get("material_assessment", "sparse"))
        if rank.get(verdict, 1) < rank.get(weakest, 1):
            weakest = verdict
    return {
        "material_assessment": weakest,
        "stages": stages,
        "turning_points": turning_points,
        "notes": " | ".join(notes),
    }


def _call_with_fallback(
    client: StageInductionClient,
    threads: Sequence[ConceptThread],
    direction: ResearchDirection,
    roster: set[str],
    retrieval_context: str,
    user_prompt: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], list[str]]:
    """Bounded retry, then thread-chunked escalation.

    Truncation is provider-side variance (see ``ModelTruncationError``), so a
    bounded retry recovers real work. When retries cannot fit the answer — a
    direction whose full thread timeline is too big — the request is re-run as
    deterministic chunks of threads that cover every thread exactly once.
    """
    attempts = 0
    last_error: BaseException | None = None
    while attempts < _STAGE_MAX_ATTEMPTS:
        attempts += 1
        try:
            return (
                client.complete_json(_SYSTEM, user_prompt),
                ({"fallback": "none", "attempts": attempts},),
                [],
            )
        except Exception as exc:  # noqa: BLE001 - provider errors vary by backend
            last_error = exc
            if not is_retryable_completion(exc):
                break

    if not isinstance(last_error, Exception) or not is_retryable_completion(last_error):
        raise last_error if last_error else RuntimeError("stage reconstruction failed")

    chunks = iter_thread_chunks(threads)
    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for chunk in chunks:
        chunk_prompt = _build_stage_prompt(
            direction.label,
            ", ".join(sorted(roster)),
            _build_timeline_text(chunk),
            retrieval_context,
        )
        chunk_attempts = 0
        chunk_payload: dict[str, Any] | None = None
        while chunk_attempts < _STAGE_MAX_ATTEMPTS:
            chunk_attempts += 1
            try:
                chunk_payload = client.complete_json(_SYSTEM, chunk_prompt)
                break
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                if not is_retryable_completion(exc):
                    errors.append(
                        f"thread chunk [{chunk[0].thread_id if chunk else '?'}] "
                        f"failed: {type(exc).__name__}: {exc}"
                    )
                    break
        if chunk_payload is not None:
            payloads.append(chunk_payload)

    if not payloads:
        raise last_error

    trace = ({
        "fallback": "thread_chunked",
        "attempts": attempts,
        "chunks": len(chunks),
        "chunks_succeeded": len(payloads),
        "chunk_size": _STAGE_CHUNK_SIZE,
    },)
    return _merge_chunk_payloads(payloads), trace, errors
