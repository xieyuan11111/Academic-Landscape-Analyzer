"""Tests for FR-05 stage reconstruction: concept threads and bounded stage induction."""

from __future__ import annotations

from typing import Any

import numpy as np

from academic_landscape.semantic_items import SemanticItem
from academic_landscape.direction_naming import ResearchDirection
from academic_landscape.llm_client import ModelTruncationError
from academic_landscape.stage_reconstruction import (
    build_concept_threads,
    reconstruct_stages,
)

SNAPSHOT_ID = "snapshot_test"


def _item(item_id: str, paper_id: str, name: str) -> SemanticItem:
    return SemanticItem(
        item_id=item_id,
        snapshot_id=SNAPSHOT_ID,
        paper_id=paper_id,
        item_type="concept",
        name=name,
        description=f"desc of {name}",
        evidence_ids=(f"{paper_id}::E1",),
    )


def _direction(papers: list[str]) -> ResearchDirection:
    return ResearchDirection(
        direction_id="dir_1",
        cluster_id=1,
        label="Direction one",
        description="",
        status="emerging",
        paper_ids=tuple(papers),
        item_ids=(),
    )


def _thread_scenario() -> tuple[list[SemanticItem], np.ndarray, dict[str, int]]:
    items = [
        _item("t1", "paper_a", "HNAD"),
        _item("t2", "paper_b", "Heterotrophic nitrification aerobic denitrification"),
        _item("t3", "paper_a", "qPCR"),
        _item("t4", "paper_c", "Quantitative PCR"),
        _item("t5", "paper_b", "free ammonia"),
    ]
    name_vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.98, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.99, 0.05],
            [0.0, 0.0, 1.0],
        ]
    )
    name_vectors = name_vectors / np.linalg.norm(
        name_vectors, axis=1, keepdims=True
    )
    years = {"paper_a": 2020, "paper_b": 2024, "paper_c": 2021}
    return items, name_vectors, years


# ---------- concept thread building (data side) ----------


def test_threads_merge_equivalent_names() -> None:
    items, name_vectors, years = _thread_scenario()
    threads = build_concept_threads(items, name_vectors, years, cos_threshold=0.90)
    assert len(threads) == 3


def test_thread_membership_and_propagation() -> None:
    items, name_vectors, years = _thread_scenario()
    threads = build_concept_threads(items, name_vectors, years, cos_threshold=0.90)
    hnad = next(t for t in threads if "HNAD" in t.canonical_name)
    assert set(hnad.paper_ids) == {"paper_a", "paper_b"}
    assert hnad.earliest_year == 2020
    assert hnad.years == (2020, 2024)


def test_qpcr_thread_variants() -> None:
    items, name_vectors, years = _thread_scenario()
    threads = build_concept_threads(items, name_vectors, years, cos_threshold=0.90)
    qpcr = next(t for t in threads if "qPCR" in t.canonical_name or "Quantitative PCR" in t.canonical_name)
    assert set(qpcr.paper_ids) == {"paper_a", "paper_c"}
    assert set(qpcr.variants) == {"qPCR", "Quantitative PCR"}


def test_thread_evidence_aggregated() -> None:
    items, name_vectors, years = _thread_scenario()
    threads = build_concept_threads(items, name_vectors, years, cos_threshold=0.90)
    hnad = next(t for t in threads if "HNAD" in t.canonical_name)
    assert set(hnad.evidence_ids) == {"paper_a::E1", "paper_b::E1"}


def test_single_paper_thread_marked_unpropagated() -> None:
    items, name_vectors, years = _thread_scenario()
    threads = build_concept_threads(items, name_vectors, years, cos_threshold=0.90)
    fa = next(t for t in threads if "free ammonia" in t.canonical_name)
    assert fa.paper_ids == ("paper_b",)


# ---------- stage induction (bounded LLM) ----------


class ScriptedClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append((system_prompt, user_prompt))
        return self.payload


def _two_stage_payload() -> dict[str, Any]:
    return {
        "material_assessment": "sufficient",
        "stages": [
            {"label": "Early methods", "year_start": 2020, "year_end": 2021,
             "change_dimension": "method",
             "summary": "Methods established",
             "key_paper_ids": ["paper_a", "paper_c"],
             "evidence": ["paper_a::E1", "paper_c::E1"],
             "confidence": "medium"},
            {"label": "Recent turn", "year_start": 2024, "year_end": 2024,
             "change_dimension": "object",
             "summary": "New objects",
             "key_paper_ids": ["paper_b"],
             "evidence": ["paper_b::E1"],
             "confidence": "low"},
        ],
        "turning_points": [
            {"year": 2024, "paper_id": "paper_b",
             "statement": "new objects arrived", "evidence": ["paper_b::E1"]}
        ],
        "notes": "",
    }


def test_stage_induction_happy_path() -> None:
    items, name_vectors, years = _thread_scenario()
    direction = _direction(["paper_a", "paper_b", "paper_c"])
    client = ScriptedClient(_two_stage_payload())
    reconstruction = reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.90
    )
    assert len(reconstruction.stages) == 2
    assert not reconstruction.validation_errors
    assert len(client.calls) == 1


def test_stage_propagation_gate() -> None:
    items, name_vectors, years = _thread_scenario()
    direction = _direction(["paper_a", "paper_b", "paper_c"])
    client = ScriptedClient(_two_stage_payload())
    reconstruction = reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.90
    )
    first, second = reconstruction.stages
    assert first.propagated is True
    assert second.propagated is False


def test_stage_call_sees_thread_timeline_not_descriptions() -> None:
    items, name_vectors, years = _thread_scenario()
    direction = _direction(["paper_a", "paper_b", "paper_c"])
    client = ScriptedClient(_two_stage_payload())
    reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.90
    )
    user_prompt = client.calls[0][1]
    assert "HNAD" in user_prompt
    # descriptions are excluded from thread timelines
    assert "desc of" not in user_prompt


def test_stage_call_lists_evidence_ids_per_thread() -> None:
    items, name_vectors, years = _thread_scenario()
    direction = _direction(["paper_a", "paper_b", "paper_c"])
    client = ScriptedClient(_two_stage_payload())
    reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.90
    )
    user_prompt = client.calls[0][1]
    # the model must be able to cite real ids without guessing E1
    assert "paper_a::E1" in user_prompt
    assert "paper_b::E1" in user_prompt


def test_non_chronological_stages_are_errors() -> None:
    items, name_vectors, years = _thread_scenario()
    direction = _direction(["paper_a", "paper_b", "paper_c"])
    payload = _two_stage_payload()
    payload["stages"][1]["year_start"] = 2010
    client = ScriptedClient(payload)
    reconstruction = reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.90
    )
    assert any("order" in error for error in reconstruction.validation_errors)


def test_unknown_stage_evidence_is_dropped() -> None:
    items, name_vectors, years = _thread_scenario()
    direction = _direction(["paper_a", "paper_b", "paper_c"])
    payload = _two_stage_payload()
    payload["stages"][0]["evidence"].append("paper_a::E99")
    client = ScriptedClient(payload)
    reconstruction = reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.90
    )
    stage = reconstruction.stages[0]
    # invalid id is removed from the stage but recorded for audit
    assert "paper_a::E99" not in stage.evidence
    assert "paper_a::E99" in reconstruction.ignored_evidence_ids
    assert not reconstruction.validation_errors


def test_stage_paper_outside_roster_is_error() -> None:
    items, name_vectors, years = _thread_scenario()
    direction = _direction(["paper_a", "paper_b", "paper_c"])
    payload = _two_stage_payload()
    payload["stages"][0]["key_paper_ids"].append("paper_z")
    client = ScriptedClient(payload)
    reconstruction = reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.90
    )
    assert any("paper_z" in error for error in reconstruction.validation_errors)


def test_insufficient_material_allowed() -> None:
    items, name_vectors, years = _thread_scenario()
    direction = _direction(["paper_a", "paper_b", "paper_c"])
    payload = {
        "material_assessment": "insufficient",
        "stages": [],
        "turning_points": [],
        "notes": "not enough density",
    }
    client = ScriptedClient(payload)
    reconstruction = reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.90
    )
    assert reconstruction.stages == ()
    assert not reconstruction.validation_errors


# ---------- truncation resilience (bounded retry + thread-chunk escalation) ----------


class TruncatingThenSucceedingClient:
    """Truncates the first ``failures`` calls, then returns ``payload``."""

    def __init__(self, payload: dict[str, Any], failures: int) -> None:
        self.payload = payload
        self.failures = failures
        self.calls: list[str] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append(user_prompt)
        if len(self.calls) <= self.failures:
            raise ModelTruncationError("model completion was truncated at 8192 tokens")
        return self.payload


class AlwaysTruncatingExceptSmallChunksClient:
    """Succeeds only when the prompt carries at most ``chunk_max`` threads.

    Simulates a direction whose full timeline answer never fits the completion
    ceiling but whose per-chunk answers do.
    """

    def __init__(self, chunk_max: int) -> None:
        self.chunk_max = chunk_max
        self.calls: list[str] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls.append(user_prompt)
        if user_prompt.count("evidence:") > self.chunk_max:
            raise ModelTruncationError("model completion was truncated at 8192 tokens")
        return _two_stage_payload()


def _wide_scenario(thread_count: int) -> tuple[list[SemanticItem], np.ndarray, dict[str, int]]:
    items: list[SemanticItem] = []
    vectors: list[list[float]] = []
    years: dict[str, int] = {}
    for i in range(thread_count):
        pid = f"paper_{i}"
        items.append(_item(f"t{i}", pid, f"concept {i}"))
        base = [0.0] * thread_count
        base[i] = 1.0
        vectors.append(base)
        years[pid] = 2020 + i
    name_vectors = np.asarray(vectors, dtype=float)
    name_vectors = name_vectors / np.linalg.norm(name_vectors, axis=1, keepdims=True)
    return items, name_vectors, years


def test_stage_truncation_is_retried_then_succeeds() -> None:
    items, name_vectors, years = _thread_scenario()
    direction = _direction(["paper_a", "paper_b", "paper_c"])
    client = TruncatingThenSucceedingClient(_two_stage_payload(), failures=2)
    reconstruction = reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.90
    )
    assert len(client.calls) == 3
    assert len(reconstruction.stages) == 2
    assert not reconstruction.validation_errors


def test_stage_truncation_exhausts_retry_and_escalates_to_thread_chunks() -> None:
    # 6 independent threads; the client only answers when a prompt carries
    # at most 2 thread evidence blocks, so the full call must be chunked.
    items, name_vectors, years = _wide_scenario(6)
    papers = [f"paper_{i}" for i in range(6)]
    direction = _direction(papers)
    client = AlwaysTruncatingExceptSmallChunksClient(chunk_max=2)
    reconstruction = reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.99
    )
    # full-attempt retries (3) come before any chunk call
    assert len(client.calls) > 3
    assert reconstruction.stages, "chunked fallback must still yield stages"
    assert reconstruction.threads, "threads must survive the fallback path"


def test_stage_escalated_calls_cover_every_thread_once() -> None:
    items, name_vectors, years = _wide_scenario(6)
    direction = _direction([f"paper_{i}" for i in range(6)])
    client = AlwaysTruncatingExceptSmallChunksClient(chunk_max=2)
    reconstruct_stages(direction, items, name_vectors, years, client, cos_threshold=0.99)
    chunk_prompts = client.calls[3:]  # after the bounded retries
    seen = 0
    for prompt in chunk_prompts:
        seen += prompt.count("evidence:")
    assert seen == len(items), f"chunks must cover all threads once, saw {seen}"


def test_stage_escalation_recorded_on_reconstruction() -> None:
    items, name_vectors, years = _wide_scenario(6)
    direction = _direction([f"paper_{i}" for i in range(6)])
    client = AlwaysTruncatingExceptSmallChunksClient(chunk_max=2)
    reconstruction = reconstruct_stages(
        direction, items, name_vectors, years, client, cos_threshold=0.99
    )
    assert reconstruction.extraction_trace, "fallback must be visible to callers"
    assert reconstruction.extraction_trace[0]["fallback"] == "thread_chunked"
