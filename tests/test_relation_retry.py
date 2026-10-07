"""Truncation must be detected, retried, and visible (audit finding D10).

Root cause (measured, `runtime/probe_d10_rootcause.py`): the relation prompt asks
for relations across every paper pair of a direction (up to 91 pairs / 14
papers). Sometimes the model finishes in ~4k tokens; sometimes it runs to the
8192-token ceiling and the JSON is cut mid-string. `complete_json` reported that
as a generic "not valid JSON", `extract_paper_relations` recorded one error and
returned zero relations, and the report never mentioned that a whole direction
lost its relation layer.

The failure is non-deterministic, so a bounded retry is effective — but only if
truncation is a *distinguishable* condition and failures are *recorded*, not
swallowed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from academic_landscape.llm_client import (
    ModelClientError,
    ModelTruncationError,
    OpenAICompatibleClient,
)
from academic_landscape.relation_extraction import (
    RELATION_EXTRACTION_FAILED,
    extract_paper_relations,
)

# ---------------------------------------------------------------------------
# fake OpenAI SDK
# ---------------------------------------------------------------------------


class _FakeSDK:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        spec = self._responses.pop(0) if self._responses else {"content": "{}"}
        content = spec.get("content", "{}")
        choice = SimpleNamespace(
            finish_reason=spec.get("finish_reason", "stop"),
            message=SimpleNamespace(content=content),
        )
        usage = SimpleNamespace(
            prompt_tokens=spec.get("prompt_tokens", 10),
            completion_tokens=spec.get("completion_tokens", 20),
        )
        return SimpleNamespace(choices=[choice], usage=usage)


def _client(sdk: _FakeSDK) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        "https://example.invalid/v1", "test-key", "test-model", client=sdk
    )


# ---------------------------------------------------------------------------
# llm_client: truncation is a distinct, diagnosable condition
# ---------------------------------------------------------------------------


def test_truncated_response_raises_truncation_error() -> None:
    sdk = _FakeSDK([{
        "finish_reason": "length",
        "completion_tokens": 8192,
        "content": '{"relations": [{"source_paper_id": "a", "state',
    }])
    with pytest.raises(ModelTruncationError) as info:
        _client(sdk).complete_json("system", "user")

    assert isinstance(info.value, ModelClientError)
    assert "8192" in str(info.value)


def test_invalid_json_that_is_not_truncated_stays_a_generic_error() -> None:
    sdk = _FakeSDK([{"finish_reason": "stop", "content": "not json at all"}])
    with pytest.raises(ModelClientError) as info:
        _client(sdk).complete_json("system", "user")
    assert not isinstance(info.value, ModelTruncationError)


def test_max_tokens_is_sent_explicitly() -> None:
    """Relying on the provider default is what made the ceiling invisible."""
    sdk = _FakeSDK([{"content": "{}"}])
    _client(sdk).complete_json("system", "user")
    assert sdk.requests[0].get("max_tokens"), "client must send an explicit max_tokens"


# ---------------------------------------------------------------------------
# relation extraction: bounded retry + visible failure
# ---------------------------------------------------------------------------


class _Direction:
    def __init__(self, paper_ids: tuple[str, ...]) -> None:
        self.direction_id = "dir_1"
        self.cluster_id = 0
        self.label = "test direction"
        self.description = ""
        self.status = "candidate"
        self.paper_ids = paper_ids
        self.item_ids = ()


def _extraction(paper_id: str) -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "research_question": "q",
        "claims": [],
        "evidence": [
            {
                "evidence_id": "E1",
                "quote": f"quote about {paper_id}",
                "page_number": 1,
                "evidence_type": "B",
            }
        ],
    }


_RELATION = {
    "source_paper_id": "paper_a",
    "target_paper_id": "paper_b",
    "relation_type": "supports",
    "statement": "paper b supports paper a",
    "source_evidence_ids": ["paper_a::E1"],
    "target_evidence_ids": [],
    "difference_dimension": None,
    "comparability_note": "",
    "uncertainty": "",
}


class _FlakyClient:
    """Fails `failures` times with truncation, then behaves."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls += 1
        if self.calls <= self.failures:
            raise ModelTruncationError("model completion was truncated at 8192 tokens")
        return {"relations": [dict(_RELATION)]}


def _run(client: Any, failures_note: str = ""):
    return extract_paper_relations(
        client,
        [_Direction(("paper_a", "paper_b"))],
        {"paper_a": _extraction("paper_a"), "paper_b": _extraction("paper_b")},
        {"paper_a": 2020, "paper_b": 2021},
        snapshot_id="snapshot_x",
    )


def test_relation_extraction_retries_after_truncation() -> None:
    client = _FlakyClient(failures=1)
    results = _run(client)

    assert client.calls >= 2, "a truncated call must be retried"
    assert len(results[0].relations) == 1


def test_relation_extraction_retries_are_bounded() -> None:
    client = _FlakyClient(failures=99)
    results = _run(client)

    assert client.calls <= 5, f"retries must be bounded, saw {client.calls}"
    assert results[0].relations == ()


def test_exhausted_retries_are_recorded_as_a_visible_failure() -> None:
    client = _FlakyClient(failures=99)
    results = _run(client)

    codes = [error.get("code") for error in results[0].validation_errors]
    assert RELATION_EXTRACTION_FAILED in codes, results[0].validation_errors
    failure = next(
        error for error in results[0].validation_errors
        if error.get("code") == RELATION_EXTRACTION_FAILED
    )
    assert failure["direction_id"] == "dir_1"
    assert failure["attempts"] >= 2
    assert "truncat" in str(failure["reason"]).lower()


def test_happy_path_still_makes_exactly_one_call() -> None:
    client = _FlakyClient(failures=0)
    results = _run(client)

    assert client.calls == 1, "retry must not add cost when nothing fails"
    assert len(results[0].relations) == 1


# ---------------------------------------------------------------------------
# visibility: the report must not stay silent about a lost relation layer
# ---------------------------------------------------------------------------


def _landscape_with(relation_results: tuple[Any, ...]) -> Any:
    from academic_landscape.landscape_report import CorpusFacts, LandscapeResult

    facts = CorpusFacts(
        document_count=2, parsed_count=2, paper_count=2,
        year_min=2020, year_max=2021, parse_failures=(), duplicates=(),
    )
    return LandscapeResult(
        snapshot_id="snapshot_x",
        generated_at="2026-09-24T00:00:00+00:00",
        corpus=facts,
        directions=(), comparisons=(), reconstructions=(),
        family_report=None, frontier_report=None,
        merged_frontiers=(), future_work={},
        relation_results=relation_results,
    )


def test_report_surfaces_relation_extraction_failures() -> None:
    from academic_landscape.landscape_report import render_landscape_report

    result = _landscape_with(tuple(_run(_FlakyClient(failures=99))))
    text = render_landscape_report(result)

    assert "关系抽取" in text, "a direction with no relations must be visible in the report"
    assert "dir_1" in text
    assert "truncat" in text.lower()


def test_report_states_coverage_when_relations_succeed() -> None:
    from academic_landscape.landscape_report import render_landscape_report

    result = _landscape_with(tuple(_run(_FlakyClient(failures=0))))
    text = render_landscape_report(result)

    assert "关系抽取" in text
    assert "1/1" in text or "1 个方向中 1" in text


# ---------------------------------------------------------------------------
# escalation: retry alone cannot save a direction whose answer never fits
# ---------------------------------------------------------------------------
#
# Measured on MAG-50 r3: `dir_65d92c166225b009` (13 papers, 78 pairs) truncated on
# all three retries. Retrying a request whose output cannot fit is not a fix, so
# the direction must fall back to bounding the work per call: enumerate every
# paper pair and evaluate small groups of pairs in separate calls.


def test_pair_chunks_cover_every_pair_exactly_once() -> None:
    import itertools

    from academic_landscape.relation_extraction import iter_pair_chunks

    papers = tuple(f"p{i}" for i in range(7))
    chunks = list(iter_pair_chunks(papers, 5))

    flat = [tuple(sorted(pair)) for chunk in chunks for pair in chunk]
    expected = {tuple(sorted(pair)) for pair in itertools.combinations(papers, 2)}

    assert set(flat) == expected, "every pair must be covered"
    assert len(flat) == len(expected), "no pair may be evaluated twice"
    assert all(len(chunk) <= 5 for chunk in chunks), "chunk size must be respected"


def test_pair_chunks_are_deterministic() -> None:
    from academic_landscape.relation_extraction import iter_pair_chunks

    papers = tuple(f"p{i}" for i in range(5))
    assert list(iter_pair_chunks(papers, 4)) == list(iter_pair_chunks(papers, 4))


def test_chunk_prompt_does_not_impose_a_per_pair_quota() -> None:
    """Scope must bound what is *considered*, not force one verdict per pair.

    Measured on MAG-50 r4: the chunked directions came back 90% `not_comparable`
    against 53% on the unbounded path, because the prompt asked for "at most one
    relation per pair" and the model filled every slot. Chunking must change which
    pairs are considered, not how selective the model is allowed to be.
    """
    from academic_landscape.relation_extraction import build_relation_prompt

    _system, user = build_relation_prompt(
        "label", "detail", "", pair_scope=(("paper_a", "paper_b"), ("paper_c", "paper_d"))
    )

    assert "ONLY these paper pairs" in user
    assert "paper_a | paper_b" in user and "paper_c | paper_d" in user
    assert "at most one relation per pair" not in user
    assert "fewer relations than pairs" in user


class _TruncatesLargeCallsClient:
    """Truncates any call asking about the whole direction; answers chunk calls.

    A chunk call is answered with a relation for the *first pair listed in that
    chunk*, so the fixture stays inside the chunk's declared scope.
    """

    def __init__(self, chunk_marker: str = "ONLY these paper pairs") -> None:
        self.chunk_marker = chunk_marker
        self.calls = 0
        self.full_calls = 0
        self.chunk_calls = 0

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        self.calls += 1
        if self.chunk_marker in user_prompt:
            self.chunk_calls += 1
            listed = user_prompt.split(self.chunk_marker, 1)[1].splitlines()[1]
            source, _, target = listed.split(",")[0].partition("|")
            source, target = source.strip(), target.strip()
            return {"relations": [{
                "source_paper_id": source,
                "target_paper_id": target,
                "relation_type": "parallel",
                "statement": "chunk-scoped relation",
                "source_evidence_ids": [f"{source}::E1"],
                "target_evidence_ids": [],
                "difference_dimension": None,
                "comparability_note": "",
                "uncertainty": "",
            }]}
        self.full_calls += 1
        raise ModelTruncationError("model completion was truncated at 8192 tokens")


class _ManyPapersDirection:
    def __init__(self, count: int) -> None:
        self.direction_id = "dir_big"
        self.cluster_id = 0
        self.label = "big direction"
        self.description = ""
        self.status = "candidate"
        self.paper_ids = tuple(f"paper_{i:02d}" for i in range(count))
        self.item_ids = ()


def _big_run(client: Any, count: int = 6):
    papers = tuple(f"paper_{i:02d}" for i in range(count))
    extractions = {
        pid: {
            "paper_id": pid,
            "research_question": "q",
            "claims": [],
            "evidence": [
                {
                    "evidence_id": "E1",
                    "quote": f"quote about {pid}",
                    "page_number": 1,
                    "evidence_type": "B",
                }
            ],
        }
        for pid in papers
    }
    years = {pid: 2020 for pid in papers}
    return extract_paper_relations(
        client, [_ManyPapersDirection(count)], extractions, years, snapshot_id="snapshot_x"
    )


def test_direction_that_always_truncates_falls_back_to_pair_chunks() -> None:
    client = _TruncatesLargeCallsClient()
    results = _big_run(client)

    assert client.full_calls == 3, "the direct path is still retried first"
    assert client.chunk_calls > 0, "it must escalate to bounded chunk calls"
    assert results[0].relations, "the direction must not lose its relation layer"
    codes = [e.get("code") for e in results[0].validation_errors]
    assert RELATION_EXTRACTION_FAILED not in codes


def test_fallback_is_recorded_in_the_extraction_trace() -> None:
    client = _TruncatesLargeCallsClient()
    results = _big_run(client)

    trace = results[0].extraction_trace
    assert trace, "the escalation must be visible in the result"
    assert trace[0]["fallback"] == "pair_chunked"
    assert trace[0]["calls"] > 0
    assert trace[0]["pairs"] == 15  # C(6,2)


def test_failure_is_still_reported_when_chunks_also_fail() -> None:
    class _AlwaysTruncates:
        def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
            raise ModelTruncationError("model completion was truncated at 8192 tokens")

    results = _big_run(_AlwaysTruncates())

    codes = [e.get("code") for e in results[0].validation_errors]
    assert RELATION_EXTRACTION_FAILED in codes, "a real loss must still be reported"


def test_report_shows_the_pair_chunk_fallback() -> None:
    from academic_landscape.landscape_report import render_landscape_report

    result = _landscape_with(tuple(_big_run(_TruncatesLargeCallsClient())))
    text = render_landscape_report(result)

    assert "配对分块" in text
    assert "dir_big" in text
    # an escalation that recovered is reported, but not as a failure
    assert "关系层为空" not in text


def test_small_direction_never_escalates() -> None:
    class _AlwaysTruncates:
        def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
            raise ModelTruncationError("truncated")

    results = _big_run(_AlwaysTruncates(), count=6)

    # escalation is attempted, but a healthy direction must not pay for it:
    client = _FlakyClient(failures=0)
    healthy = _run(client)
    assert client.calls == 1
    assert healthy[0].extraction_trace == ()
    assert results[0].extraction_trace  # the failing one did escalate
