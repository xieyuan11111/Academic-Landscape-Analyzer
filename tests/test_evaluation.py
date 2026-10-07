"""Tests for resumable evaluation metrics over validated extraction results."""

from __future__ import annotations

import json
from pathlib import Path

from academic_landscape.evaluation import evaluate_extraction_results, percentile


def _write_result(root: Path, name: str, *, evidence: list, rq: str = "RQ") -> None:
    payload = {
        "paper_id": name,
        "research_question": {"text": rq, "evidence_ids": []},
        "evidence": evidence,
    }
    (root / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _evidence(eid: str, quote: str) -> dict:
    return {
        "evidence_id": eid,
        "artifact_id": "artifact",
        "page_number": 1,
        "quote": quote,
        "locator": "page 1",
    }


def test_percentile_is_deterministic() -> None:
    assert percentile([10, 20, 30, 40], 0.5) == 25.0
    assert percentile([], 0.5) is None


def test_evaluation_metrics_count_quality_and_status(tmp_path: Path) -> None:
    _write_result(
        tmp_path,
        "paper_a",
        evidence=[_evidence("E1", "short quote")],
    )
    _write_result(
        tmp_path,
        "paper_b",
        evidence=[_evidence("E1", "x" * 700)],
        rq="",
    )
    output = tmp_path / "out"
    summary = evaluate_extraction_results(tmp_path, output_dir=output)
    assert summary["total"] == 2
    assert summary["succeeded"] == 2
    assert summary["failed"] == 0
    assert summary["evidence_count"] == 2
    assert summary["overlong_quote_count"] == 1
    assert summary["empty_quote_count"] == 0
    assert summary["research_question_coverage"] == 0.5
    assert (output / "summary.json").is_file()
    assert (output / "metrics.md").is_file()


def test_selection_limit_and_resume_ledger(tmp_path: Path) -> None:
    for name in ("paper_a", "paper_b", "paper_c"):
        _write_result(tmp_path, name, evidence=[])
    output = tmp_path / "out"
    first = evaluate_extraction_results(
        tmp_path, output_dir=output, limit=2, selected_paper_ids=["paper_a", "paper_b"]
    )
    assert first["total"] == 2
    second = evaluate_extraction_results(
        tmp_path, output_dir=output, limit=2, selected_paper_ids=["paper_a", "paper_b"]
    )
    assert second["total"] == 2
    ledger = json.loads((output / "evaluation_ledger.json").read_text(encoding="utf-8"))
    assert len(ledger["items"]) == 2
    assert all(item["status"] == "succeeded" for item in ledger["items"].values())


def test_evaluation_uses_true_duration_from_run_summary(tmp_path: Path) -> None:
    _write_result(tmp_path, "paper_a", evidence=[])
    run_summary = {
        "items": [
            {"paper_id": "paper_a", "duration_ms": 4321.0, "status": "succeeded"}
        ]
    }
    summary_path = tmp_path / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary), encoding="utf-8")
    summary = evaluate_extraction_results(
        tmp_path,
        output_dir=tmp_path / "out",
        run_summary_path=summary_path,
    )
    assert summary["duration_source"] == "run_ledger"
    assert summary["duration_ms_p50"] == 4321.0
