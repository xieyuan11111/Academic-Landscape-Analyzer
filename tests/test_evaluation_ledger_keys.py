"""Regression tests for the evaluation-ledger key drift fix.

The bug: item_key was the filename stem when the previous `attempts` count was
read, then the paper_id from inside the file when the entry was stored.  100 of
209 real MAG result files differ, so every retry was booked as a new item and
`attempts` froze at 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from academic_landscape.evaluation import evaluate_extraction_results


def _result(results: Path, filename: str, paper_id: str | None) -> Path:
    payload = {"evidence": [{"evidence_id": "p::E1", "quote": "q", "page_number": 1}]}
    if paper_id is not None:
        payload["paper_id"] = paper_id
    p = results / filename
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_attempts_increments_when_filename_differs_from_paper_id(tmp_path: Path) -> None:
    """The regression: same file, three runs -> attempts must reach 3.

    With the old code the count was read under the stem key and written under
    the paper_id key, so it stayed at 1 and the ledger grew a second entry.
    """
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out"
    _result(results, "artifact_abc123.json", "paper_inner")

    for expected in (1, 2, 3):
        evaluate_extraction_results(results, output_dir=out)
        ledger = json.loads((out / "evaluation_ledger.json").read_text(encoding="utf-8"))
        assert ledger["items"]["paper_inner"]["attempts"] == expected

    assert len(ledger["items"]) == 1, "one paper must occupy exactly one ledger key"
    assert "artifact_abc123" not in ledger["items"]


def test_ledger_key_is_paper_id_not_filename(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out"
    _result(results, "artifact_zzz.json", "paper_q")
    evaluate_extraction_results(results, output_dir=out)
    ledger = json.loads((out / "evaluation_ledger.json").read_text(encoding="utf-8"))
    assert list(ledger["items"]) == ["paper_q"]
    assert ledger["items"]["paper_q"]["status"] == "succeeded"


def test_migration_from_a_stem_keyed_ledger_keeps_attempt_history(tmp_path: Path) -> None:
    """A ledger written by the buggy version must not lose its history."""
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    _result(results, "artifact_old.json", "paper_old")

    # Simulate the old layout: an entry keyed by the filename stem.
    (out / "evaluation_ledger.json").write_text(
        json.dumps({
            "schema_version": "evaluation-v1",
            "items": {"artifact_old": {"paper_id": "artifact_old", "attempts": 4,
                                       "status": "failed", "path": "x",
                                       "error_code": "invalid_result",
                                       "error_message": "boom"}},
        }),
        encoding="utf-8",
    )
    evaluate_extraction_results(results, output_dir=out)
    ledger = json.loads((out / "evaluation_ledger.json").read_text(encoding="utf-8"))
    assert "paper_old" in ledger["items"]
    assert "artifact_old" not in ledger["items"], "stale stem key must be dropped"
    assert ledger["items"]["paper_old"]["attempts"] == 5


def test_unreadable_file_falls_back_to_stem_and_is_reported_failed(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out"
    (results / "broken.json").write_text("{ not json", encoding="utf-8")

    summary = evaluate_extraction_results(results, output_dir=out)
    ledger = json.loads((out / "evaluation_ledger.json").read_text(encoding="utf-8"))
    assert list(ledger["items"]) == ["broken"]
    entry = ledger["items"]["broken"]
    assert entry["status"] == "failed"
    assert entry["error_code"] == "invalid_result"
    assert "not readable JSON" in entry["error_message"]
    assert summary["failed"] == 1
    assert summary["succeeded"] == 0


def test_missing_paper_id_is_reported_not_crashed(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out"
    _result(results, "noid.json", None)
    summary = evaluate_extraction_results(results, output_dir=out)
    ledger = json.loads((out / "evaluation_ledger.json").read_text(encoding="utf-8"))
    assert ledger["items"]["noid"]["status"] == "failed"
    assert ledger["items"]["noid"]["error_message"] == "missing_paper_id"
    assert summary["failed"] == 1


def test_two_papers_sharing_one_filename_stem_are_distinct(tmp_path: Path) -> None:
    """The inverse case the old code also broke: a retry whose file is renamed
    between runs must land on the same ledger key via paper_id."""
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out"
    _result(results, "artifact_one.json", "paper_stable")

    evaluate_extraction_results(results, output_dir=out)
    # rename the file, same paper_id inside
    (results / "artifact_one.json").rename(results / "artifact_renamed.json")
    evaluate_extraction_results(results, output_dir=out)

    ledger = json.loads((out / "evaluation_ledger.json").read_text(encoding="utf-8"))
    assert list(ledger["items"]) == ["paper_stable"]
    assert ledger["items"]["paper_stable"]["attempts"] == 2
    assert ledger["items"]["paper_stable"]["path"].endswith("artifact_renamed.json")


def test_limit_and_selection_still_count_the_selected_files(tmp_path: Path) -> None:
    """The selected_items rewrite must not change what the summary counts."""
    results = tmp_path / "results"
    results.mkdir()
    out = tmp_path / "out"
    _result(results, "a1.json", "paper_a")
    _result(results, "b1.json", "paper_b")
    _result(results, "c1.json", "paper_c")

    summary = evaluate_extraction_results(results, output_dir=out, limit=2)
    assert summary["total"] == 2
    assert summary["succeeded"] == 2

    summary2 = evaluate_extraction_results(
        results, output_dir=out, selected_paper_ids=["paper_c"]
    )
    assert summary2["total"] == 1
    assert summary2["succeeded"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
