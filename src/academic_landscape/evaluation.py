"""Resumable quality evaluation for validated single-paper extraction results.

This module deliberately evaluates result artifacts only; it does not call an LLM or
mutate the source corpus. It produces a per-item JSON ledger plus aggregate metrics
for MAG-50/100 acceptance runs.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_paths(
    results_dir: Path,
    selected_paper_ids: Iterable[str] | None,
    limit: int | None,
) -> list[Path]:
    paths = sorted(results_dir.glob("*.json"))
    selected = set(selected_paper_ids or ())
    if selected:
        filtered = []
        for path in paths:
            try:
                paper_id = json.loads(path.read_text(encoding="utf-8")).get("paper_id")
            except (OSError, json.JSONDecodeError):
                paper_id = None
            if paper_id in selected:
                filtered.append(path)
        paths = filtered
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        paths = paths[:limit]
    return paths


def _render_markdown(summary: dict[str, Any], ledger: dict[str, Any]) -> str:
    lines = [
        "# Extraction Evaluation Metrics",
        "",
        f"- Total: {summary['total']}",
        f"- Succeeded: {summary['succeeded']}",
        f"- Failed: {summary['failed']}",
        f"- Skipped: {summary['skipped']}",
        f"- Success rate: {summary['success_rate']}",
        f"- Evidence count: {summary['evidence_count']}",
        f"- Empty quote count: {summary['empty_quote_count']}",
        f"- Overlong quote count (>600 chars): {summary['overlong_quote_count']}",
        f"- Research-question coverage: {summary['research_question_coverage']}",
        f"- Duration p50 (ms): {summary['duration_ms_p50']}",
        f"- Duration p90 (ms): {summary['duration_ms_p90']}",
        "",
        "## Items",
        "",
        "| Paper | Status | Attempts | Duration (ms) | Error |",
        "|---|---|---:|---:|---|",
    ]
    for item in ledger["items"].values():
        lines.append(
            f"| `{item['paper_id']}` | `{item['status']}` | "
            f"{item['attempts']} | {item.get('duration_ms', '')} | "
            f"{item.get('error_code', '')} |"
        )
    return "\n".join(lines) + "\n"


def evaluate_extraction_results(
    results_dir: str | Path,
    *,
    output_dir: str | Path,
    selected_paper_ids: Iterable[str] | None = None,
    limit: int | None = None,
    run_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    results_root = Path(results_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "evaluation_ledger.json"
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            ledger = {"schema_version": "evaluation-v1", "items": {}}
    else:
        ledger = {"schema_version": "evaluation-v1", "items": {}}

    real_durations: dict[str, float] = {}
    if run_summary_path is not None:
        run_summary = json.loads(
            Path(run_summary_path).expanduser().resolve().read_text(encoding="utf-8")
        )
        real_durations = {
            str(item["paper_id"]): float(item["duration_ms"])
            for item in run_summary.get("items", [])
            if item.get("status") == "succeeded" and item.get("duration_ms") is not None
        }

    paths = _selected_paths(results_root, selected_paper_ids, limit)
    for path in paths:
        started = time.perf_counter()
        # The ledger key must be stable across runs, so it may NOT change from
        # the filename stem to the paper_id found inside the file.  100 of 209
        # real MAG result files are named artifact_*.json while carrying a
        # different paper_id inside, so keying the *previous* attempts count on
        # the stem and then storing under the paper_id booked every retry as a
        # brand-new item and froze `attempts` at 1.  The paper_id is therefore
        # read first, and a per-run fallback key is used only for files whose
        # paper_id cannot be read at all.
        file_stem = path.stem
        prior: dict[str, Any] = {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            paper_id = payload.get("paper_id")
        except (OSError, json.JSONDecodeError):
            payload = None
            paper_id = None
        if isinstance(paper_id, str) and paper_id:
            item_key = paper_id
            # A previous run may have booked this paper under the filename stem
            # (the buggy layout).  Carry that attempt count over instead of
            # silently resetting the history.
            prior = ledger["items"].get(item_key) or ledger["items"].get(file_stem) or {}
        else:
            item_key = file_stem
            prior = ledger["items"].get(file_stem) or {}
        entry: dict[str, Any] = {
            "paper_id": item_key,
            "path": str(path),
            "status": "failed",
            "attempts": int(prior.get("attempts", 0)) + 1,
            "input_sha256": _sha256(path),
            "result_sha256": None,
            "error_code": None,
            "error_message": None,
        }
        if payload is not None and isinstance(paper_id, str) and paper_id:
            entry["result_sha256"] = _sha256(path)
            entry["status"] = "succeeded"
        else:
            entry["error_code"] = "invalid_result"
            if payload is None:
                entry["error_message"] = (
                    f"result file is not readable JSON: {path.name}"
                )
            else:
                entry["error_message"] = "missing_paper_id"
        entry["evaluation_duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        entry["duration_ms"] = real_durations.get(
            str(entry["paper_id"]), entry["evaluation_duration_ms"]
        )
        entry["started_at"] = time.time() - entry["duration_ms"] / 1000
        entry["finished_at"] = time.time()
        # Store under the paper_id key, and drop any stale stem-keyed entry so
        # one paper can never appear twice in the ledger.
        if item_key != file_stem and file_stem in ledger["items"]:
            del ledger["items"][file_stem]
        ledger["items"][item_key] = entry

    selected_paths = {str(path) for path in paths}
    selected_items = [
        item for item in ledger["items"].values()
        if item.get("path") in selected_paths
    ]
    succeeded = [item for item in selected_items if item["status"] == "succeeded"]
    failed = [item for item in selected_items if item["status"] == "failed"]
    evidence_count = empty_quotes = overlong_quotes = rq_count = 0
    durations = []
    for item in succeeded:
        durations.append(float(item["duration_ms"]))
        path = Path(item["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        rq = payload.get("research_question", {})
        if isinstance(rq, dict) and str(rq.get("text", "")).strip():
            rq_count += 1
        for evidence in payload.get("evidence", []):
            evidence_count += 1
            quote = str(evidence.get("quote", ""))
            if not quote.strip():
                empty_quotes += 1
            if len(quote) > 600:
                overlong_quotes += 1

    total = len(selected_items)
    summary = {
        "schema_version": "evaluation-v1",
        "duration_source": "run_ledger" if real_durations else "evaluation_read",
        "total": total,
        "succeeded": len(succeeded),
        "failed": len(failed),
        "skipped": 0,
        "success_rate": round(len(succeeded) / total, 4) if total else 0.0,
        "evidence_count": evidence_count,
        "empty_quote_count": empty_quotes,
        "overlong_quote_count": overlong_quotes,
        "research_question_coverage": round(rq_count / len(succeeded), 4) if succeeded else 0.0,
        "duration_ms_p50": percentile(durations, 0.5),
        "duration_ms_p90": percentile(durations, 0.9),
    }
    ledger["summary"] = summary
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "metrics.md").write_text(_render_markdown(summary, ledger), encoding="utf-8")
    return summary
