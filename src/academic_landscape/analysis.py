"""Apply an AI-produced analysis only after corpus-bound evidence validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import CorpusSnapshot
from .report import ReportPaths, write_report
from .semantic import ValidationResult, validate_analysis_payload
from .snapshot import load_snapshot


def load_analysis(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("analysis JSON root must be an object")
    return payload


def validate_and_apply_analysis(
    snapshot: CorpusSnapshot,
    payload: dict[str, Any],
    output_dir: str | Path,
) -> tuple[ReportPaths, ValidationResult]:
    validation = validate_analysis_payload(snapshot, payload)
    if not validation.valid:
        codes = ", ".join(sorted({error["code"] for error in validation.errors}))
        raise ValueError(f"analysis validation failed: {codes}")
    return write_report(snapshot, output_dir, analysis=validation.payload), validation


def apply_analysis(
    snapshot_dir: str | Path,
    analysis_path: str | Path,
    output_dir: str | Path,
) -> tuple[ReportPaths, ValidationResult]:
    snapshot = load_snapshot(snapshot_dir)
    payload = load_analysis(analysis_path)
    return validate_and_apply_analysis(snapshot, payload, output_dir)
