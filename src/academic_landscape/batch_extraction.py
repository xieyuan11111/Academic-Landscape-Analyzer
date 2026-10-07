"""Resumable batch orchestration for validated single-paper extraction."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Protocol

from .models import CorpusSnapshot
from .run_ledger import RunLedger
from .sqlite_store import EvidenceSpan


def persist_evidence(
    store: Any,
    snapshot: CorpusSnapshot,
    document: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Write one paper's evidence rows into the authority store.

    The store keys spans by ``(snapshot_id, evidence_id)`` while single-paper
    extraction numbers evidence ``E1..En`` per paper — the collision
    ``evidence_scope`` exists to solve — so IDs are globalized to ``paper_id::E1``
    here, matching the convention used by source cards and relations.

    ``save_evidence`` is the authority gate (it re-checks that the quote is
    verbatim in the claimed page's text). A rejection is returned rather than
    raised: it is a per-row validation outcome worth counting and reporting, not
    a reason to abandon the paper's other rows.

    Returns ``{"saved": int, "rejected": [ {evidence_id, reason}, ... ]}``.
    """
    saved = 0
    rejected: list[dict[str, Any]] = []

    evidence_type_by_id: dict[str, str] = {}
    for claim in payload.get("claims", []) or []:
        if not isinstance(claim, dict):
            continue
        for local_id in claim.get("evidence_ids", []) or []:
            evidence_type_by_id.setdefault(local_id, claim.get("evidence_type", ""))

    for row in payload.get("evidence", []) or []:
        if not isinstance(row, dict):
            continue
        local_id = row.get("evidence_id", "")
        global_id = f"{payload.get('paper_id', document.paper_id)}::{local_id}"
        try:
            store.save_evidence(EvidenceSpan(
                evidence_id=global_id,
                snapshot_id=snapshot.snapshot_id,
                paper_id=payload.get("paper_id", document.paper_id),
                artifact_id=row.get("artifact_id", document.artifact_id),
                page_number=int(row.get("page_number", 0)),
                quote=row.get("quote", ""),
                locator=row.get("locator", ""),
                evidence_type=evidence_type_by_id.get(local_id, ""),
            ))
        except Exception as error:  # noqa: BLE001 - report, do not abandon
            rejected.append({"evidence_id": global_id, "reason": str(error)})
            continue
        saved += 1

    return {"saved": saved, "rejected": rejected}


class PaperExtractor(Protocol):
    def extract(self, snapshot: CorpusSnapshot, document: Any) -> Any: ...


class BatchExtractionRunner:
    def __init__(
        self,
        ledger: RunLedger,
        extractor: PaperExtractor,
        output_dir: str | Path,
        *,
        evidence_store: Any | None = None,
    ) -> None:
        self.ledger = ledger
        self.extractor = extractor
        self.evidence_store = evidence_store
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _serialize(result: Any) -> dict[str, Any]:
        if hasattr(result, "model_dump"):
            payload = result.model_dump(mode="json")
        elif isinstance(result, dict):
            payload = result
        else:
            raise TypeError("paper extractor returned neither a Pydantic model nor a dict")
        if not isinstance(payload, dict):
            raise TypeError("paper extractor result must serialize to an object")
        return payload

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _write_result(self, artifact_id: str, payload: dict[str, Any]) -> tuple[Path, str]:
        target = self.output_dir / f"{artifact_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target, self._sha256(target)

    def run(self, snapshot: CorpusSnapshot, run_id: str, *, max_items: int | None = None) -> dict[str, int]:
        self.ledger.recover_interrupted(run_id)
        documents = {document.artifact_id: document for document in snapshot.included_documents}
        pending = self.ledger.pending_items(run_id, include_failed=True)
        evidence_saved = 0
        evidence_rejections: list[dict[str, Any]] = []
        if max_items is not None:
            if max_items <= 0:
                raise ValueError("max_items must be positive")
            pending = pending[:max_items]

        for item in pending:
            artifact_id = item["artifact_id"]
            document = documents.get(artifact_id)
            if document is None:
                self.ledger.claim_item(run_id, artifact_id)
                self.ledger.mark_failed(
                    run_id,
                    artifact_id,
                    "document_not_in_snapshot",
                    f"artifact_id is not present in parsed snapshot: {artifact_id}",
                )
                continue
            self.ledger.claim_item(run_id, artifact_id)
            started = time.perf_counter()
            try:
                result = self.extractor.extract(snapshot, document)
                payload = self._serialize(result)
                path, result_sha256 = self._write_result(artifact_id, payload)
                if self.evidence_store is not None:
                    outcome = persist_evidence(self.evidence_store, snapshot, document, payload)
                    evidence_saved += outcome["saved"]
                    for entry in outcome["rejected"]:
                        evidence_rejections.append({
                            "artifact_id": artifact_id,
                            "paper_id": payload.get("paper_id", document.paper_id),
                            **entry,
                        })
                self.ledger.mark_succeeded(
                    run_id,
                    artifact_id,
                    result_path=str(path),
                    result_sha256=result_sha256,
                    input_hash=document.sha256,
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                )
            except Exception as error:  # noqa: BLE001 - isolate one paper failure
                self.ledger.mark_failed(
                    run_id,
                    artifact_id,
                    type(error).__name__,
                    str(error),
                    duration_ms=round((time.perf_counter() - started) * 1000, 3),
                )
        counts = dict(self.ledger.status_counts(run_id))
        if self.evidence_store is not None:
            counts["evidence_saved"] = evidence_saved
            counts["evidence_rejected"] = len(evidence_rejections)
            # Rejections are validation outcomes the caller must be able to see:
            # write them out rather than leaving them as a bare count.
            if evidence_rejections:
                (self.output_dir / "evidence-rejections.json").write_text(
                    json.dumps(evidence_rejections, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
        return counts
