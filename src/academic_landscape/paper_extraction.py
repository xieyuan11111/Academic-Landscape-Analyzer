"""Pydantic contracts and cached single-paper evidence extraction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import CorpusSnapshot, DocumentRecord
from .prompts import (
    build_anchor_hint_prompt,
    build_research_card_prompt,
)
from .quote_resolver import resolve_quote
from .semantic import validate_analysis_payload


class PaperExtractionEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    artifact_id: str
    page_number: int
    quote: str
    locator: str = ""


class PaperExtractionClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    evidence_type: str
    evidence_ids: list[str] = Field(default_factory=list)


class PaperExtractionQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class PaperExtractionConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    definition: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class PaperExtractionFramework(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class PaperExtractionMethod(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class PaperExtractionDataSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    evidence_ids: list[str] = Field(default_factory=list)


class PaperExtractionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    evidence_ids: list[str] = Field(default_factory=list)


class PaperExtractionResultItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class PaperExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    research_question: PaperExtractionQuestion = Field(default_factory=PaperExtractionQuestion)
    concepts: list[PaperExtractionConcept] = Field(default_factory=list)
    theoretical_frameworks: list[PaperExtractionFramework] = Field(default_factory=list)
    methods: list[PaperExtractionMethod] = Field(default_factory=list)
    data_samples: list[PaperExtractionDataSample] = Field(default_factory=list)
    contexts: list[PaperExtractionContext] = Field(default_factory=list)
    results: list[PaperExtractionResultItem] = Field(default_factory=list)
    limitations: list[PaperExtractionResultItem] = Field(default_factory=list)
    claims: list[PaperExtractionClaim] = Field(default_factory=list)
    evidence: list[PaperExtractionEvidence] = Field(default_factory=list)


class ExtractionClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class _CacheEntry:
    input_hash: str
    result: dict[str, Any]


class PaperExtractionService:
    def __init__(self, client: ExtractionClient, cache_path: str | Path | None = None) -> None:
        self.client = client
        self.cache_path = (
            Path(cache_path).expanduser().resolve() if cache_path is not None else None
        )
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_cache(self) -> dict[str, _CacheEntry]:
        if self.cache_path is None or not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"extraction cache is unreadable: {self.cache_path}") from error
        if not isinstance(payload, dict):
            raise RuntimeError("extraction cache root must be an object")
        result: dict[str, _CacheEntry] = {}
        for key, value in payload.items():
            if isinstance(value, dict) and isinstance(value.get("input_hash"), str) and isinstance(value.get("result"), dict):
                result[str(key)] = _CacheEntry(input_hash=value["input_hash"], result=value["result"])
        return result

    def _save_cache(self, cache: dict[str, _CacheEntry]) -> None:
        if self.cache_path is None:
            return
        payload = {
            key: {"input_hash": entry.input_hash, "result": entry.result}
            for key, entry in sorted(cache.items())
        }
        temporary = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.cache_path)

    @staticmethod
    def _input_hash(snapshot: CorpusSnapshot, document: DocumentRecord, system: str, user: str) -> str:
        payload = {
            "snapshot_id": snapshot.snapshot_id,
            "paper_id": document.paper_id,
            "artifact_id": document.artifact_id,
            "prompt": {"system": system, "user": user},
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _validate_payload(self, snapshot: CorpusSnapshot, paper: dict[str, Any]) -> dict[str, Any]:
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
            raise ValueError(f"paper extraction validation failed: {codes}")
        return validation.payload["papers"][0]

    def extract(self, snapshot: CorpusSnapshot, document: DocumentRecord) -> PaperExtractionResult:
        if document.status != "parsed":
            raise ValueError(f"document is not eligible for extraction: {document.status}")
        anchor_system, anchor_user = build_anchor_hint_prompt(snapshot, document)
        cache_hash = self._input_hash(
            snapshot,
            document,
            anchor_system,
            anchor_user + "\nTWO_STAGE_HINT_ANCHOR_V1",
        )
        cache_key = f"{snapshot.snapshot_id}:{document.paper_id}:{document.artifact_id}"
        cache = self._load_cache()
        cached = cache.get(cache_key)
        if cached is not None and cached.input_hash == cache_hash:
            try:
                return PaperExtractionResult.model_validate(cached.result)
            except ValidationError:
                del cache[cache_key]

        anchor_raw = self.client.complete_json(anchor_system, anchor_user)
        if not isinstance(anchor_raw, dict):
            raise ValueError("evidence anchor response must be a JSON object")
        anchors = anchor_raw.get("anchors")
        if not isinstance(anchors, list):
            raise ValueError("evidence anchor response must contain an anchors list")
        resolved_evidence: list[dict[str, Any]] = []
        for anchor in anchors:
            if not isinstance(anchor, dict):
                continue
            evidence_id = anchor.get("evidence_id")
            hint = anchor.get("hint")
            candidate_page = anchor.get("page_number")
            if not isinstance(evidence_id, str) or not evidence_id:
                continue
            if not isinstance(hint, str) or not hint.strip():
                continue
            if candidate_page is not None and (
                not isinstance(candidate_page, int) or isinstance(candidate_page, bool)
            ):
                candidate_page = None
            resolved = resolve_quote(document.pages, hint, candidate_page=candidate_page)
            if resolved is None:
                continue
            resolved_evidence.append(
                {
                    "evidence_id": evidence_id,
                    "artifact_id": document.artifact_id,
                    "page_number": resolved.page_number,
                    "quote": resolved.quote,
                    "locator": f"page {resolved.page_number}",
                }
            )
        anchor_paper = self._validate_payload(
            snapshot,
            {
                "paper_id": document.paper_id,
                "research_question": {"text": "", "evidence_ids": []},
                "claims": [],
                "evidence": resolved_evidence,
            },
        )
        anchor_evidence = anchor_paper["evidence"]
        card_system, card_user = build_research_card_prompt(snapshot, document, anchor_evidence)
        card_raw = self.client.complete_json(card_system, card_user)
        if not isinstance(card_raw, dict):
            raise ValueError("research card response must be a JSON object")
        submitted_evidence = card_raw.pop("evidence", None)
        if submitted_evidence not in (None, [], anchor_evidence):
            raise ValueError("research_card_new_evidence")
        merged_paper = {
            **card_raw,
            "paper_id": card_raw.get("paper_id", document.paper_id),
            "evidence": anchor_evidence,
        }
        validated_paper = self._validate_payload(snapshot, merged_paper)
        try:
            result = PaperExtractionResult.model_validate(validated_paper)
        except ValidationError as error:
            raise ValueError(f"paper extraction schema validation failed: {error}") from error
        cache[cache_key] = _CacheEntry(input_hash=cache_hash, result=result.model_dump(mode="json"))
        self._save_cache(cache)
        return result
