"""Deterministic, source-bounded corpus overview statistics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .models import CorpusSnapshot
from .semantic import validate_analysis_payload


_METADATA_FIELDS = ("title", "author", "creationDate", "keywords", "subject")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class CorpusOverview:
    snapshot_id: str
    document_count: int
    parsed_document_count: int
    unique_paper_count: int
    status_counts: dict[str, int]
    page_count: int
    text_char_count: int
    quality_flag_counts: dict[str, int]
    language_counts: dict[str, int]
    metadata_coverage: dict[str, int]
    evidence_count: int
    evidence_paper_count: int
    research_direction_count: int
    semantic_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "document_count": self.document_count,
            "parsed_document_count": self.parsed_document_count,
            "unique_paper_count": self.unique_paper_count,
            "status_counts": dict(self.status_counts),
            "page_count": self.page_count,
            "text_char_count": self.text_char_count,
            "quality_flag_counts": dict(self.quality_flag_counts),
            "language_counts": dict(self.language_counts),
            "metadata_coverage": dict(self.metadata_coverage),
            "evidence_count": self.evidence_count,
            "evidence_paper_count": self.evidence_paper_count,
            "research_direction_count": self.research_direction_count,
            "semantic_status": self.semantic_status,
        }


def _language_for_text(text: str) -> str:
    has_cjk = bool(_CJK_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "unknown"


def _increment(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


def build_corpus_overview(
    snapshot: CorpusSnapshot,
    *,
    validated_analysis: dict[str, Any] | None = None,
) -> CorpusOverview:
    """Build facts about one immutable snapshot without generating semantics."""
    evidence_count = 0
    evidence_paper_count = 0
    research_direction_count = 0
    semantic_status = "not_provided"
    if validated_analysis is not None:
        validation = validate_analysis_payload(snapshot, validated_analysis)
        if not validation.valid:
            codes = ", ".join(sorted({error["code"] for error in validation.errors}))
            raise ValueError(f"validated_analysis is invalid: {codes}")
        papers = validation.payload.get("papers", [])
        evidence_count = sum(len(paper.get("evidence", [])) for paper in papers)
        evidence_paper_count = sum(1 for paper in papers if paper.get("evidence"))
        research_direction_count = len(validation.payload.get("directions", []))
        semantic_status = "validated_input_only"

    status_counts: dict[str, int] = {}
    quality_flag_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    metadata_coverage = {field: 0 for field in _METADATA_FIELDS}
    page_count = 0
    text_char_count = 0

    for document in snapshot.documents:
        _increment(status_counts, document.status)
        for flag in document.quality_flags:
            _increment(quality_flag_counts, flag)
        for field in _METADATA_FIELDS:
            if str(document.metadata.get(field, "")).strip():
                metadata_coverage[field] += 1
        if document.status != "parsed":
            continue
        page_count += document.page_count
        text_char_count += document.text_char_count
        full_text = "\n".join(page.text for page in document.pages)
        _increment(language_counts, _language_for_text(full_text))

    return CorpusOverview(
        snapshot_id=snapshot.snapshot_id,
        document_count=len(snapshot.documents),
        parsed_document_count=len(snapshot.included_documents),
        unique_paper_count=len(snapshot.unique_paper_ids),
        status_counts=dict(sorted(status_counts.items())),
        page_count=page_count,
        text_char_count=text_char_count,
        quality_flag_counts=dict(sorted(quality_flag_counts.items())),
        language_counts=dict(sorted(language_counts.items())),
        metadata_coverage=metadata_coverage,
        evidence_count=evidence_count,
        evidence_paper_count=evidence_paper_count,
        research_direction_count=research_direction_count,
        semantic_status=semantic_status,
    )
