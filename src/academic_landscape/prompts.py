"""Prompts for the two-stage semantic analysis boundary."""

from __future__ import annotations

import json
from typing import Any

from .models import CorpusSnapshot, DocumentRecord
from .relation_types import RELATION_PROMPT_VOCABULARY


def build_evidence_anchor_prompt(snapshot: CorpusSnapshot, document: DocumentRecord) -> tuple[str, str]:
    system = """You are the evidence anchoring stage for a single academic paper. Return JSON only.
Select at most 12 evidence anchors useful for a later research card. Each quote must be copied as one exact contiguous span from one supplied page, under 160 characters. Never merge text across headings, formulas, figures, citations, columns, pages, or paragraphs. Do not summarize, infer, or produce a research card. If a point cannot be supported by an exact short quote, omit it."""
    pages = "\n\n".join(
        f"[PAGE {page.page_number}]\n{page.text}" for page in document.pages
    )
    user = f"""CORPUS SNAPSHOT: {snapshot.snapshot_id}
PAPER ID: {document.paper_id}
ARTIFACT ID: {document.artifact_id}

SOURCE PAGES:
{pages}

Return exactly this JSON object:
{{
  "paper_id": "{document.paper_id}",
  "claims": [],
  "evidence": [
    {{"evidence_id": "E1", "artifact_id": "{document.artifact_id}", "page_number": 1, "quote": "exact short copied span", "locator": "page 1"}}
  ]
}}
"""
    return system, user


def build_anchor_hint_prompt(snapshot: CorpusSnapshot, document: DocumentRecord) -> tuple[str, str]:
    system = """You are the evidence anchoring stage for a single academic paper. Return JSON only.
Do NOT copy long quotes. For each important point of the paper, provide a short *hint* phrase that you believe appears word-for-word on one page. A hint is a short contiguous phrase of about 3 to 8 words — a technical term, a named concept, or a short clause. Copy the wording as it appears in the paper; do not paraphrase, do not summarize, do not compose a long sentence, and do not invent vocabulary. Give your best-guess page number per hint, but an uncertain or wrong page number is acceptable because the system searches the whole paper anyway. Select at most 12 anchors that would be most useful for a later research card."""
    pages = "\n\n".join(
        f"[PAGE {page.page_number}]\n{page.text}" for page in document.pages
    )
    user = f"""CORPUS SNAPSHOT: {snapshot.snapshot_id}
PAPER ID: {document.paper_id}
ARTIFACT ID: {document.artifact_id}

SOURCE PAGES:
{pages}

Return exactly this JSON object:
{{
  "paper_id": "{document.paper_id}",
  "anchors": [
    {{"evidence_id": "E1", "page_number": 1, "hint": "short verbatim phrase from that page"}}
  ]
}}
"""
    return system, user


def build_research_card_prompt(
    snapshot: CorpusSnapshot,
    document: DocumentRecord,
    evidence: list[dict[str, Any]],
) -> tuple[str, str]:
    system = """You are the research card stage for a single academic paper. Return JSON only.
Use ONLY the provided, already validated Evidence IDs. Do not return an evidence array, do not create new evidence IDs, and do not quote the PDF. If the validated evidence does not support a field, return an empty list or empty text. This is a structured extraction, not a literature review or a field-level conclusion."""
    user = f"""CORPUS SNAPSHOT: {snapshot.snapshot_id}
PAPER ID: {document.paper_id}
ARTIFACT ID: {document.artifact_id}
METADATA: {json.dumps(document.metadata, ensure_ascii=False, sort_keys=True)}

VALIDATED EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Return exactly this JSON object without an "evidence" key:
{{
  "paper_id": "{document.paper_id}",
  "research_question": {{"text": "...", "evidence_ids": ["E1"]}},
  "concepts": [{{"name": "...", "definition": "...", "evidence_ids": ["E1"]}}],
  "theoretical_frameworks": [{{"name": "...", "description": "...", "evidence_ids": ["E1"]}}],
  "methods": [{{"name": "...", "description": "...", "evidence_ids": ["E1"]}}],
  "data_samples": [{{"description": "...", "evidence_ids": ["E1"]}}],
  "contexts": [{{"description": "...", "evidence_ids": ["E1"]}}],
  "results": [{{"text": "...", "evidence_ids": ["E1"]}}],
  "limitations": [{{"text": "...", "evidence_ids": ["E1"]}}],
  "claims": [{{"claim_id": "C1", "text": "...", "evidence_type": "A|B", "evidence_ids": ["E1"]}}]
}}
"""
    return system, user


def build_paper_prompt(snapshot: CorpusSnapshot, document: DocumentRecord) -> tuple[str, str]:
    system = """You are the single paper evidence-extraction stage for an academic corpus.
Return JSON only. Do not invent metadata, claims, page numbers, or quotations.
Every evidence quote must be copied from the supplied page text. Each quote must be one short contiguous span from one page (preferably one sentence or clause, under 240 characters); never join text separated by a formula, heading, figure, citation, or paragraph. Do not add a heading, label, or definition lead-in unless the exact words appear in that page text. Preserve the quoted wording; page line-wraps and hyphenation may be normalized by the validator, but invented or merged text will be rejected.
Use evidence type A for an author-explicit claim and B for a result reported by the paper.
Use C only for a cross-paper synthesis (normally not needed at this stage) and D for an
unverified interpretation. The paper_id and artifact_id must be copied exactly.
This is a single paper, not a request to decide the field's consensus or frontier."""
    pages = "\n\n".join(
        f"[PAGE {page.page_number}]\n{page.text}" for page in document.pages
    )
    user = f"""CORPUS SNAPSHOT: {snapshot.snapshot_id}
PAPER ID: {document.paper_id}
ARTIFACT ID: {document.artifact_id}
FILE: {document.relative_path}
METADATA: {json.dumps(document.metadata, ensure_ascii=False, sort_keys=True)}

SOURCE PAGES:
{pages}

Return one JSON object with this shape:
{{
  "paper_id": "{document.paper_id}",
  "research_question": {{"text": "...", "evidence_ids": ["E1"]}},
  "concepts": [{{"name": "...", "definition": "...", "evidence_ids": ["E1"]}}],
  "theoretical_frameworks": [{{"name": "...", "description": "...", "evidence_ids": ["E1"]}}],
  "methods": [{{"name": "...", "description": "...", "evidence_ids": ["E1"]}}],
  "data_samples": [{{"description": "...", "evidence_ids": ["E1"]}}],
  "contexts": [{{"description": "...", "evidence_ids": ["E1"]}}],
  "results": [{{"text": "...", "evidence_ids": ["E1"]}}],
  "limitations": [{{"text": "...", "evidence_ids": ["E1"]}}],
  "claims": [
    {{"claim_id": "C1", "text": "...", "evidence_type": "A|B|C|D", "evidence_ids": ["E1"]}}
  ],
  "evidence": [
    {{"evidence_id": "E1", "artifact_id": "{document.artifact_id}", "page_number": 1, "quote": "exact source excerpt", "locator": "page 1"}}
  ]
}}
If the paper does not support a field, use an empty string or empty list. Never fill gaps from outside knowledge."""
    return system, user


def build_corpus_prompt(snapshot: CorpusSnapshot, papers: list[dict[str, Any]]) -> tuple[str, str]:
    system = """You are the cross-paper synthesis stage for a closed academic corpus.
Return JSON only. Use only the supplied paper records and evidence IDs. Do not create
new paper IDs or evidence IDs. Do not call a difference a contradiction when it may be
conditional, methodological, parallel, not comparable, or unknown. A direction may
contain overlapping papers. A frontier candidate is only a candidate within this corpus,
not a claim about the whole discipline. Do not return the papers array; the caller keeps
the evidence-validated paper records unchanged."""
    user = f"""CORPUS SNAPSHOT: {snapshot.snapshot_id}
VALIDATED PAPER RECORDS:
{json.dumps(papers, ensure_ascii=False, indent=2, sort_keys=True)}

Return one JSON object with this shape:
{{
  "directions": [
    {{"direction_id": "D1", "label": "...", "paper_ids": ["paper_id"], "evidence_ids": ["E1"], "status": "candidate|mainstream|emerging|unclear"}}
  ],
  "relations": [
    {{"relation_id": "R1", "source_paper_id": "...", "target_paper_id": "...", "relation_type": "{RELATION_PROMPT_VOCABULARY}", "evidence_ids": ["E1"]}}
  ],
  "controversies": [
    {{"id": "K1", "question": "...", "paper_ids": ["..."], "evidence_ids": ["E1"], "status": "candidate|not_comparable|insufficient_evidence"}}
  ],
  "frontier_candidates": [
    {{"id": "F1", "label": "...", "paper_ids": ["..."], "evidence_ids": ["E1"], "maturity": "early_proposal|candidate|emerging|formed_frontier"}}
  ]
}}
Empty lists are preferable to unsupported conclusions."""
    return system, user
