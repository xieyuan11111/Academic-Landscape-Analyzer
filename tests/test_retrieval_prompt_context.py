"""Tests that bounded retrieval context is visible but non-authoritative."""

from __future__ import annotations

from academic_landscape.cross_paper import _build_comparison_prompt
from academic_landscape.stage_reconstruction import _build_stage_prompt


def test_comparison_prompt_labels_retrieval_as_context_only() -> None:
    prompt = _build_comparison_prompt(
        "Direction A", "paper_a", "DETAILS", "retrieved chunk text [paper_x p2]"
    )
    assert "retrieved chunk text" in prompt
    assert "discovery context" in prompt.lower()
    assert "not authoritative evidence" in prompt.lower()


def test_stage_prompt_labels_retrieval_as_context_only() -> None:
    prompt = _build_stage_prompt(
        "Direction A", "paper_a, paper_b", "THREADS",
        "retrieved chunk text [paper_x p2]",
    )
    assert "retrieved chunk text" in prompt
    assert "discovery context" in prompt.lower()
    assert "not authoritative evidence" in prompt.lower()
