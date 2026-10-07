"""Contract tests for lifting paper-local evidence IDs to corpus-global scope."""

from __future__ import annotations

from academic_landscape.evidence_scope import globalize_evidence_ids


def test_globalize_evidence_ids_makes_them_unique_across_papers() -> None:
    papers = [
        {
            "paper_id": "P1",
            "evidence": [{"evidence_id": "E1"}, {"evidence_id": "E2"}],
            "claims": [{"claim_id": "C1", "evidence_ids": ["E1"]}],
            "research_question": {"text": "", "evidence_ids": ["E2"]},
        },
        {
            "paper_id": "P2",
            "evidence": [{"evidence_id": "E1"}],
            "claims": [{"claim_id": "C1", "evidence_ids": ["E1"]}],
        },
    ]

    out = globalize_evidence_ids(papers)

    ids = [e["evidence_id"] for paper in out for e in paper["evidence"]]
    assert len(ids) == len(set(ids))
    assert out[0]["claims"][0]["evidence_ids"] == ["P1::E1"]
    assert out[0]["research_question"]["evidence_ids"] == ["P1::E2"]
    assert out[1]["claims"][0]["evidence_ids"] == ["P2::E1"]


def test_globalize_evidence_ids_is_a_deep_copy() -> None:
    papers = [
        {
            "paper_id": "P1",
            "evidence": [{"evidence_id": "E1"}],
            "claims": [{"claim_id": "C1", "evidence_ids": ["E1"]}],
        }
    ]

    out = globalize_evidence_ids(papers)

    # Original input is untouched.
    assert papers[0]["evidence"][0]["evidence_id"] == "E1"
    assert papers[0]["claims"][0]["evidence_ids"] == ["E1"]
    # Output is a distinct object.
    assert out is not papers
    assert out[0]["evidence"][0]["evidence_id"] == "P1::E1"


def test_globalize_evidence_ids_remaps_nested_extended_fields() -> None:
    papers = [
        {
            "paper_id": "P1",
            "evidence": [{"evidence_id": "E1"}],
            "concepts": [{"name": "x", "evidence_ids": ["E1"]}],
            "methods": [{"name": "m", "evidence_ids": ["E1"]}],
            "results": [{"text": "r", "evidence_ids": ["E1"]}],
        }
    ]

    out = globalize_evidence_ids(papers)

    assert out[0]["concepts"][0]["evidence_ids"] == ["P1::E1"]
    assert out[0]["methods"][0]["evidence_ids"] == ["P1::E1"]
    assert out[0]["results"][0]["evidence_ids"] == ["P1::E1"]