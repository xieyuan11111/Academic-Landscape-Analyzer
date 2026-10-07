from pathlib import Path

import fitz

from academic_landscape.corpus_overview import build_corpus_overview
from academic_landscape.ingest import build_snapshot


def make_pdf(path: Path, text: str, *, author: str = "", title: str = "") -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.set_metadata({"title": title, "author": author})
    document.save(path)
    document.close()


def test_overview_reports_snapshot_scoped_counts_and_quality(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "English alpha text.", author="Author A", title="Paper A")
    duplicate = tmp_path / "paper-copy.pdf"
    duplicate.write_bytes((tmp_path / "paper.pdf").read_bytes())
    empty = fitz.open()
    empty.new_page()
    empty.save(tmp_path / "scan.pdf")
    empty.close()

    snapshot = build_snapshot(tmp_path)
    overview = build_corpus_overview(snapshot)

    assert overview.snapshot_id == snapshot.snapshot_id
    assert overview.document_count == 3
    assert overview.parsed_document_count == 1
    assert overview.unique_paper_count == 2
    assert overview.status_counts == {"duplicate": 1, "low_quality": 1, "parsed": 1}
    assert overview.page_count == 1
    assert overview.text_char_count > 0
    assert overview.quality_flag_counts["no_text_layer"] == 1
    assert overview.language_counts["en"] == 1
    assert overview.metadata_coverage["title"] == 1
    assert overview.metadata_coverage["author"] == 1


def test_overview_counts_validated_evidence_without_claiming_semantics(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "The study evaluates alpha.", author="Author")
    snapshot = build_snapshot(tmp_path)
    document = snapshot.included_documents[0]
    validated = {
        "schema_version": "0.1",
        "snapshot_id": snapshot.snapshot_id,
        "papers": [
            {
                "paper_id": document.paper_id,
                "claims": [],
                "evidence": [
                    {
                        "evidence_id": "E1",
                        "artifact_id": document.artifact_id,
                        "page_number": 1,
                        "quote": "The study evaluates alpha",
                    }
                ],
            }
        ],
        "directions": [],
        "relations": [],
        "controversies": [],
        "frontier_candidates": [],
    }

    overview = build_corpus_overview(snapshot, validated_analysis=validated)

    assert overview.evidence_count == 1
    assert overview.evidence_paper_count == 1
    assert overview.research_direction_count == 0
    assert overview.semantic_status == "validated_input_only"
    assert overview.to_dict()["snapshot_id"] == snapshot.snapshot_id


def test_overview_is_deterministic_for_same_snapshot(tmp_path: Path) -> None:
    make_pdf(tmp_path / "paper.pdf", "中文 alpha mixed text。", title="Mixed")
    snapshot = build_snapshot(tmp_path)

    first = build_corpus_overview(snapshot).to_dict()
    second = build_corpus_overview(snapshot).to_dict()

    assert first == second
