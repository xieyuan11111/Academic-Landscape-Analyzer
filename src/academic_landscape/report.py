"""Human-readable and machine-readable output for the analyzer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_store import archive_snapshot
from .corpus_overview import CorpusOverview, build_corpus_overview
from .models import CorpusSnapshot


@dataclass(frozen=True)
class ReportPaths:
    output_dir: Path
    json_path: Path
    markdown_path: Path


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_pages(snapshot: CorpusSnapshot, output_dir: Path) -> None:
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    for document in snapshot.documents:
        if not document.pages:
            continue
        _json_dump(
            pages_dir / f"{document.artifact_id}.json",
            {
                "artifact_id": document.artifact_id,
                "paper_id": document.paper_id,
                "relative_path": document.relative_path,
                "pages": [page.to_dict() for page in document.pages],
            },
        )


def _document_rows(snapshot: CorpusSnapshot) -> str:
    rows = [
        "| 文件 | 状态 | 页数 | 文本字符数 | 质量标记 |",
        "|---|---:|---:|---:|---|",
    ]
    for document in snapshot.documents:
        flags = ", ".join(document.quality_flags) or "—"
        rows.append(
            f"| `{document.relative_path}` | `{document.status}` | {document.page_count} | "
            f"{document.text_char_count} | {flags} |"
        )
    return "\n".join(rows)


def _paper_labels(analysis: dict[str, Any], snapshot: CorpusSnapshot) -> dict[str, str]:
    labels = {document.paper_id: document.filename for document in snapshot.documents}
    for paper in analysis.get("papers", []):
        if not isinstance(paper, dict):
            continue
        paper_id = paper.get("paper_id")
        title = paper.get("title") or paper.get("metadata", {}).get("title")
        if isinstance(paper_id, str) and isinstance(title, str) and title.strip():
            labels[paper_id] = title.strip()
    return labels


def _semantic_markdown(snapshot: CorpusSnapshot, analysis: dict[str, Any]) -> str:
    labels = _paper_labels(analysis, snapshot)
    sections: list[str] = [
        "## 研究方向候选",
        "",
    ]
    directions = analysis.get("directions", [])
    if not directions:
        sections.append("当前语义分析没有生成研究方向候选。")
    else:
        for direction in directions:
            if not isinstance(direction, dict):
                continue
            direction_id = direction.get("direction_id", "unknown")
            label = direction.get("label", "未命名方向")
            status = direction.get("status", "candidate")
            members = [labels.get(paper_id, paper_id) for paper_id in direction.get("paper_ids", [])]
            sections.extend(
                [
                    f"### {label} (`{direction_id}`)",
                    f"- 状态：`{status}`",
                    f"- 论文：{', '.join(f'`{member}`' for member in members) or '未列出'}",
                    f"- 证据：{', '.join(f'`{evidence_id}`' for evidence_id in direction.get('evidence_ids', [])) or '未列出'}",
                    "",
                ]
            )

    sections.extend(["## 语义证据卡", ""])
    evidence_count = 0
    for paper in analysis.get("papers", []):
        if not isinstance(paper, dict):
            continue
        for evidence in paper.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            evidence_count += 1
            quote = str(evidence.get("quote", "")).replace("\n", " ").strip()
            sections.append(
                f"- **{evidence.get('evidence_id', 'unknown')}** · "
                f"`{labels.get(paper.get('paper_id'), paper.get('paper_id', 'unknown'))}` · "
                f"第 {evidence.get('page_number', '?')} 页：{quote}"
            )
    if evidence_count == 0:
        sections.append("当前语义分析没有证据卡。")

    sections.extend(["", "## 争议与前沿候选", ""])
    controversies = analysis.get("controversies", [])
    frontiers = analysis.get("frontier_candidates", [])
    sections.append(f"- 争议候选：{len(controversies)} 条")
    sections.append(f"- 前沿候选：{len(frontiers)} 条")
    sections.append("- 本阶段只呈现经过来源校验的 AI 候选，不把候选自动升级为学科事实。")
    return "\n".join(sections)


def _corpus_overview_markdown(overview: CorpusOverview) -> str:
    statuses = ", ".join(f"`{name}`: {count}" for name, count in overview.status_counts.items()) or "无"
    languages = ", ".join(f"`{name}`: {count}" for name, count in overview.language_counts.items()) or "无可判定文本"
    metadata = ", ".join(f"`{name}`: {count}" for name, count in overview.metadata_coverage.items())
    return f"""## 语料概览

- **快照 ID**：`{overview.snapshot_id}`
- **文件 / 已解析 / 去重论文**：{overview.document_count} / {overview.parsed_document_count} / {overview.unique_paper_count}
- **页数 / 文本字符数**：{overview.page_count} / {overview.text_char_count}
- **文件状态**：{statuses}
- **语言线索**：{languages}
- **元数据覆盖**：{metadata}
- **已校验证据**：{overview.evidence_count} 条，覆盖 {overview.evidence_paper_count} 篇论文
- **语义输入状态**：`{overview.semantic_status}`

> 这是当前快照的确定性统计，不生成研究方向、争议和前沿结论。
"""


def _markdown(
    snapshot: CorpusSnapshot,
    analysis: dict[str, Any] | None = None,
    overview: CorpusOverview | None = None,
) -> str:
    status_counts = snapshot.status_counts()
    status_text = ", ".join(f"`{key}`: {value}" for key, value in status_counts.items()) or "无文件"
    low_quality = sum(1 for document in snapshot.documents if document.status != "parsed")
    semantic_status = "validated" if analysis is not None else "not_run"
    body = f"""# 学术语料库快照报告

- **快照 ID**：`{snapshot.snapshot_id}`
- **生成时间**：`{snapshot.created_at}`
- **输入目录**：`{snapshot.input_root}`
- **文档数**：{len(snapshot.documents)}
- **纳入文本分析的文档数**：{len(snapshot.included_documents)}
- **内容去重后的论文数**：{len(snapshot.unique_paper_ids)}
- **状态统计**：{status_text}
- **语义分析状态**：`{semantic_status}`

## 当前阶段说明

本报告已完成**文件清单、内容哈希、重复识别、PDF 页面文本抽取、基础元数据和质量状态**。

"""
    if analysis is None:
        body += "**尚未运行语义分析，因此不会伪装成研究方向或前沿结论。** 当前结果只描述在当前语料库内已经实际解析到的材料；研究方向、共识、差异、争议和前沿候选需要后续分析阶段生成。\n"
    else:
        body += "语义分析结果已经通过快照边界、页码范围和原文摘录校验；报告中的方向和关系仍是当前语料库内的 AI 候选，不代表整个学科。\n"

    body += f"""
{_corpus_overview_markdown(overview) if overview is not None else ''}
## 语料库质量

- `{low_quality}` 个文件未以正常文本状态纳入本阶段的文本分析。
- 任何 `duplicate`、`failed` 或 `low_quality` 文件都不会被静默当成正常证据。
- 页面级文本保存在 `pages/`，机器可读清单保存在 `snapshot.json`。

## 文件清单

{_document_rows(snapshot)}
"""
    if analysis is not None:
        body += "\n" + _semantic_markdown(snapshot, analysis) + "\n"
    body += """
## 下一步

后续语义分析模块将以这个快照为输入，逐条生成带论文、页码/章节和原文片段的证据记录；在证据不足时保留“尚无法判断”，不补写结论。
"""
    return body


def _analysis_manifest(analysis: dict[str, Any] | None) -> dict[str, Any]:
    if analysis is None:
        return {
            "semantic_status": "not_run",
            "research_directions": [],
            "directions": [],
            "controversies": [],
            "frontier_candidates": [],
            "limitations": [
                "本阶段只完成语料导入、解析、去重和证据页面保存。",
                "尚未运行跨论文语义分析。",
            ],
        }
    return {
        "semantic_status": "validated",
        "research_directions": analysis.get("directions", []),
        "directions": analysis.get("directions", []),
        "controversies": analysis.get("controversies", []),
        "frontier_candidates": analysis.get("frontier_candidates", []),
        "papers": analysis.get("papers", []),
        "relations": analysis.get("relations", []),
        "limitations": [
            "语义结果只对当前 corpus snapshot 有效。",
            "候选方向、争议和前沿不得外推为整个学科的确定事实。",
        ],
    }


def write_report(
    snapshot: CorpusSnapshot,
    output_dir: str | Path,
    analysis: dict[str, Any] | None = None,
) -> ReportPaths:
    if analysis is not None:
        from .semantic import validate_analysis_payload

        validation = validate_analysis_payload(snapshot, analysis)
        if not validation.valid:
            codes = ", ".join(sorted({error["code"] for error in validation.errors}))
            raise ValueError(f"analysis validation failed: {codes}")
        analysis = validation.payload

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    overview = build_corpus_overview(snapshot, validated_analysis=analysis)
    archive_snapshot(snapshot, output / "artifacts")
    _write_pages(snapshot, output)
    json_path = output / "snapshot.json"
    markdown_path = output / "report.md"
    manifest = snapshot.to_manifest_dict()
    manifest["corpus_overview"] = overview.to_dict()
    manifest["analysis"] = _analysis_manifest(analysis)
    manifest["artifacts"] = {
        "pages_directory": "pages",
        "pdf_directory": "artifacts",
    }
    _json_dump(json_path, manifest)
    markdown_path.write_text(_markdown(snapshot, analysis, overview), encoding="utf-8")
    return ReportPaths(output_dir=output, json_path=json_path, markdown_path=markdown_path)
