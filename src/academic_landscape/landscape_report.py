"""Render the complete PRD §9 discipline landscape report.

Assembles every validated analysis artifact into one human-readable report with
sections 9.1-9.11. Empty/insufficient sections say so explicitly rather than
fabricating content; candidate and corpus-scoped wording is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .direction_coherence import annotate_coherence
from .relation_extraction import RELATION_EXTRACTION_FAILED
from .serialization import to_jsonable


@dataclass(frozen=True)
class CorpusFacts:
    document_count: int
    parsed_count: int
    paper_count: int
    year_min: int
    year_max: int
    parse_failures: tuple[str, ...]
    duplicates: tuple[str, ...]


@dataclass(frozen=True)
class LandscapeResult:
    snapshot_id: str
    generated_at: str
    corpus: CorpusFacts
    directions: tuple[Any, ...]
    comparisons: tuple[Any, ...]
    reconstructions: tuple[Any, ...]
    family_report: Any
    frontier_report: Any
    merged_frontiers: tuple[Any, ...]
    future_work: dict[str, list[str]]
    graph_projection: Any = None
    # Set when projection failed. The graph is a rebuildable projection, not the
    # authority layer, so its failure must be recorded rather than raised: the
    # analysis and report stay valid and the projection can be re-run later.
    graph_projection_error: str | None = None
    relation_results: tuple[Any, ...] = ()
    analysis_version: Any = None
    debate_cards: tuple[Any, ...] = ()
    source_cards: tuple[Any, ...] = ()
    timeline_bundle: Any = None
    # paper_id -> "Author et al. (2021)" — the human-readable in-text form.
    # Used by the report renderer so prose cites author–year instead of the
    # raw internal ``paper_<hash>`` key; the key itself stays in the appendix
    # for back-reference.
    citations: dict[str, str] = field(default_factory=dict)


def _bullet(text: str) -> str:
    return f"- {text}"


def _cite(result: LandscapeResult, paper_id: Any) -> str:
    """Human-readable author–year form of a paper reference.

    Falls back to the raw internal id when the paper is unknown to the
    citation map (no metadata, no year) so the report never invents an
    author name.
    """
    if not isinstance(paper_id, str) or not paper_id:
        return ""
    return result.citations.get(paper_id, paper_id)


def _section_91(result: LandscapeResult) -> list[str]:
    corpus = result.corpus
    lines = [
        "## 9.1 执行摘要",
        _bullet(
            f"语料规模：{corpus.paper_count} 篇可分析论文"
            f"（{corpus.year_min}–{corpus.year_max}）；"
            f"解析失败 {len(corpus.parse_failures)}，重复 {len(corpus.duplicates)}"
        ),
        _bullet(f"研究方向：{len(result.directions)} 个"),
    ]
    for direction in result.directions:
        lines.append(_bullet(f"[{direction.status}] {direction.label}"))
    lines.append(
        _bullet(
            f"发展阶段重建：{len(result.reconstructions)} 个方向已重建；"
            f"前沿候选：{len(result.merged_frontiers)} 条"
        )
    )
    controversies = [
        controversy
        for comparison in result.comparisons
        for controversy in comparison.controversies
    ]
    lines.append(_bullet(f"跨论文争议：{len(controversies)} 处"))
    return lines


def _section_92(result: LandscapeResult) -> list[str]:
    corpus = result.corpus
    lines = [
        "## 9.2 本次语料范围与局限",
        _bullet(
            f"纳入：{corpus.document_count} 个文件，{corpus.parsed_count} 个可解析，"
            f"去重后 {corpus.paper_count} 篇，年份 {corpus.year_min}–{corpus.year_max}"
        ),
    ]
    if corpus.parse_failures:
        lines.append(_bullet("解析失败：" + ", ".join(corpus.parse_failures)))
    if corpus.duplicates:
        lines.append(_bullet("重复文件：" + ", ".join(corpus.duplicates)))

    coherence_by_id = annotate_coherence(result.directions, result.relation_results)
    if coherence_by_id:
        values = [c.coherence for c in coherence_by_id.values()]
        demoted = [c.direction_id for c in coherence_by_id.values() if c.low_coherence]
        lines.append(_bullet(
            f"方向一致性分布：{len(values)} 个方向，最低 {min(values):.0%}，"
            f"中位 {sorted(values)[len(values) // 2]:.0%}，最高 {max(values):.0%}"
        ))
        if demoted:
            lines.append(_bullet(
                f"低一致性（low_coherence）方向 {len(demoted)} 个，已降级为待复核，"
                "不计入主流方向：" + ", ".join(demoted)
            ))

    family = result.family_report
    if family is not None:
        lines.append(_bullet(f"识别到作者线索 {len(family.persons)} 条"))
        for person in family.persons:
            lines.append(
                _bullet(
                    f"作者 {person.key} 出现于 "
                    f"{len(person.paper_ids)} 篇：{', '.join(person.paper_ids)}"
                )
            )
        lines.append(_bullet(f"数据集家族 {len(family.datasets)} 个；"
                             f"综述类论文 {len(family.review_like_papers)} 篇"))
    lines.append(_bullet("本报告结论只对当前快照语料有效，不代表整个学科的完整面貌。"))
    return lines


def _section_93(result: LandscapeResult) -> list[str]:
    lines = ["## 9.3 学科从过去到现在的发展脉络"]
    if not result.reconstructions:
        lines.append("无：材料不足以重建发展阶段。")
        return lines
    for reconstruction in result.reconstructions:
        lines.append(
            f"### 方向 {reconstruction.direction_id}"
            f"（材料判断：{reconstruction.material_assessment}）"
        )
        for index, stage in enumerate(reconstruction.stages, start=1):
            propagated = "已传播" if stage.propagated else "单篇承载·待传播"
            lines.append(
                _bullet(
                    f"阶段 {index}｜{stage.label}（{stage.year_start}–{stage.year_end}）"
                    f"·{stage.change_dimension}·置信 {stage.confidence}·{propagated}"
                )
            )
            lines.append(f"  {stage.summary}")
        for turning_point in reconstruction.turning_points:
            lines.append(
                _bullet(
                    f"转折点 {turning_point.get('year')}："
                    f"{turning_point.get('statement')}"
                    f"（{_cite(result, turning_point.get('paper_id'))}）"
                )
            )
        if reconstruction.notes:
            lines.append(f"> 边界说明：{reconstruction.notes}")
    return lines


def _section_94(result: LandscapeResult) -> list[str]:
    lines = ["## 9.4 当前主流研究方向"]
    if not result.directions:
        lines.append("无：尚未形成可命名的研究方向。")
        return lines
    coherence_by_id = annotate_coherence(result.directions, result.relation_results)
    for direction in result.directions:
        coherence = coherence_by_id.get(direction.direction_id)
        status = direction.status
        if coherence is not None and coherence.status_override:
            status = coherence.status_override
        lines.append(
            f"### {direction.label}（{status}）"
        )
        if direction.description:
            lines.append(direction.description)
        if coherence is not None:
            # Which number are we stating: an exact score over every pair, or a
            # floor computed from the subset relation extraction judged?
            basis = "全配对评估" if coherence.is_exact else "部分配对，下界"
            lines.append(
                _bullet(
                    f"方向一致性：{coherence.coherence:.0%}"
                    f"（实质关系对 {coherence.substantive_pairs}/{coherence.judged_pairs}；"
                    f"评估覆盖 {coherence.judged_pairs}/{coherence.total_pairs} 对，{basis}；"
                    f"下界 {coherence.lower_bound:.0%}）"
                )
            )
            if coherence.low_coherence:
                lines.append(_bullet(
                    "**low_coherence**：成员论文之间多数为不可比/平行关系，"
                    "可能并非单一研究方向；已降级为待复核，不作为主流方向表述。"
                ))
        lines.append(_bullet(f"成员论文：{', '.join(direction.paper_ids)}"))
    return lines


def _section_95(result: LandscapeResult) -> list[str]:
    lines = ["## 9.5 当前热门议题"]
    report = result.frontier_report
    hot = [
        direction_id
        for direction_id, signal in (
            report.direction_signals.items() if report is not None else {}
        )
        if signal.get("popularity") == "hot"
    ]
    if not hot:
        lines.append("无：当前语料中没有可判定为热门的方向。")
    else:
        for direction_id in hot:
            lines.append(_bullet(f"{direction_id}：近期论文产出显著增加"))
    return lines


def _section_96(result: LandscapeResult) -> list[str]:
    lines = ["## 9.6 当前创新点"]
    report = result.frontier_report
    innovative = [
        direction_id
        for direction_id, signal in (
            report.direction_signals.items() if report is not None else {}
        )
        if signal.get("innovation") == "innovative"
    ]
    if not innovative:
        lines.append("无：近期没有足够密集的新概念线索支撑创新判定。")
    else:
        for direction_id in innovative:
            lines.append(_bullet(f"{direction_id}：近期出现多个新概念/方法线索"))
    return lines


def _section_97(result: LandscapeResult) -> list[str]:
    lines = ["## 9.7 前沿方向与成熟度"]
    emerging = [
        frontier
        for frontier in result.merged_frontiers
        if frontier.tier == "emerging_frontier"
    ]
    candidates = [
        frontier
        for frontier in result.merged_frontiers
        if frontier.tier != "emerging_frontier"
    ]

    lines.append("### 快速升温 / 已形成前沿")
    lines.append("（通过跨团队参与与近期增长校验）")
    if not emerging:
        lines.append("无：当前语料中没有线索同时满足新意、增长与跨团队条件。")
    for frontier in emerging:
        repeat = (
            f"（横跨 {len(frontier.direction_ids)} 个方向）"
            if frontier.repeat_count > 1 else ""
        )
        lines.append(_bullet(f"{frontier.statement}{repeat}"))

    lines.append("\n### 前沿候选")
    lines.append("（早期提法或单团队增长，尚待跨团队验证；按方向归组）")
    if not candidates:
        lines.append("无")
    grouped: dict[str, list[Any]] = {}
    for frontier in candidates:
        key = ", ".join(frontier.direction_ids)
        grouped.setdefault(key, []).append(frontier)
    for direction_key, fronts in sorted(grouped.items()):
        lines.append(f"**方向 {direction_key}**")
        for frontier in fronts:
            lines.append(_bullet(frontier.statement))
    return lines


def _section_98(result: LandscapeResult) -> list[str]:
    lines = ["## 9.8 共识、差异与争议"]
    if not result.comparisons:
        lines.append("无：没有可比较的方向。")
        return lines
    for comparison in result.comparisons:
        lines.append(f"### 方向 {comparison.direction_id}")
        lines.append("**共识**")
        if comparison.commonalities:
            for common in comparison.commonalities:
                lines.append(_bullet(common.statement))
        else:
            lines.append("无")
        lines.append("**差异**")
        if comparison.differences:
            for difference in comparison.differences:
                lines.append(
                    _bullet(
                        f"({difference.dimension}) {difference.statement}"
                    )
                )
        else:
            lines.append("无")
        lines.append("**争议**")
        if comparison.controversies:
            for controversy in comparison.controversies:
                lines.append(_bullet(controversy["statement"]))
        else:
            lines.append("无")
        if comparison.within_paper_tensions:
            lines.append("**论文内张力（待核）**")
            for tension in comparison.within_paper_tensions:
                lines.append(_bullet(tension["statement"]))
    return lines


def _section_99(result: LandscapeResult) -> list[str]:
    lines = ["## 9.9 未来趋势与未解决问题"]
    direct = [
        (paper_id, statement)
        for paper_id, statements in result.future_work.items()
        for statement in statements
    ]
    if direct:
        lines.append("**文献直接提出**")
        for paper_id, statement in direct:
            lines.append(_bullet(f"{_cite(result, paper_id)}：{statement}"))
    structural = [
        reconstruction.notes
        for reconstruction in result.reconstructions
        if reconstruction.notes
    ]
    if structural:
        lines.append("**依据结构的综合线索**")
        for note in structural:
            lines.append(_bullet(note))
    if not direct and not structural:
        lines.append("无：当前材料没有可支撑的趋势判断。")
    return lines


def _section_910(result: LandscapeResult) -> list[str]:
    lines = ["## 9.10 探索与阅读建议"]
    start_papers = [
        stage.key_paper_ids
        for reconstruction in result.reconstructions
        for stage in reconstruction.stages
    ]
    earliest = start_papers[0] if start_papers else ()
    lines.append(_bullet(f"了解领域起点：{', '.join(earliest) if earliest else '材料不足'}"))
    lines.append(
        _bullet(
            "理解主流方向：" + ", ".join(direction.direction_id for direction in result.directions)
            if result.directions else "理解主流方向：材料不足"
        )
    )
    controversy_pairs = [
        f"{controversy['statement']}"
        for comparison in result.comparisons
        for controversy in comparison.controversies
    ]
    lines.append(
        _bullet("对读核心争议：" + ("；".join(controversy_pairs) if controversy_pairs else "无"))
    )
    lines.append(
        _bullet(
            "追踪创新与前沿："
            + ", ".join(frontier.statement for frontier in result.merged_frontiers)
            if result.merged_frontiers else "追踪创新与前沿：暂无"
        )
    )
    lines.append(
        _bullet(
            "建议补充的材料："
            + (", ".join(result.corpus.parse_failures) if result.corpus.parse_failures
               else "当前无失败文件；建议补充早期年份文献以校验起点")
        )
    )
    return lines


def _section_911(result: LandscapeResult) -> list[str]:
    lines = ["## 9.11 证据附录"]

    # --- Paper index: the full back-reference for every author–year citation.
    # Prose cites "Opulente et al. (2019)"; this table is what resolves that
    # back to the internal paper_id used by the graph and the authority store.
    lines.append("**论文索引（作者—年份 → 内部标识）**")
    if result.citations:
        for paper_id in sorted(result.citations):
            lines.append(_bullet(
                f"{result.citations[paper_id]} → `{paper_id}`"
            ))
    else:
        lines.append("无")

    # --- Evidence appendix: every unique evidence_id with the full original
    # quote, the page/locator it was found at, the metadata of the paper it
    # came from, and when the quote was verified against page text.
    lines.append("**证据条目（完整原文 · 位置 · 来源元数据 · 核验时间）**")
    seen_evidence: set[str] = set()
    rendered_evidence = False
    for card in result.source_cards:
        for bucket, label in (
            (card.supporting, "支持"),
            (card.revising, "修正"),
            (card.opposing, "反向"),
        ):
            for ref in bucket:
                if ref.evidence_id in seen_evidence:
                    continue
                seen_evidence.add(ref.evidence_id)
                rendered_evidence = True
                meta = []
                if ref.title:
                    meta.append(f"《{ref.title}》")
                if ref.authors:
                    meta.append(ref.authors)
                if ref.year:
                    meta.append(f"{ref.year} 年")
                meta_text = " ".join(meta) or "来源元数据缺失"
                loc = ref.locator or (f"p{ref.page_number}" if ref.page_number else "位置缺失")
                verified = getattr(ref, "verified_at", "") or "未记录"
                lines.append(
                    f"- `{ref.evidence_id}`（{label}）｜"
                    f"{_cite(result, ref.paper_id)}｜{meta_text}"
                    f"｜{loc}｜核验于 {verified}"
                )
                if ref.quote:
                    lines.append(f"    > {ref.quote}")
    if not rendered_evidence:
        lines.append("无")

    lines.append("**研究方向成员关系**")
    for direction in result.directions:
        lines.append(
            _bullet(
                f"{direction.direction_id} {direction.label}："
                f"{', '.join(direction.paper_ids)}"
            )
        )
    if not result.directions:
        lines.append("无")

    lines.append("**观点时间线**")
    for reconstruction in result.reconstructions:
        for stage in reconstruction.stages:
            lines.append(
                _bullet(
                    f"{stage.year_start}–{stage.year_end} {stage.label}"
                )
            )
    if not result.reconstructions:
        lines.append("无")

    lines.append("**前沿候选**")
    for frontier in result.merged_frontiers:
        tier_label = (
            "快速升温前沿" if frontier.tier == "emerging_frontier" else "候选"
        )
        lines.append(_bullet(f"[{tier_label}] {frontier.statement}"))
    if not result.merged_frontiers:
        lines.append("无")

    lines.append("**跨论文关系（逐论文对）**")
    rendered_relation = False
    for relation_result in result.relation_results:
        for relation in relation_result.relations:
            rendered_relation = True
            dim = f"（{relation.difference_dimension}）" if getattr(relation, "difference_dimension", None) else ""
            lines.append(_bullet(
                f"{_cite(result, relation.source_paper_id)} --{relation.relation_type}--> "
                f"{_cite(result, relation.target_paper_id)}{dim}: {relation.statement}"
            ))
            lines.append(
                f"    `{', '.join(relation.source_evidence_ids + relation.target_evidence_ids)}`"
            )
    if not rendered_relation:
        lines.append("无")

    # A direction that lost its whole relation layer must be stated, not implied
    # by an empty list: silently returning nothing is how the first real MAG-50
    # run hid four directions that produced no relations at all.
    lines.append("**关系抽取覆盖**")
    failed_extractions = [
        error
        for relation_result in result.relation_results
        for error in relation_result.validation_errors
        if error.get("code") == RELATION_EXTRACTION_FAILED
    ]
    direction_total = len(result.relation_results)
    if direction_total:
        lines.append(
            _bullet(
                f"{direction_total} 个方向中 "
                f"{direction_total - len(failed_extractions)} 个成功、"
                f"{len(failed_extractions)} 个失败"
            )
        )
        for failure in failed_extractions:
            lines.append(
                _bullet(
                    f"{failure.get('direction_id')} 关系层为空"
                    f"（已尝试 {failure.get('attempts')} 次）："
                    f"{failure.get('reason') or failure.get('error') or '原因未记录'}"
                )
            )
        # Recovering from a ceiling overflow is not a failure, but the reader
        # still needs to know that a direction cost more calls than the others.
        for relation_result in result.relation_results:
            for note in getattr(relation_result, "extraction_trace", ()) or ():
                if note.get("fallback") == "pair_chunked":
                    lines.append(
                        _bullet(
                            f"{note.get('direction_id')} 改用配对分块"
                            f"（{note.get('pairs')} 对 → {note.get('calls')} 次调用，"
                            f"{note.get('chunk_size')} 对/次）"
                        )
                    )
    else:
        lines.append("未执行关系抽取。")

    lines.append("**演化事件链（时序轨·按时间排序）**")
    bundle = result.timeline_bundle
    events = list(bundle.events) if bundle else []
    if events:
        for event in sorted(events, key=lambda e: (e.year, e.event_type, e.subject)):
            source_part = (
                f" ← {event.source_paper_id}" if event.source_paper_id else ""
            )
            dim_part = (
                f"（{event.difference_dimension}）"
                if getattr(event, "difference_dimension", None) else ""
            )
            lines.append(_bullet(
                f"{event.year} [{event.event_type}] {event.subject}{dim_part}"
                f"：{event.paper_id}{source_part}"
            ))
            lines.append(f"    `{', '.join(event.evidence_ids)}`")
    else:
        lines.append("无")

    lines.append("**来源卡（判断级）**")
    if result.source_cards:
        # Tier distribution summary (PRD §10.1): how far the report may lean
        # on each body of evidence. Readers see the shape before the details.
        tier_counts: dict[str, int] = {}
        for card in result.source_cards:
            tier = getattr(card, "tier", "") or "未标注"
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        lines.append(_bullet(
            "证据分层分布：" + "；".join(
                f"{tier} {count} 张" for tier, count in sorted(tier_counts.items())
            )
        ))
        for card in result.source_cards:
            # Evidence tier is derived from the relation vocabulary plus the
            # evidence-family independence signal (PRD §10.1/§10.2).  The
            # source card stores the A/B/C/D judgment type; the tier states
            # how far the report may lean on it in prose.
            tier = getattr(card, "tier", "")
            tier_reason = getattr(card, "tier_reason", "")
            tier_part = f"·{tier}" if tier else ""
            lines.append(_bullet(
                f"[判断类型 {card.judgment_type}{tier_part}] {card.judgment}"
            ))
            if tier_reason:
                lines.append(f"    证据层级：{tier_reason}")
            if card.time_attribute:
                lines.append(f"    时间属性：{card.time_attribute}")
            if card.applicability:
                lines.append(f"    适用条件：{card.applicability}")
            if card.uncertainty:
                lines.append(f"    不确定性：{card.uncertainty}")
            for label, refs in (
                ("支持", card.supporting),
                ("修正", card.revising),
                ("反向", card.opposing),
            ):
                if refs:
                    lines.append(f"    {label}来源：")
                    for ref in refs:
                        meta = []
                        if ref.title:
                            meta.append(f"《{ref.title}》")
                        if ref.authors:
                            meta.append(ref.authors)
                        if ref.year:
                            meta.append(f"{ref.year} 年")
                        meta_text = " ".join(meta)
                        loc = ref.locator or (f"p{ref.page_number}" if ref.page_number else "")
                        verified = (
                            f"核验于 {ref.verified_at}" if getattr(ref, "verified_at", "") else ""
                        )
                        lines.append(
                            f"      {_cite(result, ref.paper_id)}"
                            f"（{ref.evidence_id}）"
                            f"{('：' + meta_text) if meta_text else ''}"
                            f"｜{loc}{('｜' + verified) if verified else ''}"
                        )
                        if ref.quote:
                            lines.append(f"      > {ref.quote}")
    else:
        lines.append("无")

    lines.append("**辩论候选卡（争议 → 按需剧场入口）**")
    if result.debate_cards:
        for card in result.debate_cards:
            lines.append(_bullet(f"争议问题：{card.question}"))
            lines.append(f"    是否适合进入剧场：{card.fit_for_theater}")
            for position in card.positions:
                lines.append(
                    f"    - {position.paper_id}：{position.stance}"
                )
            if card.shared_premises:
                lines.append(f"    共同前提：{'；'.join(card.shared_premises)}")
            if card.divergent_premises:
                lines.append(f"    分歧前提：{'；'.join(card.divergent_premises)}")
            if card.comparability_issues:
                lines.append(
                    f"    可比性问题：{'；'.join(card.comparability_issues)}"
                )
            if card.open_questions:
                lines.append(
                    f"    未解决问题：{'；'.join(card.open_questions)}"
                )
            if card.support_and_counter:
                refs_text = "；".join(
                    f"{ref.paper_id}：{ref.quote}" for ref in card.support_and_counter
                )
                lines.append(f"    支持与反向证据：{refs_text}")
    else:
        lines.append("无")

    lines.append("**图谱投影状态**")
    if result.graph_projection_error:
        lines.append(
            _bullet(f"投影失败（可重建层，非权威层）：{result.graph_projection_error}")
        )
        lines.append(
            _bullet("分析结果、来源卡与版本标识不受影响；修复后可直接重跑投影。")
        )
    elif result.graph_projection is not None:
        nodes = getattr(result.graph_projection, "node_count", "?")
        edges = getattr(result.graph_projection, "relationship_count", "?")
        lines.append(_bullet(f"投影成功：{nodes} 个节点，{edges} 条关系。"))
    else:
        lines.append("未启用图谱投影。")

    lines.append("**解析失败与低置信材料**")
    if result.corpus.parse_failures:
        for failure in result.corpus.parse_failures:
            lines.append(_bullet(failure))
    validation_errors = [
        error
        for comparison in result.comparisons
        for error in comparison.validation_errors
    ]
    for reconstruction in result.reconstructions:
        validation_errors.extend(reconstruction.validation_errors)
    if result.timeline_bundle is not None:
        validation_errors.extend(result.timeline_bundle.validation_errors)
    for error in validation_errors:
        lines.append(_bullet(str(error)))
    if not result.corpus.parse_failures and not validation_errors:
        lines.append("无")
    return lines


def render_landscape_report(result: LandscapeResult) -> str:
    header = [
        "# 学科发展与前沿态势报告",
        "",
        f"- 快照 ID：`{result.snapshot_id}`",
        f"- 生成时间：`{result.generated_at}`",
        (
            f"- 分析版本：`{result.analysis_version.version_id}`"
            f"（模型 {result.analysis_version.model}，配置哈希"
            f" `{result.analysis_version.config_hash[:12]}`）"
            if result.analysis_version is not None else ""
        ),
        (
            "- ⚠️ 图谱投影失败（投影为可重建层，不影响以上分析与来源卡）："
            f"`{result.graph_projection_error}`"
            if result.graph_projection_error else ""
        ),
        "",
    ]
    body: list[str] = []
    for builder in (
        _section_91,
        _section_92,
        _section_93,
        _section_94,
        _section_95,
        _section_96,
        _section_97,
        _section_98,
        _section_99,
        _section_910,
        _section_911,
    ):
        body.extend(builder(result))
        body.append("")
    return "\n".join(header + body) + "\n"


def landscape_result_to_dict(result: LandscapeResult) -> dict[str, Any]:
    """Pure JSON-safe view of the result (no ``default=str`` fallback needed).

    ``dataclasses.asdict`` cannot see the Pydantic models used for relations and
    evolution events, so it silently left them to be stringified into reprs.
    """
    payload = to_jsonable(result)
    if not isinstance(payload, dict):  # pragma: no cover - LandscapeResult is a dataclass
        raise TypeError("landscape result did not serialize to an object")
    return payload
