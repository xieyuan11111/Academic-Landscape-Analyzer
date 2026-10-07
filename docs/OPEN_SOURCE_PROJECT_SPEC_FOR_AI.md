# Academic Landscape Analyzer：开源版 AI 协作规范

> 这是一份面向 AI agent、代码助手和自动化编排器的公开说明。
> 文档不包含真实 API key、密码、cookie、内部主机地址、个人信息、工作目录或运行产物。
> 所有服务地址、凭据和数据目录都必须由使用者在本地配置。

## 0. 项目身份

- 包名：`academic-landscape-analyzer`
- CLI：`academic-landscape`
- Python：`>=3.12`
- 许可证：MIT
- 运行形态：本地、无头 CLI
- 输入：用户明确提供的一批学术 PDF
- 输出：可追溯的逐篇研究卡、跨论文关系、研究方向、时间线、前沿候选和 Markdown/JSON 报告
- 默认 embedding：通过环境变量配置的 OpenAI-compatible embedding endpoint；本项目示例默认使用 SiliconFlow 的 `BAAI/bge-m3`，但不得把此默认值当作密钥或固定服务要求
- 可选组件：本地 embedding、Neo4j、Graphiti direct-bulk
- 不包含：Web 服务、GUI、MCP server、任何真实凭据

## 1. 信任边界

### 1.1 权威数据

按可信度排序：

1. 用户提供的原始 PDF；
2. `CorpusSnapshot` 和分页文本；
3. SQLite `EvidenceSpan`；
4. 通过验证的逐篇研究卡与跨论文关系；
5. Qdrant、Neo4j、Graphiti、manifest、JSON/Markdown 报告。

第 5 类是可重建投影或派生输出，不能反向覆盖第 1–4 类。

### 1.2 AI 不得做的事

AI agent 不得：

- 把 API key、password、token、cookie 写入代码、日志、报告、测试产物或记忆；
- 输出用户本地绝对路径、个人姓名、内部主机地址或未公开运行产物；
- 自己编造 paper ID、evidence ID、页码或引文；
- 把检索结果或模型记忆直接当作 EvidenceSpan；
- 把论文间条件差异自动说成矛盾；
- 因 Graphiti 依赖存在就声称它已接管 CLI 主链；
- 因测试或子代理声称成功就跳过读回验证。

### 1.3 凭据规则

凭据只能来自：

- 当前 shell 的环境变量；
- 用户明确配置的本地 secrets 文件；
- 受保护的凭据管理器。

文档和示例只能使用：

```text
ACADEMIC_LANDSCAPE_API_KEY=<set-locally>
SILICONFLOW_API_KEY=<set-locally>
NEO4J_PASSWORD=<set-locally>
```

不得把示例占位符替换成真实值后提交。

## 2. CLI 合同

```bash
academic-landscape doctor
academic-landscape ingest INPUT_DIR --output SNAPSHOT_DIR
academic-landscape analyze SNAPSHOT_DIR --output OUTPUT_DIR
academic-landscape landscape SNAPSHOT_DIR --output OUTPUT_DIR
academic-landscape apply-analysis SNAPSHOT_DIR ANALYSIS.json --output OUTPUT_DIR
```

### 2.1 `doctor`

- 检查 Python、核心依赖、可选依赖和配置；
- 不打印 secret 内容，只报告是否设置及长度；
- 不调用 LLM；
- `--probe` 只允许进行明确的外部服务连通性检查；
- `--json` 用于机器读取；
- 退出码 `0` 表示没有阻塞问题，`1` 表示需要修复。

### 2.2 `ingest`

```bash
academic-landscape ingest ./papers --output ./snapshots/run1
```

源码当前的核心输入是 PDF。它会：

1. 发现 PDF；
2. 计算 SHA-256；
3. 用 PyMuPDF 提取分页文本和元数据；
4. 识别重复内容；
5. 记录解析失败、无文本层和稀疏文本；
6. 生成稳定的 `snapshot_id`。

输出通常包括：

```text
snapshot.json
report.md
pages/<artifact_id>.json
artifacts/<artifact_id>.pdf
```

### 2.3 `analyze`

对快照中的可解析论文执行逐篇两阶段抽取。LLM endpoint、model 和 key 必须由环境变量或 CLI 参数提供，不能写入仓库。

`--extraction-cache` 应用于长任务断点续传。缓存不属于证据权威层，删除后可以从原始快照重建。

### 2.4 `landscape`

执行语义条目、embedding、聚类、方向命名、方向比较、阶段重建、关系抽取、证据族、来源卡、争议卡、前沿和时间事件装配。

当前 CLI 的默认输出是：

```text
landscape.json
landscape.md
```

专用运行脚本可以额外生成审计文件、PDF 或投影产物；不要把这些额外产物误写成 CLI 默认输出。

embedding 路线：

- `--embedding-backend siliconflow`：默认 API 路线；
- `--embedding-backend local`：显式本地路线；
- 不允许 provider 失败时静默换模型或换维度。

### 2.5 `apply-analysis`

```bash
academic-landscape apply-analysis SNAPSHOT_DIR ANALYSIS.json --output OUTPUT_DIR
```

外部 AI JSON 必须先通过当前快照绑定的校验，包括结构、论文归属、证据 ID、页码和引文存在性。校验失败应拒绝写入。

## 3. 真实数据流

```text
PDF directory
  -> build_snapshot
  -> CorpusSnapshot
  -> PaperExtractionService
       Stage 1: model emits short hint + page
       -> quote_resolver finds a real contiguous source span
       -> evidence validation
       Stage 2: model sees only validated evidence
       -> PaperExtractionResult
  -> semantic items + embeddings
  -> clustering + direction naming
  -> direction comparison + stage reconstruction
  -> paper-pair relation extraction
  -> evidence family + evidence tier
  -> source cards + debate cards
  -> frontier analysis + cross-direction merge
  -> deterministic timeline events
  -> LandscapeResult
  -> landscape.json + landscape.md
```

当前源码的 `LandscapePipeline.run()` 是顺序编排器。不要依据历史报告或模块名称推断 CLI 自动启用了某个外部存储服务；必须先看当前 `cli.py` 和实际运行参数。

## 4. 证据契约

### 4.1 EvidenceSpan

概念结构：

```json
{
  "evidence_id": "E1",
  "artifact_id": "artifact_<hash>",
  "page_number": 1,
  "quote": "a contiguous source span from that page",
  "locator": "optional locator"
}
```

跨论文层使用：

```text
paper_<hash>::E1
```

### 4.2 两阶段隔离

Stage 1 只给锚点提示：

```json
{"evidence_id":"E1","hint":"short phrase","page_number":1}
```

`quote_resolver` 通过代码在真实页文本中：

- 进行有序 token 匹配；
- 控制匹配分数、gap 和 span；
- 扩展到句子边界；
- 保护原始锚点不被 clamp 切掉；
- 返回页面中的连续原文。

Stage 2 只能引用 Stage 1 已验证证据。新造或改写 evidence 必须拒绝，例如 `research_card_new_evidence`。

### 4.3 失败语义

重要失败包括：

- `quote_not_found`
- `evidence_owner_mismatch`
- `unknown_evidence_id`
- `invalid_page_number`
- `research_card_new_evidence`
- schema validation error

失败不得被伪装成成功，也不得以空字符串引文代替。

## 5. 关系契约

唯一 canonical vocabulary 位于 `relation_types.py`：

```text
supports
extends
revises
qualifies
challenges
contradicts
parallel
same_name_different_meaning
surface_conflict
conditional_difference
not_comparable
unknown
```

`PaperRelation` 至少包含：

```json
{
  "relation_id": "rel_<hash>",
  "source_paper_id": "paper_<hash>",
  "target_paper_id": "paper_<hash>",
  "relation_type": "supports",
  "statement": "bounded relation assertion",
  "source_evidence_ids": ["paper_a::E1"],
  "target_evidence_ids": [],
  "difference_dimension": null,
  "comparability_note": "",
  "uncertainty": "",
  "snapshot_id": "snapshot_<hash>"
}
```

校验要求：

- source 和 target 必须属于当前 direction roster；
- 禁止 self-reference；
- source evidence 至少一条且必须归属 source；
- `challenges`/`contradicts` 必须有 target evidence；
- `same_name_different_meaning`、`surface_conflict`、`conditional_difference` 必须有差异维度；
- 差异维度只能是 `definition`、`object`、`method`、`conditions`、`conclusion`；
- unknown evidence、越界关系、非法类型应 fail-closed；
- evidence density warning 是审计提示，不应静默删除潜在真实关系。

语义原则：结果不同不自动等于矛盾；对象、方法和条件不同优先标注为可解释差异。

## 6. 时间事件

`timeline_events.py` 将阶段、关系和前沿信号转换为 `EvolutionEvent`。

事件类型包括：

```text
concept_origin
explicit_proposal
first_operationalization
initial_validation
replication
object_extension
method_innovation
mechanism_extension
boundary_revision
significant_challenge
counterexample
synthesis_or_alternative
```

时间排序由代码完成。事件必须绑定证据。

cutoff 语义：

```text
valid_at <= cutoff_at
AND (invalid_at == "" OR invalid_at == "open" OR invalid_at > cutoff_at)
```

`invalid_at` 表示有证据支持的失效或边界修订，不表示普通争议。

## 7. 投影与外部组件

### 7.1 SQLite

SQLite 权威层保存 snapshot、artifact、page、document 和 evidence。它是 evidence 复核和下游重建的来源。

### 7.2 Qdrant / RetrievalIndex

用于 chunk embedding、向量检索、词法检索和候选发现。检索结果不是证据；任何用于报告的引文都必须回到原文和 EvidenceSpan 验证。

### 7.3 自研 Neo4j 图和时态层

- `graph_projector.py`：结构化论文、方向、关系图；
- `temporal_projection.py`：自研纯 Cypher 时态投影；
- 两者都是可重建投影；失败不得覆盖权威层。

### 7.4 Graphiti / GTI

Graphiti 是可选能力，不是开源安装后的必需服务。使用时：

- 仅接收已验证的 relation/event 对象；
- 使用 direct-bulk writer；
- 不使用会重新解释事实的 public `add_triplet`；
- 使用 snapshot-scoped manifest 和 readback；
- 不把原始 PDF 或全文直接喂给 Graphiti；
- 不得因安装了 `graphiti-core` 就声称 CLI 已切换到 Graphiti。

LightRAG 不属于当前项目，不应重新加入依赖或流程。

## 8. AI 修改与验证规则

修改前：

1. 读取当前文件和真实调用方；
2. 判断修改属于权威层、分析层、投影层还是报告层；
3. 评估 cache/schema/输出兼容性；
4. 涉及外部写入时明确 scope 和回滚方法；
5. 不信任 agent 自报成功。

修改后：

1. 用新进程 import 当前源码；
2. 运行全量测试；
3. 运行 lint；
4. 重新读取写入的文件；
5. 对外部系统进行读回验证；
6. 重新扫描 secrets、绝对路径和个人信息；
7. 只在真实命令证明后报告完成。

公开发布前必须确认：

- `.env.local`、`.env.production` 等不会进入版本库；
- 没有真实 API key、password、token、cookie；
- 没有开发者本地绝对路径；
- 没有内部主机地址或运行目录；
- 没有个人姓名、联系方式或内部对话；
- 示例只使用占位符和环境变量；
- `runtime/` 生成产物不随源码包分发。

## 9. 最小验证命令

```bash
academic-landscape --help
academic-landscape doctor --json
python -m pytest tests/
ruff check src tests
academic-landscape ingest ./sample-papers --output ./out/snapshot
```

如果文档数字、模块数或输出名称与当前源码冲突，重新运行命令，不要沿用历史报告中的数字。
