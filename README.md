# Academic Landscape Analyzer

证据可追溯的学术文献群落分析工具。给定一批封闭语料（PDF），它能找出这些文献的**共同点、前沿方向、研究阶段、差异与争议**，并为每一个判断附上可回查到原文页码的逐字证据。

面向 AI 调用的无头 CLI，MIT 许可，Python ≥ 3.12。

> **第一次使用请看 → [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**
> 装好、配置好、跑通，约 10 分钟。

---

## 它解决什么问题

面对几十上百篇文献，人读不过来；让大模型直接总结，又会编造引文、丢掉出处。本工具把两者拆开：

- **模型只做语义工作**：给锚点短语、命名方向、判断关系类型。
- **代码做事实工作**：在原文里定位引文、校验逐字性、维护时间顺序、生成证据 ID。

任何一句进入报告的判断，都能顺着 `paper_id::E1 → SQLite EvidenceSpan → PDF 第 N 页` 找回原文。

---

## 端到端流程

完整流程图（单文件、无依赖，浏览器打开即可）：
[docs/PIPELINE_ARCHITECTURE.html](docs/PIPELINE_ARCHITECTURE.html)

```
PDF 语料库
   │
   ▼
① Ingest 快照            解析 PDF → 页面文本 → 不可变快照（去重/质检/snapshot_id）
   │
   ▼
② 两阶段论文抽取          Stage 1 模型给锚点 hint → 代码在原文定位连续逐字片段
   │                      Stage 2 只见已验证证据 → 输出研究卡（问题/方法/结论/局限）
   │
   ├──────────────────────────────┐
   ▼                              ▼
③ 语义条目 + 聚类            权威证据层（SQLite）
   │  embed → 聚类 → 方向命名       │  EvidenceSpan · quote_exists 硬校验
   ▼                              ▼
④ 跨论文分析                来源卡 / 辩论卡 / 证据层级
   │  方向比较 · 阶段重构 · 证据族      一手 / 二手 / 综合 / 未决
   ▼
⑤ 前沿与未来工作            ⑥ 关系抽取（12 类，fail-closed）
   │  共识/差异/争议/未决             │
   ▼                              ▼
                        ⑦ 时间线事件（确定性时序）
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
   生产投影（当前主链）                  平行验证投影（可选，已实测）
   Neo4j 图 · M5 检索 · Qdrant 向量        getzep Graphiti（GTI）
   自建时序层（非 getzep Graphiti）        只吃结构化已验证条目 + manifest
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
                         报告输出
                         landscape.md（人读叙述）
                         landscape.audit.md（审计层）
                         landscape.json（机器层）
```

---

## 证据契约

这是整个系统的核心约束，不满足即拒绝入库：

1. **恰好一个来源页** —— 每条证据绑定唯一的 `(artifact_id, page_number)`。
2. **连续逐字原文** —— 引文必须是页面文本的连续子串，由 `quote_resolver` 从原文切出，模型不得抄写。
3. **quote_exists 硬校验** —— 归一化（连字符/连字/空白折叠）后必须在页文本中命中；未命中即 `quote_not_found`，绝不放行。
4. **全局证据 ID** —— 跨论文层统一为 `paper_id::E1` 形式，杜绝 ID 碰撞。
5. **两阶段隔离** —— Stage 2 只能看到已通过校验的证据；若 Stage 2 新造或改动引文，直接报 `research_card_new_evidence`。
6. **确定性时序** —— 时间线顺序由代码从证据/年份推出，LLM 不得编造先后。
7. **句界扩展** —— 证据摘录以锚点为中心扩展到完整句子（保留锚点、不切词），保证读者能看到作者实际的判断。

### 权威 vs 投影

| 层 | 内容 | 性质 |
|---|---|---|
| **权威** | 原始 PDF、解析页文本、SQLite 快照、EvidenceSpan | 唯一事实来源，只增不改 |
| **生产投影** | Qdrant 向量、Neo4j 图、M5 检索、自建时序层 | 可选的可重建派生层 |
| **可选 Graphiti 投影** | getzep Graphiti direct-bulk | 只接收已验证的结构化关系/事件，不是原始证据层 |

> 自研 `temporal_projection` 使用官方 Neo4j driver 和 Cypher；它不是 getzep/graphiti。Graphiti 是可选依赖，是否启用由使用者的部署方案决定。

- 原始 PDF / 全文不直接进入 Graphiti。
- Graphiti 只应走 direct-bulk 路径，避免公共 `add_triplet` 重新解释已经验证的学术证据。
- LightRAG 不属于当前公开版本。

---

## 安装

```bash
# 核心（含 numpy / pydantic / scikit-learn / qdrant-client / neo4j / PyMuPDF / openai）
pip install .

# 开发工具链（pytest 9.1.1 + ruff 0.15.10，均已锁定版本）
pip install .[dev]

# 可选：本地离线 embedding（默认走 SiliconFlow API，不需要这组）
pip install .[local-embedding]

# 可选：Graphiti 时态图投影（GTI，唯一可选投影引擎）
pip install .[graphiti]
```

### 装完先自检

```bash
academic-landscape doctor
```

它会逐项检查 Python 版本、核心依赖、可选组件、环境变量和 Neo4j 配置，
每项给出 `PASS` / `WARN` / `FAIL` 与修复提示，退出码 `0` 表示就绪。

- **不打印任何密钥内容**，只显示长度
- **不消耗 API 额度**（`--probe` 只测 Neo4j 连通性，不调 LLM）
- 加 `--json` 输出机器可读结果，便于 CI 或脚本判断

## 配置

复制 `.env.example` 为 `.env.local` 并填入真实值（该文件已被 gitignore，**不要提交密钥**）：

```bash
cp .env.example .env.local
```

| 变量 | 必需 | 用途 |
|---|---|---|
| `ACADEMIC_LANDSCAPE_BASE_URL` | ✅ | 抽取/关系分析 LLM 端点（OpenAI 兼容） |
| `ACADEMIC_LANDSCAPE_MODEL` | ✅ | 模型名（如 `deepseek-v4-flash`） |
| `ACADEMIC_LANDSCAPE_API_KEY` | ✅ | LLM 密钥 |
| `SILICONFLOW_API_KEY` | 仅 `landscape` | Embedding 密钥（默认路由） |
| `SILICONFLOW_EMBEDDING_BASE_URL` | ✗ | 有默认值 `https://api.siliconflow.cn/v1` |
| `SILICONFLOW_EMBEDDING_MODEL` | ✗ | 有默认值 `BAAI/bge-m3`，1024 维 |

加载进当前 shell：

```bash
# macOS / Linux
set -a && . ./.env.local && set +a
```

```powershell
# Windows PowerShell
Get-Content .env.local | ForEach-Object {
  if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
    Set-Item -Path "env:$($matches[1].Trim())" -Value $matches[2]
  }
}
```

### 成本提示

`ingest` 完全本地、零调用。`analyze` 和 `landscape` 会按文献数量计费——
上百篇语料意味着几万次 LLM 调用。**建议先用 5–10 篇试跑**。
`landscape` 支持 `--extraction-cache` 断点续传，中断不会丢已完成的抽取。

---

## 使用

### 0. 环境自检（建议每次换机器后先跑）

```bash
academic-landscape doctor
```

### 1. 建立语料快照

```bash
academic-landscape ingest ./my-corpus -o ./snapshots/run1
```

输出 `snapshot.json` / `report.md`，含 `snapshot_id`、文档数、去重结果、解析失败清单。

### 2. 单篇论文抽取（可选，独立使用）

```bash
academic-landscape analyze ./snapshots/run1 -o ./extraction \
  --base-url ... --model ... --extraction-cache ./cache.json
```

### 3. 全量景观分析（主入口）

```bash
academic-landscape landscape ./snapshots/run1 -o ./landscape-run1 \
  --base-url ... --model ... \
  --extraction-cache ./cache.json
```

输出：
- `landscape.json` —— 机器层，全量结构化结果
- `landscape.md` —— 人读叙述报告

专用 runtime 脚本可以额外生成审计报告、PDF 或投影产物；它们不是当前 CLI `landscape` 默认输出。

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--embedding-backend` | `siliconflow` | `siliconflow`（API）或 `local`（离线显式选择） |
| `--embedding-model` | `BAAI/bge-m3` | SiliconFlow embedding 模型 |
| `--local-embedding-model` | `BAAI/bge-small-en-v1.5` | 仅 `local` 模式生效 |
| `--embedding-device` | `auto` | 本地 embedding 设备（auto/cpu/cuda） |
| `--current-year` | `2026` | 前沿/未来工作的时间基准 |
| `--n-clusters` | 自动 | 研究方向聚类数 |
| `--timeout` | `120` | LLM 调用超时（秒） |

### 4. 应用已缓存的抽取结果

```bash
academic-landscape apply-analysis ./snapshots/run1 ./analysis.json -o ./out
```

---

## 关系抽取（12 类）

词表由 `relation_types.py` 单点定义，提示词、校验、Neo4j 边标签共用同一来源：

| | | | |
|---|---|---|---|
| `supports` 支持 | `extends` 延伸 | `revises` 修订 | `qualifies` 限定 |
| `challenges` 质疑 | `contradicts` 矛盾 | `parallel` 并行 | `same_name_different_meaning` 同名异义 |
| `surface_conflict` 表面冲突 | `conditional_difference` 条件差异 | `not_comparable` 不可比 | `unknown` 未知 |

**方向一致性策略**：不自动拆分或剔除方向；低内聚方向标 `pending_review`；`parallel` 算有意义接触，只有 `not_comparable` 与 `unknown` 计为回避。`contradicts`/`surface_conflict` 是有据不相容，不能仅因结果不同就断言"矛盾"。

---

## 报告形态

`landscape.md` 是给人读的叙述报告，遵循 `docs/HUMAN_READABLE_REPORT_TEMPLATE.md`：

- 以**作者-年份**引用（如 `Eng（2018）`），正文不出现内部 ID / 配置哈希；
- 每个关键判断标注**证据层级**：一手（单篇自身结果）/ 二手（综述转述）/ 综合（跨文献独立证据）/ 未决（并行、不可比、有争议或未决）；
- **附录 A：证据卡** —— 每张卡含判断、论文、逐字原文、页码，可回查权威库；
- **附录 B：引用索引与证据附录**。

`landscape.audit.md` 是审计层（机器可查的完整字段），`landscape.json` 是全量机器层。

---

## 目录结构

```
academic-landscape-analyzer/
├── src/academic_landscape/        # 54 个模块
│   ├── cli.py · doctor.py         # 命令行入口（doctor / ingest / analyze / landscape / apply-analysis）
│   ├── ingest.py · snapshot.py    # ① 语料快照
│   ├── paper_extraction.py        # ② 两阶段抽取（Stage1 锚点 / Stage2 研究卡）
│   ├── quote_resolver.py          #    锚点 → 句界扩展 → 逐字引文
│   ├── sqlite_store.py            #    权威证据层（EvidenceSpan）
│   ├── semantic_items.py · clustering.py · direction_naming.py   # ③
│   ├── cross_paper.py · stage_reconstruction.py · evidence_family.py  # ④
│   ├── frontier_analysis.py · landscape_merge.py                # ⑤
│   ├── relation_extraction.py · relation_types.py               # ⑥ 12 类关系
│   ├── timeline_events.py · temporal_projection.py              # ⑦ 时间线
│   ├── source_cards.py · evidence_tier.py · debate_cards.py · citations.py
│   ├── embedding.py · embedding_backend.py      # Qdrant / SiliconFlow / local
│   ├── graph_projector.py                       # Neo4j 图投影
│   ├── graphiti_projection.py · graphiti_bulk_runtime.py # Graphiti（GTI，可选）
│   ├── pipeline.py                              # LandscapePipeline 主编排
│   └── landscape_report.py                      # 报告渲染
├── tests/                        # 84 个测试文件（test_*.py）+ conftest.py
├── docs/                         # 快速上手 / 报告模板 / 架构图 / 发布清单
├── runtime/                      # 真实运行脚本与产物（不入包）
├── pyproject.toml
└── LICENSE                       # MIT
```

---

## 开发

```bash
# 全量测试（默认离线，live 测试需显式 --run-live）
PYTHONPATH=src pytest

# Lint（版本已锁定 0.15.10，规则集 E4/E7/E9/F）
ruff check src tests
```

约定：
- 模型调用应使用 JSON 响应、稳定的 temperature 配置和可复现的 embedding 配置。
- 密钥只从环境变量或本地 secrets 读取，不进入语料、图谱、日志、报告或会话。
- 运行时服务地址、数据库路径和端口由使用者配置，不写死进公开文档。

---

## 验证与限制

项目包含离线单元测试、协议测试和可选 live-service 测试。公开分享前请在目标环境运行 `academic-landscape doctor`、测试套件和脱敏扫描；不要把某次私有语料运行的统计数字当作通用基线。

---

## 文档

对外分发包含以下文档：

| 文件 | 内容 |
|---|---|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | **快速上手** —— 面向首次使用者 |
| [docs/OPEN_SOURCE_PROJECT_GUIDE_FOR_HUMANS.md](docs/OPEN_SOURCE_PROJECT_GUIDE_FOR_HUMANS.md) | **开源版人读说明** —— 脱敏的完整项目说明 |
| [docs/OPEN_SOURCE_PROJECT_SPEC_FOR_AI.md](docs/OPEN_SOURCE_PROJECT_SPEC_FOR_AI.md) | **开源版 AI 规范** —— 脱敏的机器协作说明 |
| [docs/OPEN_SOURCE_RELEASE_MANIFEST.md](docs/OPEN_SOURCE_RELEASE_MANIFEST.md) | **开源发布清单** —— 公开文件白名单与脱敏检查 |
| [docs/HUMAN_READABLE_REPORT_TEMPLATE.md](docs/HUMAN_READABLE_REPORT_TEMPLATE.md) | 人读报告模板（输出长什么样） |
| [docs/PIPELINE_ARCHITECTURE.html](docs/PIPELINE_ARCHITECTURE.html) | 端到端流程图（单文件，浏览器直接打开） |
| [docs/ARCHITECTURE_20260927.html](docs/ARCHITECTURE_20260927.html) | 架构总览（分层、模块、数值） |
| [docs/ARCHITECTURE_FLOW_20260927.html](docs/ARCHITECTURE_FLOW_20260927.html) | 主数据流图（每条箭头对应一次真实函数调用） |

---

## 许可

MIT
