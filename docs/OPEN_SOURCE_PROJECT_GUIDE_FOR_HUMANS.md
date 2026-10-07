# Academic Landscape Analyzer：开源版项目说明

> 本文是用于公开仓库和开源分享的说明版本。
> 文档不包含真实 API key、密码、cookie、内部机器地址、个人信息、本地工作目录或私有运行产物。
> 使用者需要在自己的环境中配置服务地址、凭据和数据目录。

## 一句话介绍

Academic Landscape Analyzer 是一个本地运行的学术文献群落分析工具。

给它一批 PDF，它可以帮助你梳理：

- 文献共同关注的研究问题；
- 研究方向和主题分支；
- 研究方法与概念的发展阶段；
- 文献之间的支持、延伸、修订和挑战关系；
- 条件差异、表面冲突和真正争议；
- 当前语料中正在出现的前沿候选；
- 每个判断对应的原始论文、PDF 页码和逐字引文。

它不是把摘要简单拼接起来的总结器，而是让模型负责理解、让程序负责核实：模型可以提出候选关系，但所有进入证据链的引文都必须回到 PDF 原文验证。

---

## 一、适合做什么

它适合用于：

- 综述论文的前期领域扫描；
- 对一批已选定的论文做结构化摸底；
- 发现研究方向和概念主线；
- 比较论文之间的结论、对象、条件和方法；
- 生成带证据卡的人读报告；
- 为后续综述、开题报告和研究计划建立阅读路线。

它不应被当作：

- 同行评议系统；
- 自动判定谁对谁错的裁判；
- 系统综述方法学的完整替代；
- 对整个学科的无偏普查；
- 不需要研究者复核的最终结论生成器。

报告结论只对导入的语料库有效。语料不完整时，结论也必须明确写成“当前语料范围内的发现”。

---

## 二、核心思路

普通的 AI 文献总结有一个风险：它可能给出合理的判断，却无法证明引用真的来自论文原文。

本项目把任务拆成两部分：

### 模型负责

- 提供论文研究卡；
- 发现概念和方法的语义相似性；
- 为候选方向命名；
- 判断论文之间可能存在的关系；
- 帮助重建概念发展阶段。

### 程序负责

- 读取 PDF；
- 生成不可变语料快照；
- 计算文件 hash 和识别重复；
- 在真实页面文本中定位引文；
- 校验页码、证据 ID 和所有权；
- 维护确定性的时间顺序；
- 生成可审计的 JSON 和 Markdown 输出。

一句话概括：

> AI 负责提出理解，代码负责证明出处。

---

## 三、架构概览

```text
PDF 文件夹
   │
   ▼
语料快照
   │  hash、重复识别、页面文本、质量状态
   ▼
逐篇论文两阶段抽取
   │  模型给短锚点 → 程序定位原文 → 模型只看已验证证据
   ▼
语义条目与 embedding
   ▼
研究方向聚类与命名
   ▼
方向内比较与阶段重建
   ▼
跨论文关系抽取
   ▼
证据族、来源卡、争议卡、证据层级
   ▼
前沿候选与时间事件
   ▼
JSON + Markdown 报告
```

项目还提供可选的检索和图投影模块：

- SQLite：权威证据层；
- Qdrant：向量检索；
- Neo4j：结构化关系和时态投影；
- Graphiti：已验证关系/事件的可选 direct-bulk 投影。

这些投影都可以从权威层重建，不应取代原始 PDF 和 EvidenceSpan。

架构图：

- `docs/PIPELINE_ARCHITECTURE.html`
- `docs/ARCHITECTURE_20260927.html`
- `docs/ARCHITECTURE_FLOW_20260927.html`

---

## 四、安装

需要 Python 3.12 或更高版本：

```bash
git clone <repository-url>
cd academic-landscape-analyzer
pip install .
```

开发环境：

```bash
pip install .[dev]
```

本地 embedding：

```bash
pip install .[local-embedding]
```

Graphiti direct-bulk：

```bash
pip install .[graphiti]
```

Graphiti 是可选组件，不安装它不影响基础导入、逐篇抽取和领域分析。

安装后运行：

```bash
academic-landscape doctor
```

`doctor` 会检查 Python、依赖、可选组件和配置，但不会打印密钥内容，也不会为了自检而调用 LLM。

---

## 五、配置

复制模板：

```bash
cp .env.example .env.local
```

然后只在本机填写：

| 变量 | 用途 | 说明 |
|---|---|---|
| `ACADEMIC_LANDSCAPE_BASE_URL` | LLM endpoint | OpenAI-compatible 地址 |
| `ACADEMIC_LANDSCAPE_MODEL` | LLM 模型 | 由使用者选择 |
| `ACADEMIC_LANDSCAPE_API_KEY` | LLM 凭据 | 只存本机环境变量 |
| `SILICONFLOW_API_KEY` | 默认 embedding 凭据 | 只在 `landscape` 的默认 API 路线需要 |
| `SILICONFLOW_EMBEDDING_BASE_URL` | embedding endpoint | 有默认值，可覆盖 |
| `SILICONFLOW_EMBEDDING_MODEL` | embedding 模型 | 默认示例为 `BAAI/bge-m3` |

不要把真实值写进：

- README；
- `.env.example`；
- Python 源码；
- 测试 fixture；
- JSON/Markdown 报告；
- issue、截图或 commit message。

建议先用 5–10 篇文献试跑。`ingest` 不需要 API，`analyze` 和 `landscape` 会产生 LLM/embedding 调用，成本和时间随语料量增加。

---

## 六、基本使用

### 1. 建立快照

```bash
academic-landscape ingest ./my-papers --output ./snapshots/run1
```

输入目录中的 PDF 会被读取、hash、解析并生成快照。

主要输出：

```text
snapshots/run1/
├── snapshot.json
├── report.md
├── pages/
└── artifacts/
```

这一步不调用 API，可以先检查 PDF 是否解析成功。

### 2. 单篇分析

```bash
academic-landscape analyze ./snapshots/run1 \
  --output ./out/extraction \
  --base-url "$ACADEMIC_LANDSCAPE_BASE_URL" \
  --model "$ACADEMIC_LANDSCAPE_MODEL" \
  --extraction-cache ./cache/papers.json
```

适合先验证模型质量和引文定位。

### 3. 全量领域分析

```bash
academic-landscape landscape ./snapshots/run1 \
  --output ./out/landscape \
  --base-url "$ACADEMIC_LANDSCAPE_BASE_URL" \
  --model "$ACADEMIC_LANDSCAPE_MODEL" \
  --extraction-cache ./cache/papers.json
```

如需本地 embedding：

```bash
academic-landscape landscape ./snapshots/run1 \
  --output ./out/landscape \
  --base-url "$ACADEMIC_LANDSCAPE_BASE_URL" \
  --model "$ACADEMIC_LANDSCAPE_MODEL" \
  --embedding-backend local \
  --embedding-device auto
```

### 4. 校验外部 AI 结果

```bash
academic-landscape apply-analysis \
  ./snapshots/run1 ./analysis.json --output ./out/validated
```

外部 JSON 必须通过证据和快照校验，不能因为模型声称“这是原文”就直接写入报告。

---

## 七、证据链为什么可靠

### 两阶段抽取

第一阶段，模型只提供短提示：

```json
{"evidence_id":"E1","hint":"short phrase","page_number":1}
```

程序随后在真实 PDF 页文本中完成定位，并将锚点扩展为完整句子。结果必须是该页文本中的连续片段。

第二阶段，模型只能读取已经验证的证据，不能新增或修改引文。

### Evidence ID

单篇论文内部可以使用 `E1`、`E2`。跨论文分析时会变成：

```text
paper_<hash>::E1
```

这样不同论文的本地编号不会冲突。

### 证据失败怎么办

如果页码错误、引文不存在、证据归属错误或模型提交了新证据，程序会拒绝该条结果或留下显式失败记录。它不会用空引文或猜测内容补齐报告。

---

## 八、项目分析的内容

### 研究方向

项目先把逐篇研究卡转换为语义条目，再通过 embedding 和聚类生成候选方向，最后由模型命名。低内聚方向不会被静默删除，而会提示人工复核。

### 发展阶段

阶段重建关注：

- 概念何时出现；
- 何时被操作化或验证；
- 何时扩展到新对象；
- 何时出现方法或机制创新；
- 何时出现边界修订或反例。

时间排序由代码完成，模型不能自行编造年代顺序。

### 跨论文关系

当前支持 12 类关系：

```text
supports                    支持
extends                     延伸
revises                     修订
qualifies                   限定
challenges                  质疑
contradicts                 直接矛盾
parallel                    并行接触
same_name_different_meaning 同名异义
surface_conflict            表面冲突
conditional_difference      条件差异
not_comparable              不可比
unknown                     未知
```

两个研究结果不同，不代表它们一定互相矛盾。需要同时考虑研究对象、实验条件、尺度、方法和结论边界。

### 证据族和证据层级

项目会尝试识别相同团队、数据集或综述带来的重复证据，并将判断分为：

- 一手；
- 二手；
- 综合；
- 未决。

这样报告不会把同一条证据的重复传播简单算成多条独立证据。

### 前沿候选

前沿判断综合近期性、跨方向重复、方法/对象/机制变化和证据传播情况。它是当前语料范围内的候选，不等于对整个学科的最终宣判。

---

## 九、输出文件

### `snapshot.json`

保存快照 ID、文件 hash、解析状态、重复关系、页数、质量标记和页面文件位置。

### `report.md`

描述导入结果和语料质量。导入报告不会把“文件已解析”伪装成“领域结论已完成”。

### `landscape.json`

完整机器结果，可用于下游 AI、统计、投影和自定义报告。

### `landscape.md`

人读/机器读混合的领域分析报告，通常包含：

1. 执行摘要；
2. 语料范围与局限；
3. 发展阶段；
4. 研究方向；
5. 共识；
6. 差异与争议；
7. 前沿候选；
8. 未来问题；
9. 证据卡和审计信息（视生成路径而定）。

证据卡应让读者看到判断、论文、页码、逐字引文、适用条件和不确定性。

---

## 十、组件边界

### SQLite：事实来源

保存原始页面和 EvidenceSpan，是下游验证的基础。

### Qdrant：向量检索

帮助发现语义相似内容，但检索命中不是证据。

### Neo4j：结构化投影

保存关系、方向和时态事件的可查询结构；可以从权威层重建。

### Graphiti：可选 direct-bulk 投影

Graphiti 只应接收已验证的结构化 relation/event。项目绕开会重新解释事实的 public `add_triplet`，不把原始 PDF 直接送入 Graphiti。

当前 Graphiti 通过了与自研时态路径的验证，但这不等于它已经成为默认主链。要切换主链，必须另做迁移、性能、失败恢复和读回验证。

### LightRAG

不属于当前版本，不应安装或恢复旧流程。

---

## 十一、限制

1. PDF 解析质量决定后续分析上限，扫描件、复杂排版、表格和公式可能需要额外 OCR。
2. 模型可能误判语义方向和关系，关键判断仍需人工回看原文。
3. 引文校验能证明出处存在，不能单独证明论文结论正确。
4. 语料选择会带来选择偏差。
5. 同一团队或数据集的多篇论文不一定代表多个独立证据。
6. 更换 LLM、embedding 模型或维度会影响可比性，不能静默切换。
7. Qdrant、Neo4j 和 Graphiti 都是派生层，应先备份、试运行、读回，再用于长期数据。
8. 服务商的限额、价格和可用性由使用者自行确认。

---

## 十二、推荐工作顺序

```text
1. 规定论文纳入标准
2. academic-landscape doctor
3. ingest 小样本
4. 检查 snapshot.json / report.md / pages/
5. analyze 小样本
6. 人工核对若干证据卡
7. 确认 LLM 和 embedding 配置
8. landscape 全量运行
9. 先阅读语料边界和局限
10. 再阅读共识、差异和前沿
11. 对重要判断回看原始 PDF
12. 使用 landscape.json 做后续自动处理
```

这套工具的目标不是替研究者做最后判断，而是把阅读中最费时间的“找结构、找关系、找出处”变得更可靠、更可复核。

## 许可

MIT。
