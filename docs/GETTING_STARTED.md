# 快速上手

面向第一次使用本工具的人。全程约 10 分钟，不需要阅读源码。

---

## 1. 它做什么

你给它一个装着 PDF 的文件夹，它会读完全部文献，找出：

- 哪些文献在讲同一件事（**方向聚类**）
- 这个方向是怎么演化的（**阶段重建**）
- 文献之间是互相支持、互相挑战，还是条件不同（**12 类关系**）
- 哪些是共识、哪些是真争议（**证据卡 / 争议卡**）
- 现在正在出现什么（**前沿识别**）

每一个判断都附带**可回查的原文页码和逐字引文**。

---

## 2. 装

需要 **Python 3.12 或更高版本**。

```bash
git clone <this-repo>
cd academic-landscape-analyzer

# 核心安装
pip install .

# 想跑测试和 lint（可选）
pip install .[dev]
```

检查装好了没有：

```bash
academic-landscape doctor
```

---

## 3. 配置

需要两个密钥（都可以用同一个服务商的）：

| 用途 | 变量 | 说明 |
|---|---|---|
| 抽取与关系分析 | `ACADEMIC_LANDSCAPE_API_KEY` | 任何 OpenAI 兼容端点 |
| 语义向量 | `SILICONFLOW_API_KEY` | 默认路由；也可用本地模型替代 |

复制模板：

```bash
cp .env.example .env.local
```

编辑 `.env.local`，填入你的密钥。**不要提交这个文件**（`.gitignore` 已拦截）。

把变量加载进当前 shell：

**macOS / Linux（bash、zsh）**
```bash
set -a && . ./.env.local && set +a
```

**Windows（PowerShell）**
```powershell
Get-Content .env.local | ForEach-Object {
  if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
    Set-Item -Path "env:$($matches[1].Trim())" -Value $matches[2]
  }
}
```

再确认一次：

```bash
academic-landscape doctor
```

看到 `结论 PASS` 就可以开始了。（它只显示密钥长度，绝不打印内容。）

---

## 4. 跑

### 第一步：把 PDF 放进一个文件夹

支持 `.pdf`，同目录的 `.docx` 也会尝试解析，`.zip` 会被展开。
重复文件、解析失败的文件都会被记录，不会中断整个流程。

### 第二步：建立快照

```bash
academic-landscape ingest ./my-papers -o ./snapshots/run1
```

输出告诉你：识别了多少文件、多少是重复、多少解析失败，以及一个 `snapshot_id`。
**这一步不需要任何 API 调用**，纯本地。

### 第三步：全量分析

```bash
academic-landscape landscape ./snapshots/run1 -o ./out/run1
```

> 这一步会调用 LLM，**按文献数量计费**。209 篇的语料需要几万次调用。
> 建议先用 5–10 篇的小文件夹试跑，确认输出符合预期再上全量。

产出三个文件：

| 文件 | 给谁看 |
|---|---|
| `landscape.md` | 人：叙述性综述报告 |
| `landscape.audit.md` | 人 + 机器：逐条证据与校验记录 |
| `landscape.json` | 机器：完整结构化结果 |

### 中断了怎么办

`landscape` 支持断点续传。用同一个 `--extraction-cache` 文件重跑：

```bash
academic-landscape landscape ./snapshots/run1 -o ./out/run1 \
  --extraction-cache ./cache/papers.json
```

已完成的论文会从缓存读取，只跑没做完的。

---

## 5. 命令速查

| 命令 | 用途 | 需要 API |
|---|---|---|
| `doctor` | 环境自检 | 否 |
| `ingest` | PDF → 快照 | 否 |
| `analyze` | 单篇论文抽取 | 是 |
| `landscape` | 全量景观分析（主命令） | 是 |
| `apply-analysis` | 校验外部 AI 产出的 JSON | 否 |

每个命令的完整参数：`academic-landscape <命令> --help`

---

## 6. 常见问题

**`missing API key: set the XXX environment variable`**
变量没加载进当前 shell。回到第 3 步。`academic-landscape doctor` 能告诉你缺哪个。

**引文校验失败（`quote_not_found`）**
说明模型给的锚点在原文里找不到。工具会拒绝这条证据而不是放行——这是设计。
如果你看到大量失败，说明换的模型对"逐字定位"不敏感，试试别的模型。

**`Insufficient Balance` / HTTP 402**
服务商余额不足。换一个端点，或等充值。

**抽取特别慢**
正常。`landscape` 有大量 LLM 调用，且是"方向级 + 分块"推进的，几十篇语料跑几小时是常态。
用 `--extraction-cache` 保证可断点续传。

**想用本地 embedding，不想花钱**
```bash
pip install .[local-embedding]
academic-landscape landscape ./snapshots/run1 -o ./out/run1 --embedding-backend local
```

---

## 7. 输出怎么读

`landscape.md` 是给人读的，结构大致是：

1. **结论** — 三条最重要的判断
2. **领域版图** — 几个方向，每个多少篇
3. **演化阶段** — 这个领域怎么走到今天
4. **共识** — 哪些结论比较稳
5. **争议** — 哪里需要把条件讲清楚
6. **前沿** — 什么正在出现
7. **证据卡** — 每个判断的原文出处（附录）

每张证据卡都会标明：证据层级、原文引文、PDF 页码、适用条件、不确定性，
以及**这篇论文共有多少条引文、本卡展示了几条**（证据密度披露）。

---

## 许可

MIT。
