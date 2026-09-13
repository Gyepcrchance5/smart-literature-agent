# smart-literature-agent

> **一句话**：基于 DeepXiv SDK 与 Anthropic Messages API 的 Agent 驱动个人科研文献工具 —— 检索 arXiv 新论文、按 token 预算精读、生成中文摘要与 enrichment JSON、输出 Markdown 报告和 `index.json`，并用轻量 Runtime 统一承载 Planning、Tool Use、Memory、Reflection、问答和项目匹配能力。

## 项目简介

这是一个**面向个人科研的文献跟踪工具**，聚焦六个研究方向：模型压缩与轻量化、知识蒸馏、故障诊断、深度学习方法、边缘部署与推理优化、可解释性。

### 核心能力

1. **自动检索**：6 领域 × 40+ 关键词，调用 DeepXiv 遍历最近 7 天的新论文，按 DeepXiv score 过滤、去重后落盘候选清单。
2. **OpenAlex 质量信号增强**：对每篇候选调 [OpenAlex API](https://openalex.org)（免费免 key），补充 **venue 类型**（期刊 / 会议 / 预印本）、**venue h-index**、**最新引用数**、**作者列表** —— 让新论文被顶会/顶刊接收后能自动被识别。DOI 查不到的老论文走 title search + 相似度校验 fallback。本地 JSON cache 避免重复查询。
3. **智能精读**：按 `token_count` 分四档策略 —— `raw`（全文）/ `selected`（精选 Introduction/Method/Experiments/Conclusion）/ `preview`（10k 字符概要）/ `metadata_only`（DeepXiv 尚未 ingest 时的降级），对超 80k token 的长文也能优雅处理。
4. **中文技术解读 + 领域综述**：通过兼容 Anthropic Messages API 的 LLM 产出**结构化中文单篇技术解读**（方法拆解 / 关键公式解释 / 迁移映射 / 面向轴承故障诊断的技术路线 / 局限）与**跨论文领域综述**（问题聚类 / 主流技术路线 / 常用数据集 / 开放问题 / 对研究者的具体迁移建议）。
5. **公式提取**：从 arXiv e-print 下载论文原始 LaTeX 源码，解析 `equation / align / eqnarray / gather / multline` 环境和常用定界符，输出带**公式编号**、**\\label**、**前后 150 字上下文**的结构化 `.formulas.json`，供摘要和 Agent 使用。
6. **结构化 Enrichment JSON**：在生成摘要的同时，用 LLM 提取方法类型、核心模块、损失函数、可迁移组件、适用场景、字段相关性等结构化信息，输出 `.enrichment.json` sidecar 文件，供下游 agent 直接查询和匹配。
7. **四维综合评分 + 本周 TOP10**：`composite = 45% × 启发式相关性 + 25% × DeepXiv score + 20% × Venue 档次 + 10% × 引用数`（每维归一到 0-100）。新论文有顶刊/顶会加成，老论文有引用加成，两条路都能筛出好东西。生成 TOP10 合并 markdown 报告。
8. **跨论文综合创新分析**：对 TOP 论文做 LLM 驱动的交叉对比，输出共享问题景观 / 方法矩阵 / 模块融合创新方向 / 关键公式交叉引用。
9. **文献知识库问答（RAG）**：基于 enrichment JSON + 摘要构建知识库，支持中英文混合提问，关键词检索 + LLM 生成回答，带流式输出和引用溯源。
10. **项目代码匹配**：只读扫描科研项目代码（模型 / 损失函数 / 数据加载器），用 LLM 匹配文献中的可迁移方法，生成集成建议报告。
11. **Agent Runtime**：`TaskPlanner → ToolRegistry → MemoryStore → Reflection` 闭环，计划依赖校验、工具白名单、JSON 参数校验、重复/越权调用拦截、异常隔离和结构化 trace；以工具调用、LLM 回合和总输出 token 控制成本。
12. **Agent 评测与质量保障**：离线规划用例检查意图准确率和工具覆盖率；历史 trace 审计工具成功率、计划覆盖、策略合规、引用可追溯率、反思通过率和预算遵守率。

### 规模数据（实测）

| 项目 | 值 |
|---|---|
| 单次完整流水线耗时 | **7 分钟**（历史实测：85 候选 / 10 精读 / 11 摘要 / 4 领域综述 / 1 TOP10） |
| Token 消耗 | **~7 万 Token**（~49k input + 22k output） |
| 覆盖关键词 | 42 个（6 领域） |
| 输出文件类型 | 8 种：候选/精读 JSON / summary / enrichment / 公式 / Markdown 报告 / Agent trace / session 记忆 |
| 运行模式 | CLI / Agent 触发（`python src/run.py`），全程可控 |

### 设计亮点

- **凭证零明文落地**：LLM key / DeepXiv token 全部从环境变量或用户级配置读取，代码仓库里没有一个字节的 key。
- **支持第三方兼容代理**：通过 `ANTHROPIC_BASE_URL` + `LLM_MODEL` 即可切换直连官方或走兼容代理（含 cc-switch 等配置工具），不改一行代码。
- **失败优雅降级**：DeepXiv 的 search 和 paper ingest 是两条管道，搜到不代表能读全文；四档精读策略 + `failed_ids` + 重试 flag 完整覆盖。
- **结构化产物**：所有核心产物都是 Markdown / JSON，方便下游 LLM Agent 直接消费；未通过 Reflection 的回答不会作为可复用答案注入 session memory。
- **LLM 配置自动同步**：API key、base_url、model 自动从 Claude Code 的 `~/.claude/settings.json` 同步，无需在 `.env` 中重复配置。

---

核心依赖：**DeepXiv SDK**（专为 AI Agent 设计的科技文献基础设施，提供 CLI 与 MCP 接口）+ **Anthropic Python SDK**（调任何兼容 Messages API 的 LLM）。

## 目录结构

```
smart-literature-agent/
├── config/
│   └── keywords.yaml          # 六大领域关键词 + 搜索/输出配置 + LLM Provider 预设
├── src/
│   ├── __init__.py
│   ├── searcher.py            # 检索：DeepXiv search → 候选论文 + OpenAlex 增强
│   ├── reader.py              # 精读：4 档 token 预算策略
│   ├── summarizer.py          # 总结：单篇中文摘要 + enrichment JSON + 跨论文领域报告
│   ├── synthesizer.py         # 跨论文综合创新分析：公式交叉引用 / 模块融合方向
│   ├── reporter.py            # 综合评分 + TOP10 Markdown 报告
│   ├── formula_handler.py     # LaTeX 公式提取：5 种环境 + 4 种定界符
│   ├── arxiv_source.py        # arXiv e-print 下载 + 解压 + 主 .tex 定位
│   ├── enricher.py            # OpenAlex 质量信号增强（venue / citation / h-index）
│   ├── project_analyzer.py    # 科研项目代码扫描 + 文献→代码匹配分析
│   ├── qa_agent.py            # 文献知识库 RAG 问答系统
│   ├── agent_runtime.py       # Planning / Tool Use / Memory / Reflection 运行时 + 策略门
│   ├── agent_eval.py          # 离线规划评测 + Agent trace 审计
│   ├── run.py                 # Pipeline 编排器 + CLI flags
│   └── utils.py               # 日志 / 配置 / 去重 / DeepXiv CLI 调用封装
├── start.py                   # 兼容入口，转发到 src/run.py
├── output/
│   ├── papers/                # 单篇精读产物（.json / .summary.md / .enrichment.json / .formulas.*）
│   ├── reports/               # 领域综述 + TOP10 + 集成分析报告
│   ├── qa/                    # 问答历史记录
│   ├── agent_runs/            # Agent 工具调用与反思 trace
│   ├── agent_memory/          # 精简 session 记忆
│   └── index.json             # 全局 Agent 读取入口
├── data/
│   ├── seen_ids.json          # 已处理论文 ID（去重）
│   ├── candidates_*.json      # 每次 search 的候选集合
│   └── openalex_cache.json    # OpenAlex 查询缓存
├── logs/                      # 运行日志，按日期切分
├── requirements.txt
└── README.md
```

## 环境搭建

推荐使用 conda 隔离环境（本项目已按此路径验证过）。

```bash
# 1. 创建并激活环境
conda create -n smart-lit python=3.11 -y
conda activate smart-lit

# 2. 安装依赖
pip install -r requirements.txt

# 3. LLM 凭证配置（二选一）
#    方式 A：自动同步 Claude Code 配置（推荐）
#      无需额外配置，项目会自动读取 ~/.claude/settings.json 中的 API key、base_url 和 model
#    方式 B：手动配置 .env
#      cp .env.example .env
#      在 .env 中设置 ANTHROPIC_API_KEY、LLM_PROVIDER、LLM_MODEL 等

# 4. （首次使用）DeepXiv 会在首次调用时自动生成匿名 token
#    并写入 ~/.env，默认 daily limit 1000。如需提升额度，邮件联系
#    tommy@chien.io。也可手动配置：
deepxiv config

# 5. 验证 CLI
deepxiv --help
deepxiv search "knowledge distillation" --limit 3 --format json
```

### LLM 配置优先级与失败边界

不指定 `LLM_PROVIDER` 时，进程环境变量优先于 `~/.claude/settings.json`，再优先于项目 `.env`；
模型会依次读取 `LLM_MODEL`、Claude Code 的 `ANTHROPIC_MODEL`，最后使用默认模型。
指定 `LLM_PROVIDER` 后，provider 预设负责默认的 endpoint/model/auth mode，进程环境变量或项目 `.env`
中的显式覆盖仍然有效，但不会把 Claude Code 的凭证自动混入该 provider。这样收到 HTTP 401 时，
系统会将其记录为不可重试的 `authentication_error`，不自动换 key 或切换服务。

配置解析结果只记录 provider、模型、endpoint 和凭证来源，不记录 API key；未知 provider 会直接报配置错误。

### Windows 中文环境注意

DeepXiv CLI 输出中含 emoji 字符，在 Windows 默认 GBK 终端上会触发 `UnicodeEncodeError`。
本项目的 `src/utils.run_deepxiv()` 已经在子进程中强制设置 `PYTHONIOENCODING=utf-8` 和 `PYTHONUTF8=1`，
因此通过代码调用没问题。**如果你想直接在终端用 deepxiv**，先设置：

```bash
# bash
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1

# PowerShell
$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
```

### 国内网络注意

conda 和 pip 建议切清华镜像（本机已配置）：

- `~/.condarc` — 清华 Anaconda 镜像
- `%APPDATA%\pip\pip.ini` — 清华 PyPI 镜像

## 使用方式

### 一键流水线（推荐）

`src/run.py` 编排 search → read → summarize → field_report 全流程，支持增量运行和 failed 重试。

```bash
# 一次性增量运行（默认最多精读 10 篇）
python src/run.py

# 常用参数
python src/run.py --max-read 20           # 本轮多读几篇
python src/run.py --retry-failed          # 重试 failed_ids 里未 ingest 的论文
python src/run.py --skip-search           # 跳过 search，复用最新 candidates（省 API）
python src/run.py --no-llm                # 只 search + read，不调 LLM（省 token）
python src/run.py --skip-formulas         # 跳过从 arXiv 源码提取公式的阶段
python src/run.py --skip-report           # 只跑前面阶段，不出 TOP10 和索引
python src/run.py --top-n 20              # TOP N 改成 20
python src/run.py --field knowledge_distillation   # 只对指定领域生成综述

# 兼容入口（等价于 python src/run.py）
python start.py

# 独立运行各阶段（方便调试）
python src/reporter.py --top10            # 只重算 TOP10
python src/formula_handler.py <arxiv_id>  # 只对一篇论文提公式（arXiv 路线）
python src/arxiv_source.py <arxiv_id>     # 只下载 + 解压 arXiv 源码
```

增量规则：
- `seen_ids.json` 里的论文不会被重复精读
- `metadata_only`（DeepXiv 全文未 ingest）和 `failed` 的论文**不写 seen**，下次可重试
- 已有 `.summary.md` 的论文不重复生成摘要
- 每个领域 ≥ 2 篇可用摘要才生成综述（低于不跑）

### Agent Runtime 与离线评测

后续开发与作品交付路线见 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)：包含 LangGraph、
中断恢复、证据检索、任务工作台、协作对照和阶段验收标准。计划中的功能不代表当前已实现。

`status=completed` 仅表示模型生成了回答。任务成功还要求 `task_status=completed`：
所有计划步骤必须有满足依赖的成功工具调用，回答通过来源检查，并有可引用证据。
`incomplete`、`insufficient_evidence`、`needs_review` 和 `failed` 不进入已验证记忆。
旧记忆缺少任务完成标记时保留记录但不直接复用。来源检查仍不等于论断事实核验。

审计中的无样本指标为 JSON `null`，应显示为 N/A；`sample_counts` 给出有效样本数。
历史 trace 会重算完成度和反思，而不是照搬历史成功标记。

```bash
# 离线验收，可附加本地项目的有限只读扫描；不会调用 LLM
python scripts/acceptance_probe.py --out output/acceptance/local --project-path /path/to/project
python -m unittest discover -s tests -v
```

项目扫描支持 `core/` 和 `work/mainline/code/core/` 的直接 Python 文件、入口说明与
`work/mainline/configs/official_mainline.json`。不会递归读取数据、模型、历史 run 或归档。
类清单表示代码定义，正式配置中的启用/禁用声明优先；扫描不核验实验成绩。

Agent Runtime 复用检索、精读、质量审查和项目扫描模块，通过显式工具白名单执行多步任务：

```bash
# 单次 Agent 任务：需要可用的 Anthropic-compatible LLM 配置
python src/run.py --agent-question "这周有哪些模型压缩论文值得优先阅读？"
python src/run.py --agent-question "2411.11707 值得作为研究依据吗？" --agent-session paper-review
python src/run.py --agent-question "比较 2411.11707 和 2501.00001 的方法差异"

# 不调用 LLM：检查规划器是否覆盖核心意图和工具
python src/run.py --agent-eval

# 不调用 LLM：审计已经保存的 output/agent_runs/*.json
python src/run.py --agent-run-audit
```

每次 Agent 运行保存一份结构化 trace（若目录不可写则保留回答并给出保存告警），其中包含 plan、步骤依赖、工具策略事件、工具参数与结果摘要、token usage、终止原因、引用来源和 reflection 结果。`inspect_project` 只读扫描目标项目，不提供任意 shell 或写文件工具；`compare_papers` 只读取已有精读产物。

### 单模块调试

```bash
python src/utils.py                       # utils 自检
python src/searcher.py                    # 单关键词冒烟
python src/searcher.py --all              # 全量批量检索 6 领域
python src/reader.py <arxiv_id>           # 精读单篇
python src/summarizer.py <arxiv_id>       # 单篇摘要
python src/summarizer.py --field knowledge_distillation   # 指定领域综述
```

### 产物位置

```
data/
├── candidates_<YYYYMMDD>.json   # 每次 search 的候选集合
├── seen_ids.json                # 已精读完成的 arxiv_id（增量去重）
└── failed_ids.json              # DeepXiv 未 ingest / head 拿不到的

output/
├── papers/
│   ├── <arxiv_id>.summary.md        # 单篇中文摘要
│   ├── <arxiv_id>.enrichment.json   # 结构化 enrichment（方法 / 模块 / 公式 / 可迁移性 / 字段相关性）
│   ├── <arxiv_id>.formulas.json     # 结构化公式清单（每条含 latex/type/env/label/eq_num/上下文）
│   └── ...
├── reports/
│   ├── <field>_<YYYYMMDD>.md        # 领域综述报告
│   ├── weekly_top<N>_<YYYYMMDD>.md  # 本周 TOP 合并报告
│   └── integration_<YYYYMMDD>.json  # 项目集成分析报告
├── qa/
│   └── qa_<timestamp>.json          # 问答历史记录
├── agent_runs/
│   └── <timestamp>_<run_id>.json     # Agent 工具调用、成本和反思 trace
├── agent_memory/
│   └── <session>.json                # 精简 session 记忆（不含凭证和全文）
└── index.json                        # Agent 读取入口

data/papers/
└── <arxiv_id>.json                   # 内部精读产物（含 head + 策略 + 正文片段）

data/
├── arxiv_src/<arxiv_id>/        # 下载并解压的 arXiv LaTeX 源码（本地缓存）
├── candidates_<YYYYMMDD>.json   # 每次 search 的候选集合
├── openalex_cache.json          # OpenAlex 查询 cache
├── seen_ids.json / failed_ids.json
```

### 综合评分

`src/reporter.py::composite_score()` 归一化到 0-100：

```
composite = 45% × 启发式相关性（6 领域 × 1-5 分总和 / 30）
          + 25% × DeepXiv search score（截断到 10 归一化）
          + 20% × Venue 档次
          + 10% × log1p(citation_count) / log(1000)
```

权重常量在 `reporter.py` 顶部 `W_RELEVANCE / W_DEEPXIV / W_VENUE / W_CITATION`，直接改即可。

## 路线图

- [x] **步骤 1**：项目骨架 + conda 环境 + DeepXiv CLI 验证
- [x] **步骤 2**：接入 DeepXiv search / paper，批量检索 + 全文策略化读
- [x] **步骤 3**：接入 LLM（通过 Anthropic Messages API，模型可配置），单篇中文摘要 + 跨论文领域综述
- [x] **步骤 4**：一键流水线 `run.py` + 增量 / 失败重试
- [x] **步骤 5**：综合评分 + 本周 TOP10 Markdown 报告 + `index.json`
- [x] **步骤 6a (Phase 1)**：arXiv 论文公式提取（下载 e-print → 解析 LaTeX → 带编号/label/上下文的结构化产物）
- [x] **步骤 7 (v1.2.0)**：跨论文综合创新分析 `synthesizer.py` + 公式交叉引用 + 融合公式输出 + 历史论文池 + LLM Provider 预设系统
- [x] **步骤 8**：结构化 Enrichment JSON（方法 / 模块 / 可迁移性 / 字段相关性）+ 文献知识库 RAG 问答 + 项目代码匹配分析
- [x] **步骤 9**：文献知识库 RAG 问答 + 项目代码匹配分析 + LLM 配置自动同步 Claude Code
- [x] **步骤 10**：Agent Runtime（Planning / Tool Use / Memory / Reflection）+ 工具白名单 + 结构化 trace
- [x] **步骤 11**：离线规划评测与 Agent trace 质量审计（不调用 LLM）
- [ ] **步骤 12（后置）**：按真实需求接入 PDF / HTML 论文来源

## 后置扩展（不属于当前核心）

### PDF 路线（IEEE Xplore / ScienceDirect 下载的 .pdf）

1. **装 MinerU**（推荐，国内友好）：
   ```bash
   pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com
   magic-pdf --help
   ```
   首次运行会下载约 3-5 GB 的模型到 `~/.cache/modelscope`。**强烈建议有 GPU**（CPU 每页 30-60 秒）。

2. **替换 stub**：打开 `src/pdf_handler.py`，把 `extract_from_pdf` 的 `NotImplementedError` 删掉，按 docstring 里的实装步骤接入 MinerU（把 MinerU 输出的 markdown 喂给 `formula_handler.extract_from_latex` 复用 LaTeX 解析器）。

3. **想换 Marker**（更快、1-2 GB 模型）：`pip install marker-pdf`，同样修改 `pdf_handler.py`。

### HTML 路线（IEEE/ScienceDirect/Nature 的在线阅读页）

1. **装依赖**：
   ```bash
   pip install beautifulsoup4 lxml mathml-to-latex
   ```
2. **每个期刊写一个 adapter**（放 `src/html_handler.py`）：
   - 根据 URL host 分派（`ieee.org` / `sciencedirect.com` / `nature.com` / `science.org`）
   - 抓 HTML 后用 BeautifulSoup 定位正文和公式节点
   - MathML → LaTeX 用 `mathml-to-latex`
   - 公式节点提取和 MathML 转换由未来 adapter 自行负责
3. **登录态**：在校园网 IP 段内直接访问通常即可；如需 EZproxy，把 proxy URL 写进 requests session。

详细设计参考每个模块文件顶部的 docstring。

## 质量审查与研究上下文

论文质量不再只由 TOP 分数表示。对已有精读产物运行：

```bash
python src/run.py --quality-audit
```

每篇论文会生成 `output/papers/<id>.evidence.json`，包含质量等级（A/B/C）、
方法/数据集/基线/指标/消融检查、缺失证据、迁移性评分和产物来源。

如需结合具体科研瓶颈审查，复制 `config/research_context.example.yaml` 为本地
`config/research_context.yaml`，填写问题、瓶颈、基线和约束，再运行：

```bash
python src/run.py --quality-audit --research-context config/research_context.yaml
python src/run.py --paper-id 2411.11707 --research-context config/research_context.yaml
```

研究上下文默认只在本地质量审查中使用，不会自动发送给外部 LLM。

## DeepXiv CLI 备忘

```bash
# 搜索
deepxiv search "QUERY" --limit 10 --format json
deepxiv search "QUERY" --categories cs.AI,cs.LG --date-from 2024-01

# 读论文（ID = arXiv ID，如 2411.11707）
deepxiv paper <ID> --brief          # 简报
deepxiv paper <ID> --head           # 元数据 + 章节列表（含每章 TLDR）
deepxiv paper <ID> --section NAME   # 读指定章节（NAME 必须来自 --head 的 sections[].name）
deepxiv paper <ID> --preview        # 预览 ~10k 字符
deepxiv paper <ID> --raw            # 完整原文 markdown

# 其他
deepxiv trending                    # 近期热门
deepxiv wsearch "QUERY"             # Web 搜索
deepxiv serve                       # 启动 MCP Server
```
