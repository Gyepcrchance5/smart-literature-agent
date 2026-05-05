# smart-literature-agent

> **一句话**：基于 DeepXiv SDK 与 Anthropic Messages API 的个人科研文献智能体 —— 交互式仪表盘一键启动，抓取 6 大研究方向的 arXiv 新论文 + 历史回溯 5 年、按 token 预算分级精读、用 LLM 产出中文摘要与跨论文领域综述 + 模块融合创新分析，综合评分输出本周 TOP10 合并报告 + 带 MathJax 的静态 HTML 索引。

## 项目简介

这是一个**面向个人科研的文献跟踪工具**，聚焦六个研究方向：模型压缩与轻量化、知识蒸馏、故障诊断、深度学习方法、边缘部署与推理优化、可解释性。

### 核心能力

1. **自动检索**：6 领域 × 40+ 关键词，调用 DeepXiv 遍历最近 7 天的新论文，按 DeepXiv score 过滤、去重后落盘候选清单。
2. **OpenAlex 质量信号增强**：对每篇候选调 [OpenAlex API](https://openalex.org)（免费免 key），补充 **venue 类型**（期刊 / 会议 / 预印本）、**venue h-index**、**最新引用数**、**作者列表** —— 让新论文被顶会/顶刊接收后能自动被识别。DOI 查不到的老论文走 title search + 相似度校验 fallback。本地 JSON cache 避免重复查询。
3. **智能精读**：按 `token_count` 分四档策略 —— `raw`（全文）/ `selected`（精选 Introduction/Method/Experiments/Conclusion）/ `preview`（10k 字符概要）/ `metadata_only`（DeepXiv 尚未 ingest 时的降级），对超 80k token 的长文也能优雅处理。
4. **中文技术解读 + 领域综述**：通过兼容 Anthropic Messages API 的 LLM 产出**结构化中文单篇技术解读**（方法拆解 / 关键公式解释 / 迁移映射 / 面向轴承故障诊断的技术路线 / 局限）与**跨论文领域综述**（问题聚类 / 主流技术路线 / 常用数据集 / 开放问题 / 对研究者的具体迁移建议）。
5. **公式提取**：从 arXiv e-print 下载论文原始 LaTeX 源码，解析 `equation / align / eqnarray / gather / multline` 环境 + `$$/\\[/\\($ /\\(/$ ` 定界符，输出带**公式编号**、**\\label**、**前后 150 字上下文**的结构化 `.formulas.json` + 可读 `.formulas.md` + MathJax 渲染的 HTML。方便 AI 理解或直接拷贝到你自己写的论文里。
6. **四维综合评分 + 本周 TOP10**：`composite = 45% × 启发式相关性 + 25% × DeepXiv score + 20% × Venue 档次 + 10% × 引用数`（每维归一到 0-100）。新论文有顶刊/顶会加成，老论文有引用加成，两条路都能筛出好东西。生成 TOP10 合并 markdown 报告。
7. **HTML 渲染 + 自动打开浏览器**：所有 markdown 产物一键转 GitHub 风格 HTML + **MathJax** 公式渲染，生成总览 `index.html`（本周 TOP / 领域综述 / 单篇摘要 + 公式速览），跑完自动弹出。
8. **DeepScientist 投喂包导出**：把本周 TOP 论文、摘要、公式和迁移路线整理成 `output/deepscientist_bundle/`，生成 `startup_prompt.md` / `literature_brief.md` / `hypotheses.md`，用于启动 DeepScientist 研究 quest。
9. **交互式启动器 + 增量 + 失败重试**：`python start.py` 仪表盘一键运行，实时展示项目状态；`seen_ids` 增量去重，`metadata_only` 和 `failed` 不写 seen 以便下次重试。

### 规模数据（实测）

| 项目 | 值 |
|---|---|
| 单次完整流水线耗时 | **7 分钟**（85 候选 / 10 精读 / 11 摘要 / 4 领域综述 / 1 TOP10 + HTML） |
| Token 消耗 | **~7 万 Token**（~49k input + 22k output） |
| 覆盖关键词 | 42 个（6 领域） |
| 输出文件类型 | 5 种：候选清单 JSON / 精读 JSON / 单篇 summary markdown / 领域综述 / HTML 索引页 |
| 运行模式 | 交互式启动器（`python start.py`），手动触发，全程可控 |

### 设计亮点

- **凭证零明文落地**：LLM key / DeepXiv token 全部从环境变量或用户级配置读取，代码仓库里没有一个字节的 key。
- **支持第三方兼容代理**：通过 `ANTHROPIC_BASE_URL` + `LLM_MODEL` 即可切换直连官方或走兼容代理（含 cc-switch 等配置工具），不改一行代码。
- **失败优雅降级**：DeepXiv 的 search 和 paper ingest 是两条管道，搜到不代表能读全文；四档精读策略 + `failed_ids` + 重试 flag 完整覆盖。
- **结构化产物**：所有中间产物都是 markdown / JSON，方便下游 LLM Agent 二次消费（后续可 MCP 化）。
- **面向 DeepScientist 的前置文献层**：本项目负责每周文献摄取、筛选、公式解释和迁移路线整理；DeepScientist 负责基于这些材料做 baseline 驱动的实验、finding 记忆和论文产出。

---

核心依赖：**DeepXiv SDK**（专为 AI Agent 设计的科技文献基础设施，提供 CLI 与 MCP 接口）+ **Anthropic Python SDK**（调任何兼容 Messages API 的 LLM）。

## 目录结构

```
smart-literature-agent/
├── config/
│   └── keywords.yaml        # 六大领域关键词 + 搜索/输出配置
├── src/
│   ├── __init__.py
│   ├── searcher.py          # 检索：DeepXiv search → 候选论文
│   ├── reader.py            # 精读：--head / --brief / --section
│   ├── summarizer.py        # 总结：单篇中文摘要 + 跨论文领域报告
│   └── utils.py             # 日志 / 配置 / 去重 / DeepXiv CLI 调用封装
├── output/
│   ├── papers/              # 单篇精读产物
│   └── reports/             # 跨论文综述报告
├── data/
│   └── seen_ids.json        # 已处理论文 ID（去重）
├── logs/                    # 运行日志，按日期切分
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

# 3. 复制一份 .env 并填入你的 LLM 凭证
cp .env.example .env
#    必填：
#      ANTHROPIC_API_KEY    你的 Anthropic API key（或兼容代理的 key）
#      LLM_MODEL            模型 ID，例如 claude-haiku-4-5-20251001
#    可选：
#      ANTHROPIC_BASE_URL   如果用第三方兼容代理，写代理的 URL；直连官方可省略

# 4. （首次使用）DeepXiv 会在首次调用时自动生成匿名 token
#    并写入 ~/.env，默认 daily limit 1000。如需提升额度，邮件联系
#    tommy@chien.io。也可手动配置：
deepxiv config

# 5. 验证 CLI
deepxiv --help
deepxiv search "knowledge distillation" --limit 3 --format json
```

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
# 一次性增量运行（默认最多精读 10 篇；末尾自动开浏览器看 index.html）
python src/run.py

# 常用参数
python src/run.py --max-read 20           # 本轮多读几篇
python src/run.py --retry-failed          # 重试 failed_ids 里未 ingest 的论文
python src/run.py --skip-search           # 跳过 search，复用最新 candidates（省 API）
python src/run.py --no-llm                # 只 search + read，不调 LLM（省 token）
python src/run.py --skip-formulas         # 跳过从 arXiv 源码提取公式的阶段
python src/run.py --skip-report           # 只跑前面阶段，不出 TOP10 和 HTML
python src/run.py --no-open               # 跑完不自动开浏览器
python src/run.py --top-n 20              # TOP N 改成 20
python src/run.py --field knowledge_distillation   # 只对指定领域生成综述

# 交互式启动器（推荐）
python start.py                            # 仪表盘 + 菜单选择

# 独立运行各阶段（方便调试）
python src/reporter.py --top10            # 只重算 TOP10
python src/reporter.py --html --open      # 只重渲 HTML 并打开浏览器
python src/reporter.py --all --open       # 重算 TOP10 + HTML + 打开
python src/formula_handler.py <arxiv_id>  # 只对一篇论文提公式（arXiv 路线）
python src/arxiv_source.py <arxiv_id>     # 只下载 + 解压 arXiv 源码
```

增量规则：
- `seen_ids.json` 里的论文不会被重复精读
- `metadata_only`（DeepXiv 全文未 ingest）和 `failed` 的论文**不写 seen**，下次可重试
- 已有 `.summary.md` 的论文不重复生成摘要
- 每个领域 ≥ 2 篇可用摘要才生成综述（低于不跑）

### 单模块调试

```bash
python src/utils.py                       # utils 自检
python src/searcher.py                    # 单关键词冒烟
python src/searcher.py --all              # 全量批量检索 6 领域
python src/reader.py <arxiv_id>           # 精读单篇
python src/summarizer.py <arxiv_id>       # 单篇摘要
python src/summarizer.py --field knowledge_distillation   # 指定领域综述
```

### 导出 DeepScientist 投喂包

本项目可以作为 DeepScientist 的前置文献情报层：先完成每周文献抓取、精读、公式提取和技术路线解读，再把少量高质量材料投喂给 DeepScientist 做后续实验规划与执行。

```bash
# 1) 先跑本项目流水线，得到候选、摘要、公式和 TOP 报告
python src/run.py --no-open

# 2) 导出 DeepScientist 启动材料（默认 TOP5）
python src/deepscientist_exporter.py --top-n 5

# 可选：指定你的故障诊断 baseline 代码路径
python src/deepscientist_exporter.py --top-n 5 --baseline-path E:\codes\bearing-fault-baseline
```

导出目录：

```
output/deepscientist_bundle/
├── manifest.json             # 本次导出的元数据和文件清单
├── candidate_papers.json     # TOP 论文结构化清单
├── literature_brief.md       # 给 DeepScientist 读的文献简报
├── hypotheses.md             # 候选研究假设
└── startup_prompt.md         # 可直接作为 DeepScientist quest 启动提示的材料
```

推荐只投喂 TOP3-TOP5，而不是把 85 篇候选全部交给 DeepScientist。DeepScientist 更适合消费“少量高质量、和 baseline 强相关”的研究材料。

## 产物位置

```
data/
├── candidates_<YYYYMMDD>.json   # 每次 search 的候选集合
├── seen_ids.json                # 已精读完成的 arxiv_id（增量去重）
└── failed_ids.json              # DeepXiv 未 ingest / head 拿不到的

output/
├── papers/
│   ├── <arxiv_id>.json          # 精读产物（含 head + 策略 + 正文片段）
│   ├── <arxiv_id>.summary.md    # 单篇中文摘要
│   ├── <arxiv_id>.formulas.json # 结构化公式清单（每条含 latex/type/env/label/eq_num/上下文）
│   └── <arxiv_id>.formulas.md   # 可读公式速览（Display 每条独立章节，Inline 汇总表）
├── reports/
│   ├── <field>_<YYYYMMDD>.md    # 领域综述报告
│   └── weekly_top<N>_<YYYYMMDD>.md  # 本周 TOP 合并报告
└── html/
    ├── index.html               # 总览索引（带 MathJax 渲染 + 📐 公式 链接）
    ├── reports/*.html
    └── papers/*.summary.html + *.formulas.html

data/
├── arxiv_src/<arxiv_id>/        # 下载并解压的 arXiv LaTeX 源码（本地缓存）
├── candidates_<YYYYMMDD>.json   # 每次 search 的候选集合
├── openalex_cache.json          # OpenAlex 查询 cache
├── seen_ids.json / failed_ids.json
```

### 综合评分

`src/reporter.py::composite_score()` 归一化到 0-100：

```
composite = 50% × 启发式相关性（6 领域 × 1-5 分总和 / 30）
          + 30% × DeepXiv search score（截断到 10 归一化）
          + 20% × log1p(citation_count) / log(1000)
```

权重常量在 `reporter.py` 顶部 `W_RELEVANCE / W_DEEPXIV / W_CITATION`，直接改即可。

## 路线图

- [x] **步骤 1**：项目骨架 + conda 环境 + DeepXiv CLI 验证
- [x] **步骤 2**：接入 DeepXiv search / paper，批量检索 + 全文策略化读
- [x] **步骤 3**：接入 LLM（通过 Anthropic Messages API，模型可配置），单篇中文摘要 + 跨论文领域综述
- [x] **步骤 4**：一键流水线 `run.py` + 交互式启动器 `start.py`（增量 / 失败重试）
- [x] **步骤 5**：综合评分 + 本周 TOP10 合并报告 + HTML 渲染 + 自动开浏览器
- [x] **步骤 6a (Phase 1)**：arXiv 论文公式提取（下载 e-print → 解析 LaTeX → 带编号/label/上下文的结构化产物）
- [x] **步骤 7 (v1.2.0)**：跨论文综合创新分析 `synthesizer.py` + 公式交叉引用 + 融合公式输出 + 历史论文池 + LLM Provider 预设系统 + 交互式启动器 `start.py`
- [ ] **步骤 8 (Phase 2)**：PDF / HTML 论文公式提取（部署到校园网电脑后启用，见下方部署指南）

## Phase 2 部署指南（仅在可访问 IEEE/ScienceDirect 等数据库的校园网电脑上启用）

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
   - MathJax 脚本生成的 `<script type="math/tex">` 直接提 innerText
3. **登录态**：在校园网 IP 段内直接访问通常即可；如需 EZproxy，把 proxy URL 写进 requests session。

详细设计参考每个模块文件顶部的 docstring。

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
