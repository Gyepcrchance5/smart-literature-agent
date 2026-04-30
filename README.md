# smart-literature-agent

> **一句话**：基于 DeepXiv SDK 与 Anthropic Messages API 的自动化科研文献智能体 —— 每周自动抓取 6 大研究方向的 arXiv 新论文、按 token 预算分级精读、用 LLM 产出中文摘要与跨论文领域综述，综合评分输出本周 TOP10 合并报告 + 静态 HTML 索引，每周一 08:07 定时跑完自动在浏览器弹出。

## 项目简介

这是一个**面向个人科研的文献跟踪工具**，聚焦六个研究方向：模型压缩与轻量化、知识蒸馏、故障诊断、深度学习方法、边缘部署与推理优化、可解释性。

### 核心能力

1. **自动检索**：6 领域 × 40+ 关键词，调用 DeepXiv 遍历最近 7 天的新论文，按 DeepXiv score 过滤、去重后落盘候选清单。
2. **智能精读**：按 `token_count` 分四档策略 —— `raw`（全文）/ `selected`（精选 Introduction/Method/Experiments/Conclusion）/ `preview`（10k 字符概要）/ `metadata_only`（DeepXiv 尚未 ingest 时的降级），对超 80k token 的长文也能优雅处理。
3. **中文摘要 + 领域综述**：通过兼容 Anthropic Messages API 的 LLM 产出**结构化中文单篇摘要**（研究问题 / 核心方法 / 关键实验 / 六领域相关性 / 局限与启发）与**跨论文领域综述**（问题聚类 / 主流技术路线 / 常用数据集 / 开放问题 / 对研究者的具体迁移建议）。
4. **综合评分 + 本周 TOP10**：`composite = 50% × 启发式相关性 + 30% × DeepXiv score + 20% × 引用数`（归一到 0-100），选出本周 TOP10 生成合并 markdown 报告。
5. **HTML 渲染 + 自动打开浏览器**：所有 markdown 产物一键转 GitHub 风格 HTML，生成总览 `index.html`（本周 TOP / 领域综述 / 单篇摘要三段，按综合分降序 + score badge），跑完自动弹出。
6. **增量 + 失败重试 + 定时调度**：`seen_ids` 增量去重，`metadata_only` 和 `failed` 不写 seen 以便下次重试；`schedule` 库每周一 08:07 常驻自动跑。

### 规模数据（实测）

| 项目 | 值 |
|---|---|
| 单次完整流水线耗时 | **7 分钟**（85 候选 / 10 精读 / 11 摘要 / 4 领域综述 / 1 TOP10 + HTML） |
| Token 消耗 | **~7 万 Token**（~49k input + 22k output） |
| 覆盖关键词 | 42 个（6 领域） |
| 输出文件类型 | 5 种：候选清单 JSON / 精读 JSON / 单篇 summary markdown / 领域综述 / HTML 索引页 |
| 运行模式 | 常驻 + 定时（`schedule` + `os.startfile` 开浏览器），全程无人值守 |

### 设计亮点

- **凭证零明文落地**：LLM key / DeepXiv token 全部从环境变量或用户级配置读取，代码仓库里没有一个字节的 key。
- **支持第三方兼容代理**：通过 `ANTHROPIC_BASE_URL` + `LLM_MODEL` 即可切换直连官方或走兼容代理（含 cc-switch 等配置工具），不改一行代码。
- **失败优雅降级**：DeepXiv 的 search 和 paper ingest 是两条管道，搜到不代表能读全文；四档精读策略 + `failed_ids` + 重试 flag 完整覆盖。
- **结构化产物**：所有中间产物都是 markdown / JSON，方便下游 LLM Agent 二次消费（后续可 MCP 化）。

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
python src/run.py --skip-report           # 只跑 search/read/summary，不出 TOP10 和 HTML
python src/run.py --no-open               # 跑完不自动开浏览器
python src/run.py --top-n 20              # TOP N 改成 20
python src/run.py --field knowledge_distillation   # 只对指定领域生成综述

# 常驻定时（每周一 08:07 自动运行）
python src/run.py --daemon                # 启动时先跑一次
python src/run.py --daemon --no-initial-run   # 只等调度触发，不立即跑

# 独立运行各阶段（方便调试）
python src/reporter.py --top10            # 只重算 TOP10
python src/reporter.py --html --open      # 只重渲 HTML 并打开浏览器
python src/reporter.py --all --open       # 重算 TOP10 + HTML + 打开
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

## 产物位置

```
data/
├── candidates_<YYYYMMDD>.json   # 每次 search 的候选集合
├── seen_ids.json                # 已精读完成的 arxiv_id（增量去重）
└── failed_ids.json              # DeepXiv 未 ingest / head 拿不到的

output/
├── papers/
│   ├── <arxiv_id>.json          # 精读产物（含 head + 策略 + 正文片段）
│   └── <arxiv_id>.summary.md    # 单篇中文摘要
├── reports/
│   ├── <field>_<YYYYMMDD>.md    # 领域综述报告
│   └── weekly_top<N>_<YYYYMMDD>.md  # 本周 TOP 合并报告
└── html/
    ├── index.html               # 总览索引（本周 TOP / 领域综述 / 单篇摘要，按综合分排序）
    ├── reports/*.html
    └── papers/*.summary.html
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
- [x] **步骤 4**：一键流水线 `run.py` + `schedule` 库定时（增量 / 失败重试）
- [x] **步骤 5**：综合评分 + 本周 TOP10 合并报告 + HTML 渲染 + 自动开浏览器

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
