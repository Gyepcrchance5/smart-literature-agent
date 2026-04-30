# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

（main 分支上的改动，未发版。每次打 tag 前把这里的内容 move 到新版本段）

---

## [1.1.0] — 2026-04-30

第一次 feature release。围绕"让综合评分更准 + 公式可复用"两个主题。

### Added

- **OpenAlex 质量信号增强**（新模块 `src/enricher.py`）
  - 免费免 key 的学术数据库，对每篇候选补充 venue 类型 / h-index / 引用数 / 作者
  - 三级查询策略：DOI 查 → title search fallback → published-sibling lookup（解决 OpenAlex 把 arXiv preprint 和 NeurIPS/CVPR 发表版当作两条独立 works 的问题）
  - 本地 cache：`data/openalex_cache.json`
  - Polite pool 支持：`OPENALEX_MAILTO` 环境变量
- **四维综合评分公式** (`src/reporter.py::composite_score`)
  - `composite = 0.45 × relevance + 0.25 × deepxiv + 0.20 × venue + 0.10 × citation`
  - `venue_prestige_score`：repository = 10，conference/journal 有 h_index 按梯度，无 h_index 基线 50（OpenAlex 的 h_index 字段普遍缺失，不能给 0）
  - TOP10 表格新增 Venue 列
- **arXiv LaTeX 源码下载**（新模块 `src/arxiv_source.py`）
  - 下载 `https://arxiv.org/e-print/<id>` tarball
  - 自动识别 tar.gz / gzip / plain 三种格式解压
  - 主 .tex 定位：优先含 `\documentclass` 且含 `\begin{document}` 的
  - 递归内联 `\input` / `\include`
  - 本地缓存：`data/arxiv_src/<id>/`
- **公式提取**（新模块 `src/formula_handler.py`）
  - 解析 5 种数学环境：`equation` / `align` / `eqnarray` / `gather` / `multline`（各带 `*` 无编号变体）
  - 解析 4 种定界符：`\[...\]` / `$$...$$` / `\(...\)` / `$...$`
  - 提取 `\label{eq:xxx}` 和自然序编号
  - 抽取前后各 150 字上下文
  - 产物：`output/papers/<id>.formulas.json` + `.formulas.md`
  - 路由器 `extract(source)`：根据 source 类型分派到 arXiv / PDF / HTML
- **Phase 2 stubs**（`src/pdf_handler.py` + `src/html_handler.py`）
  - 预留 `extract_from_pdf` / `extract_from_html` interface
  - docstring 里写明 Phase 2 实装指引（推荐 MinerU / BeautifulSoup + mathml-to-latex）
  - 当前抛 `NotImplementedError`
- **MathJax 渲染** (`src/reporter.py`)
  - 所有 HTML 产物 `<head>` 注入 MathJax 3 CDN
  - `index.html` 单篇摘要清单每篇后加 "📐 公式 X/Y" 链接
- **Pipeline 第 4/5 阶段：formulas**（`src/run.py`）
  - 对所有已精读且无 `.formulas.json` 的论文自动提取公式
  - 增量语义：以 `.formulas.json` 是否存在判断
  - 新 flag `--skip-formulas`
- **开发文档**
  - 新增 `AGENTS.md`：AI agent 接手指南（架构 / 设计决策 / 踩坑 / TODO / 操作约定）
  - 新增 `CHANGELOG.md`：本文件
  - `README.md` 核心能力从 7 条升到 8 条，新增 Phase 2 部署指南段
- **CLI 命令**
  - `python src/arxiv_source.py <arxiv_id>`：单篇下载 + 主 .tex 定位
  - `python src/formula_handler.py <arxiv_id>`：单篇公式提取
  - `python src/run.py --skip-formulas`：跳过公式阶段

### Changed

- **Pipeline 从 3 阶段升到 5 阶段**：`[1/5 search] [2/5 read] [3/5 llm] [4/5 formulas] [5/5 report]`
- **`SINGLE_SUMMARY_MAX_TOKENS` 从 1500 升到 3000**、`FIELD_REPORT_MAX_TOKENS` 升到 6000 — reasoning model 的 thinking 过程占用 `output_tokens`，1500 会在摘要生成到一半时截断
- **`LLM_MODEL` 改为从 env 变量读**（`os.environ.get("LLM_MODEL", "xiaomi/mimo-v2.5-pro")`）—— 默认值保留作为示例
- **`utils.py`/`requirements.txt` 注释中性化** — 把具体提到 "cc-switch" 的措辞改成"某些配置工具（例如 cc-switch）"
- **`README.md` 顶部加入"一句话 / 项目简介 / 核心能力 / 规模数据 / 设计亮点"** 作为项目阶段性总结
- **`.gitignore` 扩展**：加 `data/arxiv_src/`、`data/openalex_cache.json`
- **`.env.example` 扩展**：加 `OPENALEX_MAILTO` 说明

### Fixed

- **OpenAlex 对老论文的 venue 信号丢失** — `_find_published_sibling` 用 title search 找 conference/journal 版 sibling work 补全 venue
- **`venue_h_index` 字段 null 时计分过低** — 改为 conference/journal 无 h_index 时基线 50 分，不至于跟 repository 的 10 分混淆

### Dependencies

- 新增：`markdown>=3.4`
- 已有：`deepxiv-sdk>=0.2.5`、`pyyaml>=6.0`、`schedule>=1.2.0`、`anthropic>=0.39.0`
- Phase 2 待装（不在默认 requirements.txt 里）：`magic-pdf[full]`（MinerU） / `beautifulsoup4` / `mathml-to-latex`

### Validation

- 14 篇候选全部 OpenAlex 增强成功（100% hit rate；均为新预印本所以 venue_type=repository）
- 8 篇经典老论文（ViT/Stable Diffusion/ResNet/Hinton KD/LLaMA/DPO/DDPM/BERT）venue 信号验证：5/8 拿到 conference/journal tag，其余因 OpenAlex 数据稀疏保持 repository
- 混合候选池（6 经典 + 85 新）综合评分排名：**TOP 6 全部为经典论文**（综合分 30-40），新预印本 TOP 7-15（综合分 24-28）— 区分度明确
- 14 篇候选公式提取：14/14 成功、0 失败、39 秒总耗时
- 产物规模示例：FedCoLLM (6 display + 82 inline，6 带 label)、2604.22588 (87 display + 228 inline)、2604.22577 (0 display + 27 inline — 工程论文无核心公式)

---

## [1.0.0] — 2026-04-30

Initial release. MVP 5 步路线图完成（骨架、检索、精读、摘要/综述、流水线/调度、综合评分/TOP10/HTML）。

### Added

- 项目骨架：`config/keywords.yaml`（6 领域 × 40+ 关键词）、`src/` 6 个模块（utils/searcher/reader/summarizer/reporter/run）、`data/` 目录
- conda 环境：`smart-lit` (Python 3.11.15)
- DeepXiv SDK 封装 (`utils.run_deepxiv`)：subprocess 自动找同 env 的 deepxiv.exe、强制 UTF-8 子进程编码
- 批量检索 `searcher.search_all_fields`：6 领域 × 40+ 关键词 × lookback_days，自动去重 + 分数阈值过滤
- 智能精读 `reader.full_read`：4 档 token 预算策略（raw / selected / preview / metadata_only），`failed_ids` 管理
- LLM 摘要与领域综述 `summarizer.summarize_single_paper` / `generate_field_report`：Anthropic Messages API，结构化中文 prompt
- 一键流水线 `run.py`：3 阶段（search → read → llm）+ 增量（seen_ids / failed_ids）+ `schedule` 每周一 08:07 常驻
- 三维综合评分 + TOP10 合并报告 + HTML 索引 + 自动打开浏览器
- MIT License、.gitignore、.env.example

### Validation

- 实测：85 篇候选一周 / 10 篇精读 + 11 篇摘要 + 4 份领域综述 + 1 份 TOP10 / 7 分钟 / ~70k tokens
- Schedule 下次触发时间验证：`2026-05-04 08:07:00`（周一）

### Known Limitations (addressed in v1.1.0)

- 综合评分只用 relevance / deepxiv_score / citation 三维，新论文引用 0 时区分度不足 → v1.1.0 加 venue_prestige
- 公式未提取 → v1.1.0 Phase 1
- 对"老论文/发表论文"和"新预印本"无区分机制 → v1.1.0 venue_prestige + published-sibling lookup

---

[Unreleased]: https://github.com/Gyepcrchance5/smart-literature-agent/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Gyepcrchance5/smart-literature-agent/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Gyepcrchance5/smart-literature-agent/releases/tag/v1.0.0
