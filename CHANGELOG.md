# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

（main 分支上的改动，未发版。每次打 tag 前把这里的内容 move 到新版本段）

---

## [1.2.0] — 2026-05-01

Major feature release: cross-paper synthesis, historical pool, launch dashboard, LLM provider system.

### Added

- **跨论文综合创新分析** (`src/synthesizer.py`)
  - 对 TOP 论文做 LLM 驱动的交叉对比，输出 6 章节报告：共享问题景观 / 方法矩阵 / 冲突与互补 / 模块融合创新方向 / 推荐阅读顺序 / 关键公式交叉引用
  - 公式注入：每篇送 top 3 关键公式 + LaTeX token 相似度预计算矩阵
  - 融合公式结构化输出：自动提取 `$$...$$` 块 → `synthesis_<date>.formulas.json`
  - 变量映射表：每个融合公式附带源论文变量 → 融合变量的对照表
- **历史论文池** (`src/searcher.py` new `build_historical_pool()`)
  - 一次性回溯 5 年建池 → `data/candidates_historical_pool.json`
  - 每次 pipeline 自动从中取 30%×max_read 篇高分未读论文混入阅读
  - `seen_ids` 跟踪进度，池子耗尽时可重建
- **交互式启动器** (`start.py`)
  - 仪表盘：API Key / LLM Provider / 上次运行 / 各产物统计 / 历史池状态一目了然
  - 菜单：[1] 完整流水线 [2] 快速运行 [3] 建历史池 [4] 仅重生成报告 [5] 打开报告 [6] 调整参数
  - 参数调整：实时修改 max_read / historical_ratio / top_n / auto_open
- **LLM Provider 预设系统** (`config/keywords.yaml` new `providers:`)
  - 支持 DeepSeek / Anthropic Official / MiniMax 三家
  - `.env` 中设 `LLM_PROVIDER=deepseek` 一键切换
  - `start.py` 菜单可视化切换 Provider 和 Model，自动写回 `.env`
- **`get_llm_config()`** (`src/utils.py`) — 统一 LLM 配置入口，兼容旧 `.env` 格式
- **`start.py`** — 项目级入口点

### Changed

- **移除定时调度**：去掉 `run_daemon()` 和 `schedule` 依赖，主动权完全交给用户（`start.py` 手动触发）
- **Pipeline 从 5 阶段升级为 6 阶段**：`[1] search → [2] read → [3] formulas → [4] summarize → [5] report → [6] synthesis`
- **HTML 公式渲染修复** (`reporter.py`)：新增 `_protect_math()` / `_restore_math()`，防止 markdown 转换器破坏 LaTeX 公式
- **Anthropic 客户端包装** (`summarizer.py`)：改用 `get_llm_config()` 读 provider 预设，auth_mode 适配 Bearer 认证
- **`sitecustomize.py`**：移除硬编码 MiniMax 默认值注入
- **`.env.example`**：重写为多 Provider 通用模板
- **`keywords.yaml`**：新增 `search_config.historical_*` 和 `providers` 段
- **`requirements.txt`**：显式声明 `requests>=2.31.0`，移除 `schedule`

### Fixed

- `run.py` 阶段注释编号不一致（2/5 → 2/6 等）
- `synthesizer.py` 未使用 import 和死代码
- 日志 "剩余 N 篇未读" 含义错误

---

## [1.1.1] — 2026-04-30

Small integration release: make smart-literature-agent a feeder for DeepScientist.

### Added

- **DeepScientist bundle export** (`src/deepscientist_exporter.py`)
  - Reads the latest `data/candidates_<date>.json`, generated `.summary.md`, `.formulas.md`, and composite scores.
  - Exports a curated handoff package under `output/deepscientist_bundle/`.
  - Generated files:
    - `manifest.json`
    - `candidate_papers.json`
    - `literature_brief.md`
    - `hypotheses.md`
    - `startup_prompt.md`
  - Supports `--top-n`, `--out-dir`, `--research-context`, and `--baseline-path`.
- **DeepScientist workflow docs** in `README.md`
  - Documents the intended role split:
    `smart-literature-agent` = literature ingestion / filtering / technical interpretation,
    DeepScientist = baseline-driven research execution / experiments / findings.
  - Adds a two-command workflow:
    `python src/run.py --no-open`
    then `python src/deepscientist_exporter.py --top-n 5`.

### Changed

- **Single-paper summaries are now DeepScientist-oriented**
  - The prompt reads selected formula context from `.formulas.json`.
  - Output now emphasizes method decomposition, key formula explanation, transfer mapping, and a concrete route for bearing fault diagnosis + compression/distillation + edge deployment.
- **Pipeline order changed**
  - Formula extraction now runs before the LLM stage, so summaries can explain formulas.
- **LLM token budgets raised for MiniMax-M2.7**
  - `SINGLE_SUMMARY_MAX_TOKENS = 12000`
  - `FIELD_REPORT_MAX_TOKENS = 16000`
  - `MAX_CONTEXT_CHARS = 100000`
  - `MAX_FORMULA_CONTEXT_CHARS = 16000`
- **Run CLI**
  - Added `python src/run.py --regen-summary` to force regeneration of existing `.summary.md` files after prompt upgrades.

### Validation

- Re-generated `2604.22276.summary.md` with the new method/formula/transfer-route template.
- Verified dry run: `python src/run.py --skip-search --no-llm --no-open --max-read 0`.
- Verified exporter: `python src/deepscientist_exporter.py --top-n 3`.

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

[Unreleased]: https://github.com/Gyepcrchance5/smart-literature-agent/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/Gyepcrchance5/smart-literature-agent/compare/v1.1.1...v1.2.0
[1.1.1]: https://github.com/Gyepcrchance5/smart-literature-agent/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Gyepcrchance5/smart-literature-agent/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Gyepcrchance5/smart-literature-agent/releases/tag/v1.0.0
