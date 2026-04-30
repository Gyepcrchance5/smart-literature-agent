# AGENTS.md — AI Agent 接手指南

> 这是给 AI agent（Claude Code / Cursor / Copilot / Gemini 等）的**项目接手指南**。
> 本人维护者换电脑 / 换 agent 时，agent 首先读此文件即可快速接手项目。
> 面向人的介绍在 [README.md](README.md)，版本历史在 [CHANGELOG.md](CHANGELOG.md)。

## 一句话

**smart-literature-agent**：一个面向个人科研的**自动化文献跟踪智能体**。每周自动抓 arXiv 新论文 → OpenAlex 质量信号增强 → LLM 生成中文摘要与领域综述 → 提取论文 LaTeX 公式 → 综合评分输出本周 TOP10 合并报告 + 带 MathJax 的静态 HTML 索引 → 每周一 08:07 定时自动运行并自动开浏览器。

## 当前状态（以最新 git log 为准；本文件在 release 时同步更新）

- **最新版本**：v1.1.0（2026-04-30）
- **Phase 1 (local dev machine)**：✅ 完成
  - 数据源：arXiv（via DeepXiv）
  - 公式提取：arXiv e-print LaTeX 源码 → equation/align/eqnarray 等环境解析
- **Phase 2 (campus-network machine)**：⏳ 待做（`src/pdf_handler.py` 和 `src/html_handler.py` 是 stub）
  - 数据源扩展：IEEE Xplore / ScienceDirect 的 PDF；期刊网站 HTML
  - 需要校园网 IP 访问 + 本地装 PDF 解析器（推荐 MinerU）

## 架构（5 阶段 pipeline）

```
[1/5 search]    searcher.py  → DeepXiv 按关键词抓 6 领域 × N 关键词的新论文
                               → 候选落盘 data/candidates_<YYYYMMDD>.json
                               → 同时自动调 enricher 做 OpenAlex 增强

[2/5 read]      reader.py    → 按 token_count 4 档策略精读：
                               raw (≤8k) / selected (8k-20k, 选关键 section)
                               / preview (>20k, ~10k 字符预览)
                               / metadata_only (DeepXiv 未 ingest 全文)
                               → 产物 output/papers/<id>.json
                               → 更新 seen_ids.json（metadata_only/failed 不写 seen，以便重试）

[3/5 llm]       summarizer.py → 对每篇新 read 的论文生成中文单篇摘要（output/papers/<id>.summary.md）
                               → 每领域 ≥2 篇摘要时生成跨论文综述（output/reports/<field>_<date>.md）
                               → 默认模型 xiaomi/mimo-v2.5-pro（可通过 LLM_MODEL env 覆盖）

[4/5 formulas]  formula_handler.py + arxiv_source.py
                             → 对每篇精读论文下载 arXiv e-print tarball → 解压 → 找主 .tex
                               → 解析 5 种数学环境 + 4 种定界符 → 带 eq_num / label / 上下文
                               → 产物 output/papers/<id>.formulas.{json,md}

[5/5 report]    reporter.py  → 4 维综合评分（relevance 45% + DeepXiv 25% + venue 20% + citation 10%）
                               → 本周 TOP10 合并报告（output/reports/weekly_top<N>_<date>.md）
                               → 所有 md 转 HTML + MathJax 渲染 + 生成 output/html/index.html
                               → 默认自动在浏览器打开 index.html
```

关键模块清单（src/）：

| 模块 | 职责 | 核心函数 |
|---|---|---|
| `utils.py` | 日志 / 配置 / seen_ids / `run_deepxiv` subprocess 封装 / `get_anthropic_config` 凭证读取 | |
| `searcher.py` | 批量检索 + 本轮去重 + 分数阈值 + OpenAlex 增强入口 | `search_all_fields()` |
| `reader.py` | DeepXiv paper wrapper + 4 档读策略 + failed_ids 管理 | `full_read()` |
| `summarizer.py` | Anthropic SDK 封装 + 中文 prompt + 单篇摘要 + 领域综述 + 启发式相关性打分 | `summarize_single_paper()` / `generate_field_report()` / `score_relevance()` |
| `enricher.py` | OpenAlex 客户端 + DOI 查询 + title search fallback + published-sibling 查询 + venue prestige 计分 | `enrich_all()` / `venue_prestige_score()` |
| `arxiv_source.py` | arXiv e-print 下载 + tar/gz/plain 解压 + 主 .tex 定位 + `\input/\include` 递归内联 | `fetch_latex()` |
| `formula_handler.py` | LaTeX 数学环境 + 定界符解析 + 上下文抽取 + 路由器（分派 arXiv/PDF/HTML） | `extract()` / `extract_from_latex()` / `save_formulas()` |
| `pdf_handler.py` | [Phase 2 stub] PDF → 公式提取（计划接 MinerU） | `extract_from_pdf()` |
| `html_handler.py` | [Phase 2 stub] HTML → 公式提取（计划写期刊 adapter） | `extract_from_html()` |
| `reporter.py` | composite_score 4 维评分 + weekly TOP10 合并 + HTML 渲染 + MathJax 注入 + 浏览器自动打开 | `composite_score()` / `generate_weekly_top10()` / `render_html_all()` |
| `run.py` | Pipeline orchestrator + CLI flags + schedule 常驻每周一 08:07 | `pipeline_run()` / `run_daemon()` |

## 核心设计决策（& 为什么）

### 配置 / 凭证

- **LLM 凭证从环境变量读，代码不含明文 key**（`utils.get_anthropic_config`）。env 里没有时从 `~/.claude/settings.json` 的 `env` 段兜底 —— 兼容 cc-switch 之类配置工具对 Claude Code 进程注入 env 的场景。
- **DEFAULT_MODEL 从 `os.environ.get("LLM_MODEL", "xiaomi/mimo-v2.5-pro")` 读**。默认值是一个私有代理的路由标识，fork 本仓库后必须用自己的模型 ID 覆盖（`.env` 里设 `LLM_MODEL=claude-haiku-4-5-20251001`）。
- **Agent 当你需要调 LLM 时，直接用 `anthropic.Anthropic(api_key=..., base_url=...)`**，不要硬编码任何 URL。`base_url` 是可选的，官方直连时空即可。

### 评分公式（4 维，权重在 `reporter.py` 顶部）

```
composite = 0.45 × relevance (6 领域启发式关键词命中)
          + 0.25 × deepxiv_score (DeepXiv search 返回的相关性分)
          + 0.20 × venue_prestige (OpenAlex venue h-index)
          + 0.10 × citation (OpenAlex cited_by_count log 归一化)
```

**为什么选这 4 维**：
- relevance 权重最高 —— 本项目核心是"跟踪你关心的 6 个领域"
- venue 信号为了识别"这论文已被 NeurIPS/CVPR 接收" vs "还是预印本"
- citation 权重压到 10% —— **新论文普遍 0 引用**，高权重会把新论文全压低

### `venue_prestige_score` 打分梯度（`enricher.py`）

- 未命中 OpenAlex → 0
- `repository`（纯 arXiv 预印本）→ 10（基线，避免跟"未命中"混）
- `conference` / `journal` 有 h_index → `min(h, 250) / 250 × 100`，但不低于 50
- `conference` / `journal` 无 h_index → 50（**OpenAlex 数据稀疏，大多数 venue 没填 h_index**）
- **不能**把 "无 h_index 的 NeurIPS" 和 "repository" 都给 10 分——那就失去区分度了

### OpenAlex 查询策略（三级 fallback）

见 `enricher.py:enrich_one`：

1. **DOI 查询**：`https://api.openalex.org/works/doi:10.48550/arXiv.<arxiv_id>`（2022 年后 arXiv 论文都有这个 DOI）
2. **Title search fallback**：DOI 查不到时（2022 年前老论文），用标题搜 + SequenceMatcher ≥ 0.8 相似度校验
3. **Published-sibling lookup**：**拿到的 primary work 如果是 `repository`（只是 arXiv 预印本）**，额外用标题搜找 `type=conference/journal` 的**另一条独立 work**（OpenAlex 把 arXiv 预印本和 NeurIPS 发表版当成两条 works，DOI 不同）

### reader.full_read 的 4 档策略

```
token_count == 0      → metadata_only （DeepXiv 只有元数据没全文，不写 seen，下次可重试）
0 < tc ≤ 8000         → raw           （全文 markdown）
8000 < tc ≤ 20000     → selected      （匹配 Introduction/Method/Experiments/Conclusion 关键章节）
tc > 20000            → preview       （~10k 字符预览）
任一下载子步骤失败    → 降级到下一档  （raw 失败 → preview → metadata_only）
```

### 增量语义

- **`seen_ids.json`** 里的 arxiv_id 表示"已完整精读 + 写入 json"。下次 pipeline 不再 read。
- **`failed_ids.json`** 里的 arxiv_id 表示"DeepXiv 连 head 都拿不到"（404 / 论文不存在 / DeepXiv 未收录）。默认不重试；`--retry-failed` 开启时重试。
- **`.summary.md` / `.formulas.json` 是否存在** 作为"是否需要生成 LLM 摘要 / 公式"的增量标记。
- **`data/openalex_cache.json`** 缓存 OpenAlex 查询结果，避免重复查。
- **`data/arxiv_src/<id>/`** 缓存 arXiv LaTeX 源码，避免重复下载。

## 已知的坑 + 解决方案（Gotchas）

### 1. Windows + 中文系统 + 子进程编码

**症状**：调用 `deepxiv` CLI 子进程时 `UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f3e5'`。

**原因**：Windows 中文系统 Python stdout 默认 GBK，DeepXiv CLI 输出含 emoji。

**解决**：`utils.run_deepxiv` 里给子进程显式设置：

```python
env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
subprocess.run([...], env=env, encoding="utf-8", ...)
```

### 2. subprocess 找不到 `deepxiv.exe`

**症状**：`FileNotFoundError: [WinError 2] 系统找不到指定的文件`。

**原因**：Python 脚本直接跑（没 activate conda env）时，`deepxiv` 不在 PATH。

**解决**：`utils._resolve_deepxiv_cli()` 用 `sys.executable` 推导出同 env 的 Scripts 目录，用绝对路径。

### 3. Reasoning model 的 `output_tokens` 包含 thinking

**症状**：调 `xiaomi/mimo-v2.5-pro` 摘要时，`max_tokens=1500` 会导致 summary 在中间位置被截断（"基于人工神经..."）。

**原因**：`output_tokens` usage 计数包含 reasoning 过程的 tokens，不只是最终输出。

**解决**：`summarizer.py` 顶部常量 `SINGLE_SUMMARY_MAX_TOKENS = 3000`、`FIELD_REPORT_MAX_TOKENS = 6000`，给 reasoning 留足够 budget。Agent 如果换其他非 reasoning model，可以调小。

### 4. DeepXiv search 命中 ≠ 有全文

**症状**：search 返回 85 篇候选，其中 3 篇 `paper --head` 返回 `token_count=0` 或 `--raw`/`--preview` 全 404 (`Resource not found: https://data.rag.ac.cn/arxiv/`)。

**原因**：DeepXiv 的 search 索引和 paper 全文 ingest 是两条管道，搜到不代表能读。

**解决**：`reader.full_read` 的 4 档降级 + `failed_ids.json` + `--retry-failed` flag。新论文当天 search 到但 token_count=0，过几天再跑 pipeline 可能 DeepXiv 已经 ingest。

### 5. OpenAlex 把 arXiv 预印本和会议版当作两条独立 works

**症状**：查 ViT 的 arXiv DOI `10.48550/arXiv.2010.11929`，拿到 `primary_location.source.type = "repository"`，完全看不到 ICLR 信息，以为是普通预印本。

**原因**：OpenAlex 里每个独立 DOI 对应一个 work，arXiv preprint 和 ICLR proceedings 是不同 DOI → 两条 works，不 merge。

**解决**：`enricher._find_published_sibling` 额外用 title search 找 `type=conference/journal` 的 sibling work，把 venue 信息合到 signals 里（citation 仍用主 work 的，通常更高）。

### 6. OpenAlex 的 `summary_stats.h_index` 普遍为 null

**原因**：OpenAlex 数据稀疏，即使是 NeurIPS、ICLR、Nature 这种顶刊顶会，`source.summary_stats.h_index` 字段很多时候缺失。

**解决**：`venue_prestige_score` 给 conference/journal **基线 50 分**，有 h_index 才往上加。不能直接用 0 或 10 —— 会把"发表在 NeurIPS 但 h_index 字段缺失"的论文和"纯 arXiv 预印本"混淆。

### 7. Python `$` 在 bash 里被当变量

**症状**：在 Bash 里写 `grep '$$...$$'` 报错 `$$: arithmetic syntax error`。

**解决**：Bash 里跑复杂 LaTeX 相关的 probe 时用独立 Python 脚本文件，不要 inline `python -c "..."`。`scripts/` 目录放这类探测脚本。

### 8. GitHub 网页拖拽上传会绕过 .gitignore

**症状**：在 GitHub 网页用 "Upload files" 拖拽目录，`.gitignore` 里的 `logs/ output/` 仍然被上传。

**原因**：网页上传是客户端把文件塞给 GitHub，不走本地 git add → .gitignore 过滤流程。

**解决**：**永远用 `git add -A` + `git commit` + `git push` 从本地推**。网页拖拽只在首次建 repo 试试，要删库重建很贵。

### 9. arXiv ID 格式

- **新格式**：`YYYY.NNNNN` 或 `YYYY.NNNN`（如 `2411.11707`、`1706.03762`）
- **旧格式**：`category/NNNNNNN`（如 `cs/0501001`）
- **版本号**：尾部可能带 `vN`（如 `2411.11707v3`）

`formula_handler.extract()` 的路由用的正则：

```python
re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", source)  # 新格式
re.fullmatch(r"[a-z\-]+/\d{7}", source)          # 旧格式
```

## 开发约定

### Git / commit

- **commit email** 用 GitHub noreply（`157470502+Gyepcrchance5@users.noreply.github.com`）而非真实邮箱
- **commit author** 用 `Gyepcrchance5`（与 GitHub handle 一致）
- **commit message** 用英文，第一行简短（≤72 字符），空行，再详细段落
- **分支**：所有工作在 `main`；版本稳定点打 annotated tag（`v1.0.0` / `v1.1.0` / ...）
- **永远不 force push main**

### 敏感信息

**不允许进入 git 的内容**（`.gitignore` 已配）：
- `.env`（真实 key）
- `logs/`（运行日志里含调用 URL）
- `output/`（个人精读产物）
- `data/candidates_*.json / seen_ids.json / failed_ids.json / openalex_cache.json / arxiv_src/`

### 提交前自检

```bash
# 1) 验证 staged 文件清单（不应含 logs/output/data/arxiv_src）
git ls-files | grep -E "(logs/|output/|candidates_|seen_ids|failed_ids|openalex_cache|arxiv_src)"
# 预期：空

# 2) 扫描敏感词
git diff --cached | grep -iE "mify|ChengRui|sk-ant-[a-zA-Z0-9]{20,}|@qq\\.com"
# 预期：只有 LICENSE 的 copyright 行可能命中 "Gyepcrchance5"，没有真实 key / QQ 邮箱 / 公司内网 URL
```

### 依赖管理

- 新增依赖 → 写进 `requirements.txt`
- 重依赖（>500 MB 或需要 GPU）→ **不要**进 Phase 1 默认 `requirements.txt`，放 Phase 2 docstring 里让用户按需安装
- 当前依赖（Phase 1）：`deepxiv-sdk`、`pyyaml`、`schedule`、`anthropic`、`markdown`、`requests`（anthropic 的传递依赖，显式引用）

## 下一步 TODO（Phase 2 + 优化）

### Phase 2：PDF 路线（`src/pdf_handler.py`）

- [ ] 装 MinerU：`pip install -U "magic-pdf[full]" --extra-index-url https://wheels.myhloli.com`
- [ ] 实装 `extract_from_pdf(pdf_path)`：
  - 调 MinerU / Marker 把 PDF → markdown（公式保留 LaTeX）
  - markdown 喂给 `formula_handler.extract_from_latex(md_text)` 复用解析器
  - 返回 `list[Formula]`
- [ ] 在 `candidates_*.json` 的 paper entry 加 `source_type: "pdf" | "arxiv" | "html"` 字段
- [ ] `formula_handler.extract(source)` 路由新增 PDF 分派
- [ ] `pipeline_run` 的 [4/5 formulas] 阶段按 source_type 分派
- [ ] 测试数据：在校园网下载 1-2 篇 IEEE PDF 做端到端验证

### Phase 2：HTML 路线（`src/html_handler.py`）

- [ ] 为每个期刊写 adapter（IEEE / ScienceDirect / Nature / Science）：
  - 识别：按 URL host 分派
  - 抓取：`requests` + BeautifulSoup；如遇 JS 渲染，用 Playwright
  - 提取：正文 + 公式节点（MathML / MathJax script）
  - MathML → LaTeX：`pip install mathml-to-latex`
- [ ] 登录态：校园网 IP 段直接通常即可；EZproxy 场景写进 requests session
- [ ] `searcher.py` 扩展 IEEE Xplore API / Elsevier Engineering Village API（需 API key）做检索入口

### V2 持续优化

- [ ] **LLM 评分升级**（score_relevance）：对 title+abstract 调一次 LLM 给对你方向的专项评分（1-10 分）。成本：每篇 +1 次 LLM 调用，85 篇 ≈ 2 万 token/周
- [ ] **定期 re-enrich**：`data/openalex_cache.json` 里 cache 了老结果，每月强制 ignore_cache 对 seen_ids 里的论文重查，吸收 OpenAlex 的 venue/citation 更新（今天的 arXiv 预印本 6 个月后被 NeurIPS 接收，venue 字段才会填上）
- [ ] **跨周趋势月报**：累积多周 candidates，产出"趋势月报 / 新人论文 / 本月热点"跨时段报告
- [ ] **作者 h-index 维度**：目前公式是 4 维，可以升到 5 维加 author_h_index（每篇额外查作者 OpenAlex，~5 次 API）
- [ ] **定制关键词**：`config/keywords.yaml` 里加一个 `user_context` 段声明"你的研究方向"，让 summarizer prompt 里的"与你方向的相关性"判断更准

### 工具化

- [ ] **MCP 化**：DeepXiv 已经有 MCP server；可以把本项目的 `search/read/summarize/formulas` 也包装成 MCP tool，让 Claude Desktop / Cursor 直接调
- [ ] **Web UI**：FastAPI + 静态 HTML index；支持在浏览器里搜 / 删 / 重跑单篇
- [ ] **桌面通知**：周一跑完后 win10toast 弹通知"本周 TOP10 已生成，点此查看"
- [ ] **Email 通知**：SMTP 发到你邮箱，含 TOP3 论文的精简摘要

## Agent 操作建议

### 当你第一次打开这个项目时

1. 读本文件（AGENTS.md）
2. 读 [CHANGELOG.md](CHANGELOG.md) 看最新版本做了什么
3. 跑 `python src/utils.py` 冒烟验证 env 正常（会输出配置加载、logger 初始化）
4. 跑 `python src/run.py --skip-search --no-llm --no-open --max-read 0` 确认 pipeline 各阶段能 dry run

### 当用户说"继续开发"时

1. 看 "下一步 TODO" 章节，按优先级建议一项
2. 动手前先 `git status` 看是否有未 commit 改动
3. 实现 → 冒烟 → commit（遵循"开发约定"里的 commit 规范）→ push

### 当用户说"我换到新电脑了"时

1. 指引 `git clone https://github.com/Gyepcrchance5/smart-literature-agent.git`
2. 按 README.md 的 "环境搭建" 走（conda env + pip install + `.env` 配置）
3. 读本文件理解项目状态
4. 问用户是否继续上次的 TODO，或有新方向

### 当用户说"上次做到哪了"时

- 看本文件的"当前状态"段
- 看 `git log --oneline -10`
- 看最新 commit 改动了哪些文件
- 看 CHANGELOG 的 `[Unreleased]` 段（如有）

### 不要做的事

- ❌ 不要把 logs/output/data 里的东西 commit 进去
- ❌ 不要在代码里硬编码任何 API key / 个人邮箱 / 真名 / 公司内网 URL
- ❌ 不要用 GitHub 网页拖拽上传（绕 .gitignore）
- ❌ 不要 force push main 或 delete tags
- ❌ 不要在没验证本地能跑通时就 push

## 快速命令参考

```bash
# 一次性跑完整 pipeline
python src/run.py

# 只跑 4/5 formulas + 5/5 report（不跑 search / read / llm）
python src/run.py --skip-search --no-llm --max-read 0

# 单篇公式提取
python src/formula_handler.py 2411.11707

# 常驻（每周一 08:07 自动）
python src/run.py --daemon

# 查看当前项目状态
git log --oneline -10
git status
```

## 联系方式 / 上下文

- **仓库**：https://github.com/Gyepcrchance5/smart-literature-agent
- **维护者**：Gyepcrchance5（GitHub handle）
- **主项目**：轴承故障诊断 R1 路线（CWRU 主 + PU 跨域 + FA-KD + 结构化剪枝），本工具为主项目服务
