# AGENTS.md — AI Agent 接手指南

> 这是给 AI agent（Claude Code / Cursor / Copilot / Gemini 等）的**项目接手指南**。
> 本人维护者换电脑 / 换 agent 时，agent 首先读此文件即可快速接手项目。
> 面向人的介绍在 [README.md](README.md)，版本历史在 [CHANGELOG.md](CHANGELOG.md)。

## 当前开发主线（优先于下方历史 TODO）

后续开发按 [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) 执行：A 基线与配置 → B LangGraph/恢复 →
C 证据与记忆 → D 工作台 → E 协作对照 → F 发布展示。每轮更新任务状态、验收证据和下一项。
当前下一项为 A1；真实调用遇到凭证阻塞时继续可独立完成的离线任务，不默认回到 PDF/HTML 历史 TODO。

## 一句话

## 当前新增：质量审查与研究会话

- Agent `status` 是回答执行状态，`task_status` 才是任务结果；漏步骤、缺证据和失败答案不能进入已验证记忆。
- 历史 trace 审计重建计划完成度；无样本比率返回 `null`，展示时标 N/A 并附样本数。
- 科研项目扫描支持 `work/mainline/code/core`；读取正式配置声明，区分类定义与实际启用模块，不递归扫描数据/run/归档。
- `scripts/acceptance_probe.py --out output/acceptance/local --project-path <path>` 提供本地只读验收；项目内容不得混入 `--live` 模式。

- `config/research_context.example.yaml` 是研究问题上下文模板；复制为本地 `research_context.yaml` 后填写当前瓶颈、基线和约束。
- `quality_gate.py` 对已有精读产物做本地、可解释的质量初筛，不把研究上下文默认发送给外部 LLM。
- 每篇完成摘要的论文会新增 `output/papers/<id>.evidence.json`，记录质量等级、缺失证据、迁移性分数和产物来源。
- `python src/run.py --quality-audit` 可只审查已有论文库，不调用 LLM；可配合 `--research-context <path>` 使用当前研究上下文。

**smart-literature-agent**：一个面向个人科研的**Agent 驱动文献跟踪工具**。通过 `python src/run.py` 执行，抓取 arXiv 新论文 → OpenAlex 质量信号增强 → LLM 生成中文摘要 + 结构化 enrichment JSON → 可选公式提取 → Markdown 报告和 Agent 索引；附带统一的 Planning / Tool Use / Memory / Reflection Runtime。

## 当前状态（以最新 git log 为准；本文件在 release 时同步更新）

- **最新版本**：v1.2.0（2026-05-01）
- **Phase 1 (local dev machine)**：✅ 完成
  - 数据源：arXiv（via DeepXiv）
  - 公式提取：arXiv e-print LaTeX 源码 → equation/align/eqnarray 等环境解析
- **PDF / HTML**：⏸ 后置，不属于当前核心闭环
  - 数据源扩展：IEEE Xplore / ScienceDirect 的 PDF；期刊网站 HTML
  - 需要校园网 IP 访问 + 本地装 PDF 解析器（推荐 MinerU）

## 架构（6 阶段 pipeline）

```
[1/6 search]    searcher.py  → DeepXiv 按关键词抓 6 领域 × N 关键词的新论文
                               → 候选落盘 data/candidates_<YYYYMMDD>.json
                               → 同时自动调 enricher 做 OpenAlex 增强

[2/6 read]      reader.py    → 按 token_count 4 档策略精读：
                               raw (≤8k) / selected (8k-20k, 选关键 section)
                               / preview (>20k, ~10k 字符预览)
                               / metadata_only (DeepXiv 未 ingest 全文)
                               → 产物 data/papers/<id>.json
                               → 更新 seen_ids.json（metadata_only/failed 不写 seen，以便重试）

[3/6 formulas]  formula_handler.py + arxiv_source.py
                             → 对每篇精读论文下载 arXiv e-print tarball → 解压 → 找主 .tex
                               → 解析 5 种数学环境 + 4 种定界符 → 带 eq_num / label / 上下文
                               → 产物 output/papers/<id>.formulas.json

[4/6 summarize] summarizer.py → 对每篇新 read 的论文生成中文单篇摘要 + 结构化 enrichment JSON
                               → 产物 output/papers/<id>.summary.md + <id>.enrichment.json
                               → 每领域 ≥2 篇摘要时生成跨论文综述（output/reports/<field>_<date>.md）

[5/6 report]    reporter.py  → 4 维综合评分（relevance 45% + DeepXiv 25% + venue 20% + citation 10%）
                               → 本周 TOP10 Markdown 合并报告 + output/index.json

[6/6 synthesis] synthesizer.py → 对 TOP 论文做跨论文交叉对比：公式交叉引用 / 模块融合创新方向
```

**附加模块**（非 pipeline 内，独立运行）：
- `qa_agent.py` — 文献知识库 RAG 问答（`python src/run.py --qa`）
- `project_analyzer.py` — 科研项目代码匹配分析（`python src/run.py --project-path <路径>`）

关键模块清单（src/）：

| 模块 | 职责 | 核心函数 |
|---|---|---|
| `utils.py` | 日志 / 配置 / seen_ids / `run_deepxiv` subprocess 封装 / `get_llm_config` 凭证读取（自动同步 Claude Code） | `get_llm_config()` / `get_anthropic_config()` |
| `searcher.py` | 批量检索 + 本轮去重 + 分数阈值 + OpenAlex 增强入口 + 历史论文池 | `search_all_fields()` / `build_historical_pool()` |
| `reader.py` | DeepXiv paper wrapper + 4 档读策略 + failed_ids 管理 | `full_read()` |
| `summarizer.py` | Anthropic SDK 封装（自定义 httpx transport）+ 中文 prompt + 单篇摘要 + enrichment JSON + 领域综述 | `summarize_single_paper()` / `generate_field_report()` / `load_enrichment()` |
| `enricher.py` | OpenAlex 客户端 + DOI 查询 + title search fallback + published-sibling 查询 + venue prestige 计分 | `enrich_all()` / `venue_prestige_score()` |
| `arxiv_source.py` | arXiv e-print 下载 + tar/gz/plain 解压 + 主 .tex 定位 + `\input/\include` 递归内联 | `fetch_latex()` |
| `formula_handler.py` | LaTeX 数学环境 + 定界符解析 + 上下文抽取 + 路由器（分派 arXiv/PDF/HTML） | `extract()` / `extract_from_latex()` / `save_formulas()` |
| `reporter.py` | composite_score 4 维评分 + weekly TOP10 Markdown 报告 | `composite_score()` / `generate_weekly_top10()` |
| `synthesizer.py` | 跨论文综合创新分析：公式交叉引用 / 相似度预计算 / 模块融合方向 / 融合公式提取 | `synthesize_top_papers()` |
| `qa_agent.py` | 文献知识库 RAG 问答：关键词检索 + 中英文混合提问 + 流式输出 + 引用溯源 | `ask()` / `retrieve()` / `interactive_loop()` |
| `project_analyzer.py` | 科研项目代码扫描（只读）+ LLM 匹配文献可迁移方法 + 集成建议报告 | `scan_project()` / `generate_integration_report()` |
| `agent_runtime.py` | 轻量 Agent Runtime：规划依赖、工具白名单、session 记忆、引用反思、论文比较与 trace | `AgentRuntime.run()` / `TaskPlanner.plan()` / `reflect_answer()` |
| `agent_eval.py` | Agent 离线规划评测与历史 trace 质量审计 | `evaluate_planner()` / `evaluate_trace()` / `audit_agent_runs()` |
| `run.py` | Pipeline 编排器 + CLI flags（含 --qa / --project-path） | `pipeline_run()` |
| `research_session.py` | 加载本地研究问题、瓶颈、约束和目标指标 | `load_research_context()` / `format_research_context()` |
| `quality_gate.py` | 本地质量初筛、证据缺口、迁移性评分和 evidence card | `assess_paper()` / `audit_library()` |
| `start.py` | 兼容入口，转发到 `src/run.py` | `main()` |

## 核心设计决策（& 为什么）

### 配置 / 凭证

- **LLM 凭证自动同步 Claude Code**（`utils.get_llm_config`）。优先级：显式环境变量 > Claude Code 的 `~/.claude/settings.json` > `.env` 文件。ANTHROPIC_AUTH_TOKEN 和 ANTHROPIC_API_KEY 都会被识别。
- **自定义 httpx transport**（`summarizer._make_http_client`）：禁用 HTTP/2 以兼容部分代理服务器的 SSL 实现（如 MiMo 代理）。
- **Agent 当你需要调 LLM 时，直接用 `summarizer.Anthropic()`**（不要用原生 `anthropic.Anthropic`），它会自动注入正确的凭证和 http transport。

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
- 当前依赖：`deepxiv-sdk`、`pyyaml`、`anthropic`、`httpx`、`requests`

## 下一步 TODO（Phase 2 + 优化）

### Phase 2：PDF 路线（`src/pdf_handler.py`，需校园网环境）

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
- [x] ~~**Web UI**~~：已移除，改用 Markdown 报告 + `index.json` + Agent 对话
- [ ] **桌面通知**：周一跑完后 win10toast 弹通知"本周 TOP10 已生成，点此查看"
- [ ] **Email 通知**：SMTP 发到你邮箱，含 TOP3 论文的精简摘要

## Agent 操作建议

### 主交互模式：Agent 驱动（重要）

用户希望 **Claude Code 作为项目的主交互入口**，替代菜单式交互。

工作方式：
1. 用户用自然语言描述需求（如"这周有什么新论文"、"帮我看看这篇论文"）
2. Agent 判断该执行哪些 pipeline 阶段，通过 `python src/run.py` + 对应 flags 执行
3. 结果以对话形式呈现，不要让用户自己去看文件

典型场景映射：
- "这周有什么新论文" → `python src/run.py`（search → read → summarize → report），展示 TOP10
- "帮我看看这篇 [arxiv_id]" → 单篇 `reader → summarizer → formula_handler`，给出完整解读
- "这篇论文值不值得作为研究依据" → 运行 `python src/run.py --paper-id <arxiv_id>` 后读取生成的 `.evidence.json`，或审查已有库
- "结合我的瓶颈判断这篇论文" → 配置本地 `research_context.yaml`，再运行 `--paper-id <arxiv_id> --research-context <path>`
- "有没有关于 X 的方法" → `python src/run.py --qa-question "问题"`
- "这个项目怎么改进" → `python src/run.py --project-path <路径>`
- "让 Agent 自己完成这个文献任务" → `python src/run.py --agent-question "<问题>" --agent-session <会话名>`
- "比较多篇论文的方法" → `python src/run.py --agent-question "比较 <arxiv_id1> 和 <arxiv_id2> 的方法差异"`
- "检查 Agent 规划和质量" → `python src/run.py --agent-eval` 或 `python src/run.py --agent-run-audit`
- "上次做到哪了" → 读 `seen_ids.json` + `git log` + 最新 candidates
- "跑一遍完整的" → `python src/run.py`（全 pipeline 六阶段）
- "只重新生成报告" → `python src/run.py --skip-search --no-llm --max-read 0`
- "重试之前失败的" → `python src/run.py --retry-failed`

`start.py` 仅保留为兼容包装，实际逻辑都在 `src/run.py`。

### 当你第一次打开这个项目时

1. 读本文件（AGENTS.md）
2. 读 [CHANGELOG.md](CHANGELOG.md) 看最新版本做了什么
3. 跑 `python src/utils.py` 冒烟验证 env 正常（会输出配置加载、logger 初始化）
4. 跑 `python src/run.py --skip-search --no-llm --skip-formulas --max-read 0` 确认 pipeline 各阶段能 dry run

### 当用户说"继续开发"时

1. 读 `DEVELOPMENT_PLAN.md` 的任务状态和执行记录，执行最早且依赖满足的未完成项
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

# 审查已有论文库（不调用 LLM）
python src/run.py --quality-audit

# Agent 规划离线评测 / 历史 trace 审计（不调用 LLM）
python src/run.py --agent-eval
python src/run.py --agent-run-audit

# 交互式启动器（CLI 备用入口，主方式是通过 Claude Code 对话）
python start.py

# 查看当前项目状态
git log --oneline -10
git status
```

## 联系方式 / 上下文

- **仓库**：https://github.com/Gyepcrchance5/smart-literature-agent
- **维护者**：Gyepcrchance5（GitHub handle）
- **主项目**：轴承故障诊断 R1 路线（CWRU 主 + PU 跨域 + FA-KD + 结构化剪枝），本工具为主项目服务
