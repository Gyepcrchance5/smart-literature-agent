"""总结与归纳模块：对精读结果生成结构化中文摘要 + 跨论文领域报告。

通过 Anthropic Messages API 调用 LLM。base_url 和 api_key 通过 utils.get_anthropic_config()
读取（env var 优先，fallback 到 ~/.claude/settings.json），不在代码里硬编码凭证。

默认模型可通过环境变量 LLM_MODEL 覆盖。示例值 "xiaomi/mimo-v2.5-pro" 是一个第三方路由代理
的内部模型标识（仅当 ANTHROPIC_BASE_URL 指向该代理时可用）；如果你直连 Anthropic 官方 API，
把 LLM_MODEL 改成 "claude-haiku-4-5-20251001" 或 "claude-sonnet-4-5" 之类的公开模型 ID。
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic as _Anthropic


def Anthropic(*args, **kwargs):
    """Create an Anthropic client, adding MiniMax bearer auth when needed."""
    base_url = kwargs.get("base_url")
    api_key = kwargs.get("api_key")
    base_url_text = str(base_url or "")
    if api_key and ("minimax.com" in base_url_text or "minimax.io" in base_url_text):
        headers = dict(kwargs.get("default_headers") or {})
        headers.setdefault("Authorization", f"Bearer {api_key}")
        kwargs["default_headers"] = headers
    return _Anthropic(*args, **kwargs)

from utils import (
    OUTPUT_DIR,
    get_anthropic_config,
    get_logger,
    load_keywords,
)

log = get_logger("summarizer")

PAPERS_DIR = OUTPUT_DIR / "papers"
REPORTS_DIR = OUTPUT_DIR / "reports"

# 默认模型：可通过环境变量 LLM_MODEL 覆盖。
# 默认值是一个示例路由名（配合兼容 Anthropic 协议的第三方代理使用），
# 克隆本仓库后请在 .env 或环境变量里设为你自己的模型 ID，例如
#   LLM_MODEL=claude-haiku-4-5-20251001
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "MiniMax-M2.7")

# 单篇摘要的最大输出长度（tokens）
# 注意：部分 reasoning model 的 thinking 过程也记在 output_tokens 里，
# 所以上限要给得宽一些，否则会在"思考完→正式输出到一半"时被截断。
SINGLE_SUMMARY_MAX_TOKENS = 12000
# 领域报告的最大输出长度
FIELD_REPORT_MAX_TOKENS = 16000
# 喂给模型的论文正文最多多少字符（避免超 context，同时控制成本）
MAX_CONTEXT_CHARS = 100000
MAX_FORMULA_CONTEXT_CHARS = 16000

FORMULA_IMPORTANCE_KEYWORDS = (
    "loss",
    "objective",
    "optimization",
    "arg",
    "max",
    "min",
    "attention",
    "distillation",
    "student",
    "teacher",
    "prune",
    "quant",
    "score",
    "similarity",
    "reconstruction",
    "regularization",
    "constraint",
    "error",
)


def _client() -> Anthropic:
    """创建 Anthropic 客户端（支持自定义 base_url，适配国内代理）。"""
    cfg = get_anthropic_config()
    if not cfg.get("api_key"):
        raise RuntimeError(
            "未找到 ANTHROPIC_API_KEY。请检查环境变量或 ~/.claude/settings.json 的 env 段。"
        )
    log.info("Anthropic 客户端 base_url=%s（key 已隐藏）", cfg.get("base_url") or "<default>")
    return Anthropic(api_key=cfg["api_key"], base_url=cfg.get("base_url"))


def _extract_content(paper_data: dict) -> str:
    """从 full_read 保存的 JSON 里抽取可供 LLM 阅读的正文片段。

    策略对齐 reader.full_read 的 4 种 strategy：
      raw          → 取 raw markdown，截断到 MAX_CONTEXT_CHARS
      selected     → 把所有 sections 拼起来（已经是精选章节），必要时截断
      preview      → 直接用 preview 文本
      metadata_only → 只有 abstract + tldr + keywords
    """
    strategy = paper_data.get("strategy")
    parts: list[str] = []

    abstract = paper_data.get("abstract") or ""
    if abstract:
        parts.append(f"## Abstract\n{abstract}")

    if strategy == "raw" and paper_data.get("raw"):
        parts.append(str(paper_data["raw"]))
    elif strategy == "selected":
        for name, body in (paper_data.get("sections") or {}).items():
            parts.append(f"## {name}\n{body}")
    elif strategy == "preview" and paper_data.get("preview"):
        parts.append(f"## Preview\n{paper_data['preview']}")
    else:
        # metadata_only 或其他：只用元数据
        tldr = paper_data.get("tldr")
        if tldr:
            parts.append(f"## TLDR\n{tldr}")

    content = "\n\n".join(parts)
    if len(content) > MAX_CONTEXT_CHARS:
        content = content[:MAX_CONTEXT_CHARS] + "\n\n[...正文过长已截断...]"
    return content


def _author_list(paper_data: dict, n: int = 3) -> str:
    authors = paper_data.get("authors") or []
    if isinstance(authors, str):
        return authors
    names = [a.get("name", "") for a in authors if isinstance(a, dict)]
    if not names:
        return "未知"
    if len(names) <= n:
        return "、".join(names)
    return f"{'、'.join(names[:n])} et al.（共 {len(names)} 人）"


def _formula_score(formula: dict) -> int:
    """Heuristic score for selecting formulas that explain the method."""
    text = " ".join(
        str(formula.get(k) or "")
        for k in ("label", "latex", "context_before", "context_after")
    ).lower()
    score = 0
    if formula.get("type") == "display":
        score += 10
    if formula.get("numbered"):
        score += 4
    if formula.get("label"):
        score += 3
    score += sum(2 for kw in FORMULA_IMPORTANCE_KEYWORDS if kw in text)
    latex = str(formula.get("latex") or "")
    if re.search(r"\\mathop|\\operatorname|\\sum|\\prod|\\frac|\\min|\\max|\\arg", latex):
        score += 2
    return score


def _load_formula_context(arxiv_id: str, max_display: int = 6) -> str:
    """Load selected formulas for the LLM prompt.

    Formula extraction is intentionally separate from summarization; this helper
    turns the saved JSON into a compact prompt block with context.
    """
    safe_id = arxiv_id.replace("/", "_")
    path = PAPERS_DIR / f"{safe_id}.formulas.json"
    if not path.exists():
        return "（未找到公式提取结果。本次摘要只能基于正文生成。）"

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "（公式提取结果无法读取。本次摘要只能基于正文生成。）"

    formulas = [
        f for f in payload.get("formulas", [])
        if isinstance(f, dict) and f.get("type") == "display" and f.get("latex")
    ]
    if not formulas:
        return "（未提取到 display 公式。若论文主要使用行内公式，请参考公式速览页。）"

    ranked = sorted(
        formulas,
        key=lambda f: (_formula_score(f), -(f.get("char_start") or 0)),
        reverse=True,
    )[:max_display]
    selected = sorted(ranked, key=lambda f: (f.get("eq_num") is None, f.get("eq_num") or 9999, f.get("char_start") or 0))

    lines = [
        f"共提取 display 公式 {payload.get('counts', {}).get('display', len(formulas))} 条；以下是按方法解释价值筛选的关键公式候选：",
        "",
    ]
    for f in selected:
        meta = [f.get("id", "formula")]
        if f.get("eq_num") is not None:
            meta.append(f"Eq.{f['eq_num']}")
        if f.get("label"):
            meta.append(f"label={f['label']}")
        before = re.sub(r"\s+", " ", str(f.get("context_before") or "")).strip()
        after = re.sub(r"\s+", " ", str(f.get("context_after") or "")).strip()
        lines.extend(
            [
                f"### {' · '.join(meta)}",
                f"上下文前：{before[-220:]}",
                "```latex",
                str(f.get("latex")),
                "```",
                f"上下文后：{after[:220]}",
                "",
            ]
        )

    block = "\n".join(lines)
    if len(block) > MAX_FORMULA_CONTEXT_CHARS:
        block = block[:MAX_FORMULA_CONTEXT_CHARS] + "\n\n[...公式上下文过长已截断...]"
    return block


def _build_single_paper_prompt(paper_data: dict, fields_config: dict) -> str:
    field_labels = [fc.get("label", k) for k, fc in fields_config.items()]
    content = _extract_content(paper_data)
    arxiv_id = paper_data.get("arxiv_id")
    formula_context = _load_formula_context(str(arxiv_id))
    return f"""你是深度学习、模型压缩、故障诊断和边缘部署方向的资深研究员。请基于下面这篇 arXiv 论文的正文与关键公式，产出一份**能帮助研究者看懂方法并判断迁移价值**的中文技术解读。

【论文元数据】
- arxiv_id: {arxiv_id}
- 英文标题: {paper_data.get('title')}
- 作者: {_author_list(paper_data)}
- 发表时间: {paper_data.get('publish_at')}
- arXiv 分类: {paper_data.get('categories')}
- 关键词: {paper_data.get('keywords')}
- 引用数: {paper_data.get('citations')}
- 精读策略: {paper_data.get('strategy')}（内容完整度：raw > selected > preview > metadata_only）

【论文正文（部分）】
{content}

【关键公式候选】
{formula_context}

请严格按照以下 Markdown 模板输出。重点解释"作者到底做了什么、怎么做、关键变量/公式在流程里的作用"。不要空泛复述摘要；如果正文或公式不足，明确说明证据不足。

# {{中文标题}}
**arXiv**：[{arxiv_id}](https://arxiv.org/abs/{arxiv_id}) | **作者**：{_author_list(paper_data)} | **发表**：{paper_data.get('publish_at')}

## 一句话结论
（1-2 句：本文提出了什么，解决什么痛点，最核心的技术抓手是什么）

## 方法拆解
### Step 1：问题建模与输入输出
（说明输入、输出、目标变量，以及论文如何把问题形式化）

### Step 2：核心模块与信息流
（按模块/阶段拆解，不少于 3 点；说明每个模块接收什么、输出什么、为什么需要它）

### Step 3：训练或推理流程
（说明训练损失、推理步骤、搜索/剪枝/量化/蒸馏等流程；如果论文只有推理方法也要说清楚）

## 关键公式解释
选择 2-5 个最关键公式。每个公式按下面格式写：

**公式 X：作用标题**
$$
公式 LaTeX
$$
- **符号含义**：解释主要变量，不要逐字翻译。
- **方法作用**：说明它约束/优化/计算了什么。
- **迁移映射**：如果迁移到轴承故障诊断，这个公式中的输入、标签、目标或约束分别可对应什么。

## 关键实验结果
（列出数据集/任务、baseline、关键指标和提升幅度；没有数字就说明论文未提供或正文未读到）

## 与六大研究方向的相关性
逐一评估本文与下列方向的相关性（高/中/低/无），简短说明一句为什么：
{chr(10).join('- ' + lbl for lbl in field_labels)}

## 可迁移技术路线
面向 **轴承故障诊断 + 模型压缩/知识蒸馏 + 边缘部署** 给出一条可执行路线，必须包含：
1. **可迁移组件**：本文哪一个模块/公式/训练策略值得迁移。
2. **融合方式**：如何与 CWRU/PU 跨域故障诊断、FA-KD、剪枝或量化结合。
3. **训练目标**：建议的损失函数组合或优化目标。
4. **部署路径**：边缘端如何落地，哪些部分离线做，哪些部分在线推理。
5. **风险点**：至少 2 个可能失败的原因和验证方法。

## 局限
（作者自述或你判断的局限；区分论文自身局限和迁移到故障诊断时的额外风险）
"""


def summarize_single_paper(
    paper_data: dict,
    model: str = DEFAULT_MODEL,
    save: bool = True,
) -> dict:
    """生成单篇结构化中文摘要，返回 {arxiv_id, model, summary, saved_to}。"""
    arxiv_id = paper_data.get("arxiv_id")
    if not arxiv_id:
        raise ValueError("paper_data 缺少 arxiv_id")

    cfg = load_keywords()
    fields_config = cfg.get("fields", {})
    prompt = _build_single_paper_prompt(paper_data, fields_config)

    log.info("生成单篇摘要：%s（模型=%s，prompt 长度=%d 字符）", arxiv_id, model, len(prompt))
    client = _client()
    msg = client.messages.create(
        model=model,
        max_tokens=SINGLE_SUMMARY_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    summary_text = "".join(
        blk.text for blk in msg.content if getattr(blk, "type", None) == "text"
    )

    usage = getattr(msg, "usage", None)
    log.info(
        "  摘要生成完成：输出 %d 字符（input_tokens=%s, output_tokens=%s）",
        len(summary_text),
        getattr(usage, "input_tokens", "?"),
        getattr(usage, "output_tokens", "?"),
    )

    result = {
        "arxiv_id": arxiv_id,
        "model": model,
        "summary": summary_text,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if save:
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = arxiv_id.replace("/", "_")
        md_path = PAPERS_DIR / f"{safe_id}.summary.md"
        md_path.write_text(summary_text, encoding="utf-8")
        log.info("  已保存：%s", md_path)
        result["saved_to"] = str(md_path)

    return result


def load_paper(arxiv_id: str) -> dict:
    """从 output/papers/<id>.json 加载精读产物。"""
    safe_id = arxiv_id.replace("/", "_")
    path = PAPERS_DIR / f"{safe_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_summary(arxiv_id: str) -> str | None:
    """读取已生成的单篇摘要 markdown。不存在返回 None。"""
    safe_id = arxiv_id.replace("/", "_")
    path = PAPERS_DIR / f"{safe_id}.summary.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _build_field_report_prompt(field_label: str, summaries: list[tuple[str, str]]) -> str:
    """summaries: [(arxiv_id, markdown_summary), ...]"""
    sep = "\n\n" + "=" * 60 + "\n\n"
    papers_block = sep.join(
        f"### 论文 {i+1}：{aid}\n\n{md}" for i, (aid, md) in enumerate(summaries)
    )
    return f"""你是 **{field_label}** 方向的综述作者。下面是 {len(summaries)} 篇最新相关论文的结构化中文摘要，请基于它们产出一份**领域进展综述**。

【已读论文摘要】
{papers_block}

【输出要求】
请生成一份 Markdown 领域综述，严格使用下面的章节：

# {field_label} 领域近期进展（基于 {len(summaries)} 篇论文）

## 1. 核心问题聚类
把 N 篇论文按"研究问题"聚成 2-4 类，每类列出哪些论文（用 arxiv_id 引用），一句话说明共同问题。

## 2. 主流技术路线
归纳 2-4 条主流方法流派，每条给代表论文（arxiv_id）+ 一句技术要点。

## 3. 常用数据集与评测
若多篇提到同一数据集或基准，汇总列出；若各做各的，也如实说明"暂无统一基准"。

## 4. 当前局限与开放问题
2-4 条观察到的痛点、有争议处、尚未解决的问题。

## 5. 对"轴承故障诊断 + 模型压缩/知识蒸馏 + 边缘部署"研究者的启发
具体、可操作的迁移建议（最多 5 条），每条点明要迁移哪篇论文的什么想法到这个目标场景。

要求：
- 引用论文时用 `[arxiv_id]` 的形式。
- 不要把论文摘要原文抄回去，要**抽取+归纳**。
- 避免"本文讨论了..."这样的空话。
- 若论文数量太少（≤3）导致某些结论不牢靠，明确写"样本量有限，仅作初步观察"。
"""


def generate_field_report(
    field_key: str,
    arxiv_ids: list[str],
    model: str = DEFAULT_MODEL,
    save: bool = True,
) -> dict:
    """基于已生成的单篇摘要，合成领域综述。

    Args:
        field_key: config/keywords.yaml 里 fields.* 的 key
        arxiv_ids: 属于该领域的 arxiv_id 列表（调用方决定怎么筛选）
    """
    cfg = load_keywords()
    field_conf = cfg.get("fields", {}).get(field_key)
    if not field_conf:
        raise ValueError(f"未知 field_key: {field_key}")
    field_label = field_conf.get("label", field_key)

    summaries: list[tuple[str, str]] = []
    missing: list[str] = []
    for aid in arxiv_ids:
        md = _load_summary(aid)
        if md:
            summaries.append((aid, md))
        else:
            missing.append(aid)
    if missing:
        log.warning("以下 id 尚无 summary.md，本次领域报告跳过：%s", missing)
    if not summaries:
        raise RuntimeError(f"领域 {field_key} 没有任何可用的单篇摘要")

    log.info("生成领域综述：%s（%d 篇可用）", field_label, len(summaries))
    prompt = _build_field_report_prompt(field_label, summaries)
    log.info("  prompt 长度=%d 字符", len(prompt))

    client = _client()
    msg = client.messages.create(
        model=model,
        max_tokens=FIELD_REPORT_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    report_md = "".join(
        blk.text for blk in msg.content if getattr(blk, "type", None) == "text"
    )

    usage = getattr(msg, "usage", None)
    log.info(
        "  综述生成完成：输出 %d 字符（input_tokens=%s, output_tokens=%s）",
        len(report_md),
        getattr(usage, "input_tokens", "?"),
        getattr(usage, "output_tokens", "?"),
    )

    result = {
        "field": field_key,
        "field_label": field_label,
        "model": model,
        "arxiv_ids": [aid for aid, _ in summaries],
        "report": report_md,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / f"{field_key}_{datetime.now():%Y%m%d}.md"
        out.write_text(report_md, encoding="utf-8")
        log.info("  已保存：%s", out)
        result["saved_to"] = str(out)

    return result


def field_ids_from_candidates(
    field_key: str, candidates_path: Path | str | None = None
) -> list[str]:
    """从 data/candidates_<date>.json 里提取属于指定 field 的 arxiv_id 列表。
    不传 candidates_path 就找最新的一个。"""
    if candidates_path is None:
        files = sorted((OUTPUT_DIR.parent / "data").glob("candidates_*.json"))
        if not files:
            return []
        candidates_path = files[-1]
    data = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    return [p["arxiv_id"] for p in data.get("papers", []) if field_key in p.get("_fields", [])]


def _score_relevance_heuristic(paper_data: dict, fields_config: dict) -> dict[str, int]:
    """简单命中计数评分（1-5），不调 LLM。
    统计 title + abstract + keywords 里命中每个领域关键词的次数，映射到 1-5 档。
    """
    text = " ".join(
        str(paper_data.get(k, ""))
        for k in ("title", "abstract", "tldr")
    ).lower()
    keywords_list = paper_data.get("keywords") or []
    text += " " + " ".join(str(k).lower() for k in keywords_list)

    scores: dict[str, int] = {}
    for field_key, field_conf in fields_config.items():
        hits = sum(1 for kw in field_conf.get("keywords", []) if kw.lower() in text)
        # 0 hits → 1, 1 → 2, 2 → 3, 3-4 → 4, ≥5 → 5
        if hits == 0:
            score = 1
        elif hits == 1:
            score = 2
        elif hits == 2:
            score = 3
        elif hits <= 4:
            score = 4
        else:
            score = 5
        scores[field_key] = score
    return scores


def score_relevance(paper_data: dict) -> dict[str, int]:
    """论文与 6 领域的相关性评分 1-5。"""
    cfg = load_keywords()
    return _score_relevance_heuristic(paper_data, cfg.get("fields", {}))


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "--field":
        # 用法：python src/summarizer.py --field <field_key> [id1 id2 ...]
        field_key = sys.argv[2]
        manual_ids = sys.argv[3:]
        if manual_ids:
            ids = manual_ids
        else:
            ids = field_ids_from_candidates(field_key)
            log.info("从 candidates 自动抽取 %s 下的 %d 个 id", field_key, len(ids))
        out = generate_field_report(field_key, ids)
        print("\n=========== GENERATED FIELD REPORT ===========")
        print(out["report"])
    else:
        # 单篇：python src/summarizer.py <arxiv_id> [model]
        arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2411.11707"
        model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
        paper = load_paper(arxiv_id)
        out = summarize_single_paper(paper, model=model)
        print("\n=========== GENERATED SUMMARY ===========")
        print(out["summary"])
