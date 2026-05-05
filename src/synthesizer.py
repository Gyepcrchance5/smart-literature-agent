"""跨论文综合创新分析模块：对本周 TOP 论文做交叉对比，发现可融合的模块与创新方向。

输出：output/reports/synthesis_<date>.md —— 综合创新分析报告
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

from summarizer import (
    Anthropic,
    DEFAULT_MODEL,
    PAPERS_DIR,
    REPORTS_DIR,
    _load_formula_context,
)
from utils import get_anthropic_config, get_logger, load_keywords

log = get_logger("synthesizer")

SYNTHESIS_MAX_TOKENS = 16000
MAX_PAPERS_FOR_SYNTHESIS = 10
MAX_SYNTHESIS_FORMULA_CHARS = 8000


def _load_synthesis_formulas(arxiv_id: str, max_display: int = 3) -> str | None:
    """Load compact formula context for the synthesis prompt.

    Returns a shortened version of _load_formula_context() with a tighter
    token budget.  Returns None when no display formulas are available so
    the caller can omit the formula block entirely.
    """
    block = _load_formula_context(arxiv_id, max_display=max_display)
    if block.startswith("（未"):
        return None
    if len(block) > MAX_SYNTHESIS_FORMULA_CHARS:
        block = block[:MAX_SYNTHESIS_FORMULA_CHARS] + "\n\n[...公式上下文过长已截断...]"
    return block


def _load_formula_structs(arxiv_id: str) -> list[dict]:
    """Load structured display formulas from .formulas.json, returning [{id, latex, eq_num, label}, ...]."""
    safe_id = arxiv_id.replace("/", "_")
    path = PAPERS_DIR / f"{safe_id}.formulas.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        {"id": f.get("id", ""), "latex": f.get("latex", ""),
         "eq_num": f.get("eq_num"), "label": f.get("label")}
        for f in data.get("formulas", [])
        if isinstance(f, dict) and f.get("type") == "display" and f.get("latex")
    ]


_LATEX_TOKEN_RE = re.compile(
    r"\\[a-zA-Z]+|\\[α-ω]|[+=\-*/<>^_{}|]|\\[()[\]]|\\frac|\\sum|\\prod|\\int|\\min|\\max|\\arg|\\mathcal|\\mathbf|\\mathbb|\\text"
)


def _tokenize_latex(latex: str) -> set[str]:
    """Extract a bag of structural LaTeX tokens for similarity comparison."""
    return set(_LATEX_TOKEN_RE.findall(latex.lower()))


def _formula_similarity_hints(formula_structs: dict[str, list[dict]]) -> str:
    """Pre-compute cross-paper LaTeX token similarity, return a markdown hint block.

    Args:
        formula_structs: {arxiv_id: [{id, latex, eq_num, label}, ...]}
    Returns:
        Markdown table for prompt injection, or empty string if insufficient data.
    """
    # Build flat list of (paper, formula_id, latex, tokens)
    entries = []
    for aid, formulas in formula_structs.items():
        for f in formulas:
            toks = _tokenize_latex(f["latex"])
            if len(toks) >= 3:  # skip trivial formulas
                entries.append((aid, f["id"], f.get("eq_num"), f.get("label"), f["latex"], toks))
    if len(entries) < 4:
        return ""

    # Cross-paper pairs only, compute Jaccard
    pairs = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if entries[i][0] == entries[j][0]:
                continue  # same paper
            shared = entries[i][5] & entries[j][5]
            total = entries[i][5] | entries[j][5]
            if not total:
                continue
            sim = len(shared) / len(total)
            if sim >= 0.15:
                shared_names = ", ".join(sorted(shared, key=lambda t: (-len(t), t))[:5])
                pairs.append((sim, entries[i], entries[j], shared_names))

    if not pairs:
        return ""

    pairs.sort(key=lambda x: x[0], reverse=True)

    lines = [
        "## 公式相似度提示（自动计算，供交叉引用参考）",
        "",
        "以下公式对在 LaTeX 结构上相似，可能具有数学上的可组合性：",
        "",
        "| 公式 A | 公式 B | 共享 LaTeX 结构 | 相似度 |",
        "| :--- | :--- | :--- | ---: |",
    ]
    for sim, a, b, shared in pairs[:8]:
        a_label = f"{a[0]} {a[1]}"
        if a[2] is not None:
            a_label += f" (Eq.{a[2]})"
        b_label = f"{b[0]} {b[1]}"
        if b[2] is not None:
            b_label += f" (Eq.{b[2]})"
        lines.append(f"| {a_label} | {b_label} | {shared} | {sim:.2f} |")

    lines.append("")
    return "\n".join(lines)


def _build_paper_card(
    arxiv_id: str, score: dict, summary_md: str, formula_context: str | None = None
) -> str:
    """构建单篇论文的结构化卡片，供给 LLM 做交叉比较。"""
    title = score.get("title", "") or ""
    composite = score.get("composite", 0)
    relevance = score.get("relevance", 0)
    venue_name = score.get("venue_name") or "arXiv"
    citation_count = score.get("citation_count", 0)
    fields = ", ".join(score.get("breakdown", {}).keys())

    card = f"""### [{arxiv_id}] {title}
- 综合分: {composite:.1f} | 相关性: {relevance:.0f} | 来源: {venue_name} | 引用: {citation_count}
- 涉及领域: {fields}
- 论文摘要:
{summary_md}
"""
    if formula_context:
        card += f"""- 关键公式（用于方法对比与融合设计）:
{formula_context}
"""
    return card


def _load_summary(arxiv_id: str) -> str:
    safe_id = arxiv_id.replace("/", "_")
    path = PAPERS_DIR / f"{safe_id}.summary.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "(摘要尚未生成)"


def _build_comparison_prompt(
    paper_cards: list[str], fields_config: dict, similarity_hints: str = ""
) -> str:
    cards_text = "\n---\n".join(paper_cards)
    field_names = ", ".join(fields_config.keys())

    hints_section = ""
    if similarity_hints:
        hints_section = f"\n{similarity_hints}\n"

    return f"""你是一位资深科研方法学家，擅长跨领域、跨论文的方法融合与创新发现。
以下是你本周筛选出的 TOP 论文的结构化摘要，覆盖领域：{field_names}。
每篇论文卡片可能包含"关键公式"段，其中给出了该论文最重要的 display 公式（含 LaTeX 源码和上下文）。请充分利用这些公式进行精确的方法对比和创新设计。

{cards_text}
{hints_section}

请完成以下分析，输出中文：

## 一、共享问题景观

找出这些论文共同关注的核心问题（若有），以及各自独有但可能互补的问题。
对每个问题，标注哪些论文在解决它。

## 二、方法矩阵

以表格形式列出每篇论文的核心方法/模块，并标注它们之间的潜在关联。
**如果某篇论文提供了关键公式，请在"核心方法/模块"列中引用相关公式编号（如 Eq.1、f12）。**

| 论文 | 核心方法/模块 | 可与之组合的其他论文方法 | 组合潜力说明 |

## 三、冲突与互补

如果某些论文的结论冲突或互为补充，逐条指出，并说明可能的调和方式。

## 四、模块融合创新方向

基于以上分析，提出 2-4 个具体的**模块级融合创新方向**。每个方向：
- 命名（一个简短的代号）
- 涉及哪些论文的哪些模块（引用公式编号，如 [paper_id] Eq.3）
- **融合公式建议**：用 LaTeX 写出融合后的核心公式（例如组合损失函数、联合优化目标、或合并的约束条件）。公式写在 $$...$$ 或 $...$ 中。
- 融合后的预期收益
- 潜在风险或挑战

## 五、推荐深入阅读顺序

给出一个推荐的阅读顺序（不是按评分，而是按理解难度和依赖关系），并简短说明原因。

## 六、关键公式交叉引用

如果多篇论文提供了关键公式，请完成以下分析：

### 6.1 公式层面的可组合性
找出 2-4 组来自不同论文的公式，说明它们在数学上如何组合。格式：

| 公式来源 | 公式编号 | 可组合的公式来源 | 组合方式 | 潜在价值 |
| :--- | :--- | :--- | :--- | :--- |
| [paper_id] | Eq.N | [paper_id2] Eq.M | （例如：将 Eq.N 作为正则项加入 Eq.M 的损失函数） | （一句话） |

### 6.2 建议的融合公式
写出 1-2 个具体的融合公式（LaTeX 格式，写在 $$...$$ 块中）。

对每个融合公式，提供 **变量映射表**：

| 源论文变量 | 含义 | → | 融合公式变量 | 含义 |
| :--- | :--- | :--- | :--- | :--- |
| [paper_id] x_i | 输入特征 | → | x_i^fused | 融合后的输入 |

（如果论文未提供公式，此节可写"无可用于交叉引用的公式"并跳过。）
"""


def _parse_synthesis_response(text: str) -> str:
    """清理 LLM 返回的文本，去掉可能的 JSON wrapper 或 leading/trailing 噪声。"""
    text = text.strip()
    # 如果被包在 ```markdown ``` 或 ``` 里，提取出来
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_fused_formulas(synthesis_md: str, report_path: Path | None = None) -> list[dict]:
    """Parse $$...$$ blocks from the synthesis report into structured JSON."""
    fused = []
    heading_re = re.compile(r"^#{2,4}\s+(.+)$", re.MULTILINE)

    for m in re.finditer(r"\$\$(.+?)\$\$", synthesis_md, re.DOTALL):
        latex = m.group(1).strip()
        # Skip trivial or non-LaTeX blocks
        if len(latex) < 15 or "\\" not in latex:
            continue

        pos = m.start()
        pre_text = synthesis_md[:pos]

        # Find nearest heading
        headings = list(heading_re.finditer(pre_text))
        name = headings[-1].group(1) if headings else ""

        # Get preceding text context (up to 200 chars)
        context_start = max(0, pos - 400)
        context_before = synthesis_md[context_start:pos].strip()
        # Take the last ~200 chars before the formula
        if len(context_before) > 200:
            context_before = "..." + context_before[-200:]

        # Extract paper and formula references
        search_text = latex + " " + context_before
        paper_refs = set(re.findall(r"\[(\d{4}\.\d{4,5}(?:v\d+)?)\]", search_text))
        formula_refs = set(re.findall(r"(Eq\.\d+|f\d+)", search_text))

        fused.append({
            "name": name[:80],
            "latex": latex,
            "context_before": context_before,
            "source_papers": sorted(paper_refs),
            "source_formulas": sorted(formula_refs),
        })

    if fused and report_path:
        sidecar = report_path.with_suffix(".formulas.json")
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_report": str(report_path.name),
            "fused_formulas_count": len(fused),
            "fused_formulas": fused,
        }
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("  融合公式已提取并保存：%s（%d 个公式）", sidecar, len(fused))

    return fused


def synthesize_top_papers(
    arxiv_ids: list[str],
    scores: list[dict],
    fields_config: dict | None = None,
    save: bool = True,
    model: str | None = None,
) -> dict:
    """对 TOP 论文做交叉综合创新分析。

    The synthesis prompt includes top-3 display formulas per paper (loaded
    from output/papers/<id>.formulas.json when available) and requests
    formula-level cross-reference analysis.

    Args:
        arxiv_ids: 按综合分降序排列的论文 ID 列表
        scores: 对应的 composite_score 输出列表（需与 arxiv_ids 一一对应）
        fields_config: keywords.yaml 里的 fields dict
        save: 是否保存到 output/reports/
        model: LLM 模型 ID（默认从环境变量读）

    Returns:
        {"synthesis_md": str, "saved_to": str | None}
    """
    if fields_config is None:
        fields_config = load_keywords().get("fields", {})

    n = min(len(arxiv_ids), MAX_PAPERS_FOR_SYNTHESIS)
    if n < 2:
        raise RuntimeError(f"至少需要 2 篇有摘要的论文才能做综合创新分析（当前 {n}）")

    log.info("综合创新分析：对 TOP %d 篇论文做交叉比较", n)

    # 构建论文卡片（含关键公式，如已提取）+ 收集结构化公式
    paper_cards = []
    formula_count = 0
    formula_structs: dict[str, list[dict]] = {}
    for i in range(n):
        aid = arxiv_ids[i]
        s = scores[i] if i < len(scores) else {}
        summary = _load_summary(aid)
        formula_ctx = _load_synthesis_formulas(aid, max_display=3)
        if formula_ctx is not None:
            formula_count += 1
        card = _build_paper_card(aid, s, summary, formula_ctx)
        paper_cards.append(card)
        # 加载结构化公式用于相似度计算
        fs = _load_formula_structs(aid)
        if fs:
            formula_structs[aid] = fs
    log.info("  其中 %d/%d 篇论文有关键公式可供交叉引用", formula_count, n)

    # 公式相似度预计算提示
    similarity_hints = _formula_similarity_hints(formula_structs)

    # 调用 LLM
    cfg = get_anthropic_config()
    client = Anthropic(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        max_retries=2,
    )
    resolved_model = model or DEFAULT_MODEL

    prompt = _build_comparison_prompt(paper_cards, fields_config, similarity_hints)

    log.info("  正在调用 %s 生成综合创新分析报告...", resolved_model)
    response = client.messages.create(
        model=resolved_model,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = ""
    for block in response.content:
        if getattr(block, "type", "") == "text":
            raw_text += getattr(block, "text", "")

    synthesis_md = _parse_synthesis_response(raw_text)

    # 构建完整报告
    today = datetime.now().strftime("%Y-%m-%d")
    full_report = f"""# 跨论文综合创新分析报告

> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}
> 分析范围：本周 TOP {n} 篇论文（其中 {formula_count} 篇有关键公式）
> 模型：{resolved_model}

## 分析论文清单

"""
    for i in range(n):
        aid = arxiv_ids[i]
        s = scores[i] if i < len(scores) else {}
        title = s.get("title", "") or ""
        venue = s.get("venue_name") or "arXiv"
        full_report += f"{i+1}. **[{aid}](https://arxiv.org/abs/{aid})** — {title} ({venue}, 综合分 {s.get('composite', 0):.1f})\n"

    full_report += f"\n---\n\n{synthesis_md}"

    result: dict = {
        "synthesis_md": full_report,
        "saved_to": None,
        "analyzed_count": n,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / f"synthesis_{today}.md"
        out.write_text(full_report, encoding="utf-8")
        log.info("  综合创新分析报告已保存：%s", out)
        result["saved_to"] = str(out)
        # 提取融合公式为结构化 JSON
        fused = _extract_fused_formulas(full_report, out)
        result["fused_formulas_count"] = len(fused)

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python src/synthesizer.py <weekly_top_md_path>")
        print("  or:  python src/synthesizer.py --from-scores <scores_json>")
        sys.exit(1)

    # 从 reporter 的输出快速测试
    if sys.argv[1] == "--from-scores":
        data = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        result = synthesize_top_papers(
            arxiv_ids=data["arxiv_ids"],
            scores=data["scores"],
        )
    else:
        # 从 weekly top markdown 解析
        top_md = Path(sys.argv[1]).read_text(encoding="utf-8")
        ids = []
        for line in top_md.split("\n"):
            match = __import__("re").search(r"\[(\d{4}\.\d{4,5}(?:v\d+)?)\]", line)
            if match:
                ids.append(match.group(1))
        if not ids:
            print("未能从文件中解析出 arXiv ID")
            sys.exit(1)
        result = synthesize_top_papers(arxiv_ids=ids, scores=[{}] * len(ids))

    print(result["synthesis_md"][:500])
