"""报告增强模块：综合评分 + 本周 TOP10 合并报告 + HTML 渲染 + 自动打开浏览器。

综合评分权重（按用户设定）：
  启发式相关性 50% + DeepXiv search score 30% + 引用数 20%

HTML 渲染：把所有 .md 产物（单篇 summary / 领域综述 / 本周 TOP10）一一转成同名 .html，
          并生成 output/index.html 总览页，按"本周 TOP10 / 领域综述 / 单篇摘要"三块组织。
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

import markdown

from summarizer import PAPERS_DIR, REPORTS_DIR, score_relevance
from utils import DATA_DIR, OUTPUT_DIR, get_logger, load_keywords

log = get_logger("reporter")

HTML_DIR = OUTPUT_DIR / "html"
INDEX_HTML = HTML_DIR / "index.html"

# 综合评分权重
W_RELEVANCE = 0.5
W_DEEPXIV = 0.3
W_CITATION = 0.2

# 启发式相关性：6 领域 × 最高 5 分 = 30 为满分
_RELEVANCE_MAX = 6 * 5
# DeepXiv score 通常 3~6，截断到 10 归一化
_DEEPXIV_CAP = 10.0
# 引用数 log 归一化：log1p(999)/log(1000) ≈ 1.0（即引用 ~1000 即满分）
_CITATION_CAP = 1000

# GitHub 风格的简单 CSS
_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       line-height: 1.6; color: #24292f; max-width: 920px; margin: 2em auto; padding: 0 1em; }
h1, h2, h3, h4 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.3em; margin-top: 1.6em; }
h1 { font-size: 2em; } h2 { font-size: 1.5em; } h3 { font-size: 1.25em; }
a { color: #0969da; text-decoration: none; }
a:hover { text-decoration: underline; }
code { background: #f6f8fa; padding: 0.2em 0.4em; border-radius: 4px; font-size: 0.9em; }
pre { background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto; }
blockquote { border-left: 4px solid #d0d7de; margin: 0; padding-left: 1em; color: #57606a; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; }
table th, table td { border: 1px solid #d0d7de; padding: 6px 13px; }
table th { background: #f6f8fa; }
.meta { color: #57606a; font-size: 0.9em; }
.score-badge { display: inline-block; padding: 2px 8px; background: #0969da; color: white;
               border-radius: 12px; font-size: 0.85em; font-weight: 600; }
.rank { color: #57606a; font-weight: 600; font-family: monospace; }
.index-section { margin: 1em 0 2em 0; }
.index-section li { margin: 0.3em 0; }
.nav-back { display: inline-block; margin-top: 2em; padding: 6px 12px;
            background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; }
"""


# ================================================================
# 综合评分
# ================================================================


def composite_score(paper: dict, fields_config: dict | None = None) -> dict:
    """对 candidates.json 里的 paper entry 计算综合评分。

    返回 {"composite", "relevance", "deepxiv", "citation", "breakdown": {field: 1-5}}
    composite / 其他分量都归一化到 0-100 区间。
    """
    if fields_config is None:
        fields_config = load_keywords().get("fields", {})

    rel_per_field = score_relevance(paper)  # {field: 1-5}
    rel_total = sum(rel_per_field.values())  # 最多 30
    rel_norm = rel_total / _RELEVANCE_MAX * 100

    deepxiv_raw = float(paper.get("score") or 0)
    deepxiv_norm = min(deepxiv_raw, _DEEPXIV_CAP) / _DEEPXIV_CAP * 100

    cite_raw = int(paper.get("citation_count") or 0)
    cite_norm = (
        min(math.log1p(cite_raw) / math.log(_CITATION_CAP), 1.0) * 100 if cite_raw > 0 else 0
    )

    composite = W_RELEVANCE * rel_norm + W_DEEPXIV * deepxiv_norm + W_CITATION * cite_norm

    return {
        "composite": round(composite, 2),
        "relevance": round(rel_norm, 2),
        "deepxiv": round(deepxiv_norm, 2),
        "citation": round(cite_norm, 2),
        "citation_count": cite_raw,
        "deepxiv_raw": round(deepxiv_raw, 3),
        "breakdown": rel_per_field,
    }


def _latest_candidates_path() -> Path | None:
    files = sorted(DATA_DIR.glob("candidates_*.json"))
    return files[-1] if files else None


def _has_summary(arxiv_id: str) -> bool:
    safe_id = arxiv_id.replace("/", "_")
    return (PAPERS_DIR / f"{safe_id}.summary.md").exists()


def _summary_path(arxiv_id: str) -> Path:
    safe_id = arxiv_id.replace("/", "_")
    return PAPERS_DIR / f"{safe_id}.summary.md"


# ================================================================
# 本周 TOP N 合并报告
# ================================================================


def generate_weekly_top10(
    top_n: int = 10,
    candidates_path: Path | str | None = None,
    require_summary: bool = True,
    save: bool = True,
) -> dict:
    """基于最新 candidates 算综合分，取 TOP N（优先要求有 summary），生成合并 markdown 报告。"""
    if candidates_path is None:
        candidates_path = _latest_candidates_path()
    if not candidates_path:
        raise RuntimeError("没有 candidates 文件，先跑一次 search")
    data = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    papers = data.get("papers", [])
    fields_config = load_keywords().get("fields", {})

    # 算分
    scored = []
    for p in papers:
        s = composite_score(p, fields_config)
        scored.append((s, p))
    scored.sort(key=lambda x: x[0]["composite"], reverse=True)

    # 优先取有 summary 的；如果不足 top_n，用全部有 summary 的
    with_summary = [(s, p) for s, p in scored if _has_summary(p["arxiv_id"])]
    if require_summary:
        top = with_summary[:top_n]
    else:
        top = scored[:top_n]

    if not top:
        raise RuntimeError("没有任何论文满足 TOP 条件（可能是还没生成 summary）")

    log.info(
        "TOP %d：候选总数 %d，有 summary 的 %d，本次入选 %d",
        top_n, len(papers), len(with_summary), len(top),
    )

    # 拼 markdown
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# smart-literature-agent 本周 TOP {len(top)} 论文速递",
        "",
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}  ",
        f"> 候选池：`{Path(candidates_path).name}`（共 {len(papers)} 篇，时间窗 {data.get('date_from')} 起）  ",
        f"> 评分公式：启发式相关性 × {W_RELEVANCE:.0%} + DeepXiv 分 × {W_DEEPXIV:.0%} + 引用数 × {W_CITATION:.0%}  ",
        "> 入选条件：已生成 .summary.md 的候选，按综合分降序",
        "",
        "## 综合排名",
        "",
        "| 排名 | arXiv | 标题 | 综合分 | 相关性 | DeepXiv | 引用 | 领域命中 |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for rank, (s, p) in enumerate(top, 1):
        aid = p["arxiv_id"]
        title = (p.get("title") or "").replace("|", "/")
        fields_hit = ", ".join(p.get("_fields") or [])
        lines.append(
            f"| #{rank} | [{aid}](https://arxiv.org/abs/{aid}) | {title[:60]} "
            f"| {s['composite']:.1f} | {s['relevance']:.0f} | {s['deepxiv']:.0f} "
            f"| {s['citation_count']} | {fields_hit} |"
        )
    lines.append("")

    # 每篇的完整 summary
    lines.append("---")
    lines.append("")
    lines.append("## 详细摘要")
    lines.append("")
    for rank, (s, p) in enumerate(top, 1):
        aid = p["arxiv_id"]
        summary_md = _summary_path(aid).read_text(encoding="utf-8")
        # 给原 summary 加个排名条和评分条
        rank_line = (
            f"### #{rank} · 综合分 {s['composite']:.1f}"
            f"（相关 {s['relevance']:.0f} / DeepXiv {s['deepxiv']:.0f} / 引用 {s['citation_count']}）"
        )
        lines.extend([rank_line, "", summary_md, "", "---", ""])

    report_md = "\n".join(lines)

    result = {
        "top_n": len(top),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "arxiv_ids": [p["arxiv_id"] for _, p in top],
        "scores": [s for s, _ in top],
        "report_md": report_md,
    }

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out = REPORTS_DIR / f"weekly_top{top_n}_{datetime.now():%Y%m%d}.md"
        out.write_text(report_md, encoding="utf-8")
        log.info("本周 TOP %d 已保存：%s", top_n, out)
        result["saved_to"] = str(out)

    return result


# ================================================================
# HTML 渲染
# ================================================================


def _md_to_html(md_text: str, title: str, back_link: str | None = None) -> str:
    """把一段 markdown 渲染为带样式的独立 HTML 页面。"""
    body_html = markdown.markdown(
        md_text,
        extensions=["extra", "nl2br", "tables", "sane_lists"],
    )
    back = (
        f'<a class="nav-back" href="{back_link}">&larr; 返回索引</a>' if back_link else ""
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>{_CSS}</style>
</head>
<body>
{body_html}
{back}
</body>
</html>
"""


def render_html_all(open_browser: bool = False) -> Path:
    """把 output/reports/*.md 和 output/papers/*.summary.md 全部转成 HTML，生成 index.html。
    返回 index.html 的绝对路径。open_browser=True 时最后自动打开。
    """
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    (HTML_DIR / "reports").mkdir(exist_ok=True)
    (HTML_DIR / "papers").mkdir(exist_ok=True)

    back = "../index.html"

    # 1) 领域综述 / 本周 TOP → output/html/reports/
    report_entries: list[tuple[str, str]] = []  # [(display_name, href)]
    for md_path in sorted(REPORTS_DIR.glob("*.md")):
        html_path = HTML_DIR / "reports" / (md_path.stem + ".html")
        md_text = md_path.read_text(encoding="utf-8")
        html_path.write_text(_md_to_html(md_text, md_path.stem, back), encoding="utf-8")
        report_entries.append((md_path.stem, f"reports/{html_path.name}"))

    # 2) 单篇 summary → output/html/papers/
    paper_entries: list[tuple[str, str]] = []
    for md_path in sorted(PAPERS_DIR.glob("*.summary.md")):
        html_path = HTML_DIR / "papers" / (md_path.stem + ".html")
        md_text = md_path.read_text(encoding="utf-8")
        html_path.write_text(_md_to_html(md_text, md_path.stem, back), encoding="utf-8")
        arxiv_id = md_path.stem.replace(".summary", "")
        paper_entries.append((arxiv_id, f"papers/{html_path.name}"))

    # 3) 组索引页
    # 分组：weekly_top 单独置顶，其他 report 作为领域综述区
    weekly = [e for e in report_entries if e[0].startswith("weekly_top")]
    field_reports = [e for e in report_entries if not e[0].startswith("weekly_top")]

    # 单篇摘要按综合分排序，以便 index 一眼看到最值得的
    cand_path = _latest_candidates_path()
    scored_paper_entries = paper_entries
    cand_scores: dict[str, float] = {}
    if cand_path:
        cand_data = json.loads(cand_path.read_text(encoding="utf-8"))
        fc = load_keywords().get("fields", {})
        for p in cand_data.get("papers", []):
            s = composite_score(p, fc)
            cand_scores[p["arxiv_id"]] = s["composite"]
    scored_paper_entries = sorted(
        paper_entries, key=lambda e: cand_scores.get(e[0], -1), reverse=True
    )

    def li(entries, show_score=False):
        items = []
        for name, href in entries:
            if show_score and name in cand_scores:
                badge = f'<span class="score-badge">{cand_scores[name]:.1f}</span> '
            else:
                badge = ""
            items.append(f'  <li>{badge}<a href="{href}">{name}</a></li>')
        return "\n".join(items)

    candidates_name = cand_path.name if cand_path else "（无）"
    idx_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>smart-literature-agent 报告索引</title>
<style>{_CSS}</style>
</head>
<body>
<h1>smart-literature-agent 报告索引</h1>
<p class="meta">生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}　|　候选池：<code>{candidates_name}</code>　|　单篇摘要：{len(paper_entries)}　|　领域综述：{len(field_reports)}　|　本周 TOP：{len(weekly)}</p>

<div class="index-section">
<h2>本周 TOP</h2>
<ul>
{li(weekly)}
</ul>
</div>

<div class="index-section">
<h2>领域综述</h2>
<ul>
{li(field_reports)}
</ul>
</div>

<div class="index-section">
<h2>单篇摘要（按综合评分降序）</h2>
<ul>
{li(scored_paper_entries, show_score=True)}
</ul>
</div>

<p class="meta">评分公式：启发式相关性 × {W_RELEVANCE:.0%} + DeepXiv 分 × {W_DEEPXIV:.0%} + 引用数 × {W_CITATION:.0%}</p>
</body>
</html>
"""
    INDEX_HTML.write_text(idx_html, encoding="utf-8")
    log.info(
        "HTML 渲染完成：%s（%d 综述 + %d 单篇 + %d 本周 TOP）",
        INDEX_HTML, len(field_reports), len(paper_entries), len(weekly),
    )

    if open_browser:
        open_in_browser(INDEX_HTML)

    return INDEX_HTML


def open_in_browser(path: Path | str) -> None:
    """用系统默认浏览器打开一个本地 HTML 文件（Windows os.startfile；其他平台 webbrowser）。"""
    path = Path(path)
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            import webbrowser
            webbrowser.open(path.resolve().as_uri())
        log.info("已在浏览器打开：%s", path)
    except Exception as e:
        log.warning("自动打开浏览器失败：%s；请手动访问：%s", e, path)


if __name__ == "__main__":
    import sys

    if "--top10" in sys.argv:
        generate_weekly_top10()
    elif "--html" in sys.argv:
        render_html_all(open_browser="--open" in sys.argv)
    elif "--all" in sys.argv:
        generate_weekly_top10()
        render_html_all(open_browser="--open" in sys.argv)
    else:
        print("usage: python src/reporter.py [--top10 | --html | --all] [--open]")
