"""报告增强模块：综合评分 + 本周 TOP10 合并报告。

综合评分权重（按用户设定）：
  启发式相关性 45% + DeepXiv search score 25% + Venue 档次 20% + 引用数 10%
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

from enricher import venue_prestige_score
from summarizer import score_relevance
from utils import DATA_DIR, PAPERS_DIR, REPORTS_DIR, get_logger, load_keywords, try_write_text

log = get_logger("reporter")

# 综合评分权重（V2：加入 venue_prestige 维度）
W_RELEVANCE = 0.45
W_DEEPXIV = 0.25
W_VENUE = 0.20
W_CITATION = 0.10

# 启发式相关性：6 领域 × 最高 5 分 = 30 为满分
_RELEVANCE_MAX = 6 * 5
# DeepXiv score 通常 3~6，截断到 10 归一化
_DEEPXIV_CAP = 10.0
# 引用数 log 归一化：log1p(999)/log(1000) ≈ 1.0（即引用 ~1000 即满分）
_CITATION_CAP = 1000


def composite_score(paper: dict, fields_config: dict | None = None) -> dict:
    """对 candidates.json 里的 paper entry 计算综合评分（V2 四维）。

    四维（归一到 0-100）：
      - relevance  启发式领域关键词命中
      - deepxiv    DeepXiv search score
      - venue      venue prestige（OpenAlex venue_h_index；预印本基线 10 分）
      - citation   引用数 log 归一化（优先 OpenAlex 的更新值，fallback DeepXiv）

    composite 按 W_RELEVANCE/W_DEEPXIV/W_VENUE/W_CITATION 加权求和（也是 0-100）。
    """
    if fields_config is None:
        fields_config = load_keywords().get("fields", {})

    rel_per_field = score_relevance(paper)  # {field: 1-5}
    rel_total = sum(rel_per_field.values())  # 最多 30
    rel_norm = rel_total / _RELEVANCE_MAX * 100

    deepxiv_raw = float(paper.get("score") or 0)
    deepxiv_norm = min(deepxiv_raw, _DEEPXIV_CAP) / _DEEPXIV_CAP * 100

    # OpenAlex venue prestige（来自 enricher）
    oa = paper.get("_openalex") or {}
    venue_norm = venue_prestige_score(oa)

    # 引用数：优先用 OpenAlex 的（更新更准），fallback DeepXiv
    cite_raw = int(oa.get("cited_by_count") or paper.get("citation_count") or 0)
    cite_norm = (
        min(math.log1p(cite_raw) / math.log(_CITATION_CAP), 1.0) * 100 if cite_raw > 0 else 0
    )

    composite = (
        W_RELEVANCE * rel_norm
        + W_DEEPXIV * deepxiv_norm
        + W_VENUE * venue_norm
        + W_CITATION * cite_norm
    )

    return {
        "composite": round(composite, 2),
        "relevance": round(rel_norm, 2),
        "deepxiv": round(deepxiv_norm, 2),
        "venue": round(venue_norm, 2),
        "citation": round(cite_norm, 2),
        "citation_count": cite_raw,
        "deepxiv_raw": round(deepxiv_raw, 3),
        "venue_type": oa.get("venue_type"),
        "venue_name": oa.get("venue_name"),
        "venue_h_index": oa.get("venue_h_index"),
        "breakdown": rel_per_field,
    }


def _venue_badge(score_dict: dict) -> str:
    """把 venue 信息渲染成 markdown 单元格里的一个短 badge。"""
    vtype = score_dict.get("venue_type")
    vname = score_dict.get("venue_name")
    vh = score_dict.get("venue_h_index")
    if not vtype:
        return "-"
    if vtype == "repository":
        return "arXiv"
    # 截断过长 venue 名字
    short = (vname or vtype)[:22] + ("…" if vname and len(vname) > 22 else "")
    return f"{short} (h={vh})" if vh else short


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
    lines = [
        f"# smart-literature-agent 本周 TOP {len(top)} 论文速递",
        "",
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}  ",
        f"> 候选池：`{Path(candidates_path).name}`（共 {len(papers)} 篇，时间窗 {data.get('date_from')} 起）  ",
        f"> 评分公式：启发式相关性 × {W_RELEVANCE:.0%} + DeepXiv 分 × {W_DEEPXIV:.0%} + Venue 档次 × {W_VENUE:.0%} + 引用数 × {W_CITATION:.0%}  ",
        "> 入选条件：已生成 .summary.md 的候选，按综合分降序",
        "",
        "## 综合排名",
        "",
        "| 排名 | arXiv | 标题 | 综合分 | 相关性 | DeepXiv | Venue | 引用 | 领域命中 |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | :--- | ---: | :--- |",
    ]
    for rank, (s, p) in enumerate(top, 1):
        aid = p["arxiv_id"]
        title = (p.get("title") or "").replace("|", "/")
        fields_hit = ", ".join(p.get("_fields") or [])
        venue_cell = _venue_badge(s)
        lines.append(
            f"| #{rank} | [{aid}](https://arxiv.org/abs/{aid}) | {title[:60]} "
            f"| {s['composite']:.1f} | {s['relevance']:.0f} | {s['deepxiv']:.0f} "
            f"| {venue_cell} | {s['citation_count']} | {fields_hit} |"
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
            f"（相关 {s['relevance']:.0f} / DeepXiv {s['deepxiv']:.0f}"
            f" / Venue {s['venue']:.0f} / 引用 {s['citation_count']}）"
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
        out = REPORTS_DIR / f"weekly_top{top_n}_{datetime.now():%Y%m%d}.md"
        saved = try_write_text(out, report_md, logger=log)
        if saved:
            log.info("本周 TOP %d 已保存：%s", top_n, saved)
            result["saved_to"] = str(saved)
        else:
            result["save_error"] = f"无法写入 {out}"

    return result


# ================================================================
# HTML 渲染
# ================================================================



if __name__ == "__main__":
    import sys

    if "--top10" in sys.argv:
        generate_weekly_top10()
    else:
        print("usage: python src/reporter.py [--top10]")
