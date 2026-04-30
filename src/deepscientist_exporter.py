"""Export a DeepScientist-ready literature bundle.

This module turns the weekly outputs of smart-literature-agent into a compact
research handoff package for DeepScientist. It does not call any LLM. It only
collects the latest candidates, generated summaries, formula notes, and scores.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from reporter import composite_score
from utils import DATA_DIR, OUTPUT_DIR, get_logger, load_keywords

log = get_logger("deepscientist_exporter")

PAPERS_DIR = OUTPUT_DIR / "papers"
REPORTS_DIR = OUTPUT_DIR / "reports"
DEFAULT_OUT_DIR = OUTPUT_DIR / "deepscientist_bundle"

DEFAULT_RESEARCH_CONTEXT = (
    "轴承故障诊断 R1 路线：CWRU 主 + PU 跨域 + FA-KD + 结构化剪枝，"
    "目标是在边缘设备上获得高精度、可压缩、可迁移的故障诊断模型。"
)

LOW_QUALITY_SUMMARY_MARKERS = (
    "论文内容缺失",
    "正文内容为空",
    "无法生成摘要",
    "无法根据实际内容生成",
    "请提供完整的论文正文",
    "抱歉",
)


def _latest_candidates_path() -> Path:
    files = sorted(DATA_DIR.glob("candidates_*.json"))
    if not files:
        raise FileNotFoundError("未找到 data/candidates_*.json，请先运行 search 阶段。")
    return files[-1]


def _safe_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def _summary_path(arxiv_id: str) -> Path:
    return PAPERS_DIR / f"{_safe_id(arxiv_id)}.summary.md"


def _formulas_path(arxiv_id: str) -> Path:
    return PAPERS_DIR / f"{_safe_id(arxiv_id)}.formulas.md"


def _paper_json_path(arxiv_id: str) -> Path:
    return PAPERS_DIR / f"{_safe_id(arxiv_id)}.json"


def _read_text(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def _strip_md_heading(text: str) -> str:
    return re.sub(r"^#+\s+", "", text.strip())


def _section(md: str, heading: str) -> str:
    """Extract a level-2 Markdown section by exact Chinese heading text."""
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s+|\Z)",
        re.MULTILINE,
    )
    match = pattern.search(md)
    return match.group(1).strip() if match else ""


def _first_heading(md: str) -> str:
    match = re.search(r"^#\s+(.+)$", md, re.MULTILINE)
    return _strip_md_heading(match.group(1)) if match else "未命名论文"


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("---"):
            return re.sub(r"\s+", " ", line)
    return ""


def _formula_counts(arxiv_id: str) -> dict[str, int]:
    path = PAPERS_DIR / f"{_safe_id(arxiv_id)}.formulas.json"
    if not path.exists():
        return {"total": 0, "display": 0, "inline": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        counts = data.get("counts") or {}
        return {
            "total": int(counts.get("total") or 0),
            "display": int(counts.get("display") or 0),
            "inline": int(counts.get("inline") or 0),
        }
    except (OSError, json.JSONDecodeError, ValueError):
        return {"total": 0, "display": 0, "inline": 0}


def _summary_is_usable(arxiv_id: str) -> bool:
    summary = _read_text(_summary_path(arxiv_id))
    if len(summary.strip()) < 500:
        return False
    return not any(marker in summary for marker in LOW_QUALITY_SUMMARY_MARKERS)


def _top_papers(candidates: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    fields_config = load_keywords().get("fields", {})
    scored: list[dict[str, Any]] = []
    for paper in candidates.get("papers", []):
        arxiv_id = paper.get("arxiv_id")
        if not arxiv_id or not _summary_path(arxiv_id).exists():
            continue
        if not _summary_is_usable(arxiv_id):
            continue
        score = composite_score(paper, fields_config)
        item = dict(paper)
        item["_score_detail"] = score
        scored.append(item)
    scored.sort(key=lambda p: p["_score_detail"]["composite"], reverse=True)
    return scored[:top_n]


def _paper_record(paper: dict[str, Any]) -> dict[str, Any]:
    arxiv_id = paper["arxiv_id"]
    summary_md = _read_text(_summary_path(arxiv_id))
    route = _section(summary_md, "可迁移技术路线")
    formula_counts = _formula_counts(arxiv_id)
    return {
        "arxiv_id": arxiv_id,
        "title": paper.get("title"),
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        "score": paper.get("_score_detail"),
        "fields": paper.get("_fields", []),
        "matched_keywords": paper.get("_matched_keywords", []),
        "summary_title": _first_heading(summary_md),
        "one_line": _first_nonempty_line(_section(summary_md, "一句话结论")),
        "transfer_route_excerpt": route[:1600],
        "formula_counts": formula_counts,
        "local_files": {
            "paper_json": str(_paper_json_path(arxiv_id).resolve()),
            "summary_md": str(_summary_path(arxiv_id).resolve()),
            "formulas_md": str(_formulas_path(arxiv_id).resolve()) if _formulas_path(arxiv_id).exists() else None,
        },
    }


def _write_literature_brief(
    out_dir: Path,
    records: list[dict[str, Any]],
    candidates_path: Path,
    research_context: str,
) -> Path:
    lines = [
        "# DeepScientist Literature Brief",
        "",
        f"> Generated at: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"> Candidate pool: `{candidates_path.name}`",
        f"> Research context: {research_context}",
        "",
        "## Selected Papers",
        "",
    ]
    for i, rec in enumerate(records, 1):
        score = rec.get("score") or {}
        counts = rec.get("formula_counts") or {}
        lines.extend(
            [
                f"### {i}. {rec['title']}",
                "",
                f"- arXiv: [{rec['arxiv_id']}]({rec['arxiv_url']})",
                f"- Composite score: {score.get('composite', 0):.1f}",
                f"- Fields: {', '.join(rec.get('fields') or []) or 'unknown'}",
                f"- Formula coverage: display={counts.get('display', 0)}, total={counts.get('total', 0)}",
                f"- One-line conclusion: {rec.get('one_line') or '未抽取到'}",
                "- Local files:",
                f"  - summary: `{rec['local_files']['summary_md']}`",
                f"  - formulas: `{rec['local_files']['formulas_md'] or 'N/A'}`",
                "",
            ]
        )
        route = rec.get("transfer_route_excerpt") or ""
        if route:
            lines.extend(["#### Transfer Route Excerpt", "", route.strip(), ""])
    path = out_dir / "literature_brief.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_hypotheses(out_dir: Path, records: list[dict[str, Any]], research_context: str) -> Path:
    lines = [
        "# Candidate Research Hypotheses for DeepScientist",
        "",
        f"Research context: {research_context}",
        "",
        "The following hypotheses are machine-prepared handoff items. Treat them as starting points, not established claims.",
        "",
    ]
    for i, rec in enumerate(records, 1):
        route = rec.get("transfer_route_excerpt") or ""
        lines.extend(
            [
                f"## H{i}: Adapt `{rec['arxiv_id']}` to bearing fault diagnosis",
                "",
                f"Source paper: {rec['title']}",
                "",
                "Working hypothesis:",
                "",
                (
                    f"The method idea from `{rec['arxiv_id']}` may improve the current bearing fault diagnosis "
                    "pipeline when converted into a constrained experiment against the existing CWRU/PU baseline."
                ),
                "",
                "Evidence from literature summary:",
                "",
                route.strip() or "No transfer-route section was found in the summary.",
                "",
                "Suggested validation:",
                "",
                "- Define a minimal baseline comparison on CWRU first.",
                "- Run a PU cross-domain validation only after the CWRU result is non-trivial.",
                "- Track accuracy, macro-F1, parameter count, FLOPs/latency, and compression ratio.",
                "- Separate the paper's original claims from the new fault-diagnosis transfer hypothesis.",
                "",
            ]
        )
    path = out_dir / "hypotheses.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_startup_prompt(
    out_dir: Path,
    records: list[dict[str, Any]],
    research_context: str,
    baseline_path: str | None,
) -> Path:
    lines = [
        "# DeepScientist Startup Prompt",
        "",
        "You are starting a DeepScientist research quest from a curated literature bundle.",
        "",
        "## Primary Research Goal",
        "",
        research_context,
        "",
    ]
    if baseline_path:
        lines.extend(["## Baseline Code", "", f"Use this local baseline path as the main implementation context: `{baseline_path}`", ""])

    lines.extend(
        [
            "## Literature Inputs",
            "",
            "Use the following local files as review material. Prioritize the top 3 papers first; do not overfit to unrelated papers.",
            "",
        ]
    )
    for i, rec in enumerate(records, 1):
        lines.extend(
            [
                f"### {i}. {rec['arxiv_id']} - {rec['title']}",
                f"- Summary: `{rec['local_files']['summary_md']}`",
                f"- Formulas: `{rec['local_files']['formulas_md'] or 'N/A'}`",
                f"- arXiv: {rec['arxiv_url']}",
                f"- One-line: {rec.get('one_line') or 'N/A'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Requested DeepScientist Behavior",
            "",
            "1. Read the selected summaries and formula notes.",
            "2. Extract 2-4 experimentally testable hypotheses for the bearing fault diagnosis baseline.",
            "3. Prefer small, verifiable experiments before broad architectural changes.",
            "4. Keep original paper claims separate from transfer hypotheses.",
            "5. Report every result with dataset split, seed, metric, and changed files.",
            "",
            "## Constraints",
            "",
            "- Main target: bearing fault diagnosis with model compression / knowledge distillation / edge deployment.",
            "- Preferred validation order: CWRU first, then PU cross-domain.",
            "- Favor changes that can be ablated against the current baseline.",
            "- Avoid adding heavy dependencies unless the experimental gain justifies them.",
            "",
        ]
    )
    path = out_dir / "startup_prompt.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_bundle(
    top_n: int = 5,
    out_dir: Path = DEFAULT_OUT_DIR,
    research_context: str = DEFAULT_RESEARCH_CONTEXT,
    baseline_path: str | None = None,
) -> dict[str, Any]:
    candidates_path = _latest_candidates_path()
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    papers = _top_papers(candidates, top_n)
    if not papers:
        raise RuntimeError("没有可导出的论文：请先生成至少一篇 .summary.md。")

    out_dir.mkdir(parents=True, exist_ok=True)
    records = [_paper_record(p) for p in papers]

    candidate_path = out_dir / "candidate_papers.json"
    candidate_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    brief_path = _write_literature_brief(out_dir, records, candidates_path, research_context)
    hypotheses_path = _write_hypotheses(out_dir, records, research_context)
    startup_path = _write_startup_prompt(out_dir, records, research_context, baseline_path)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "top_n": len(records),
        "source_candidates": str(candidates_path.resolve()),
        "research_context": research_context,
        "baseline_path": baseline_path,
        "files": {
            "candidate_papers": str(candidate_path.resolve()),
            "literature_brief": str(brief_path.resolve()),
            "hypotheses": str(hypotheses_path.resolve()),
            "startup_prompt": str(startup_path.resolve()),
        },
        "papers": [r["arxiv_id"] for r in records],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("DeepScientist bundle exported: %s", out_dir.resolve())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DeepScientist literature bundle")
    parser.add_argument("--top-n", type=int, default=5, help="导出综合评分最高的 N 篇已摘要论文")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出目录")
    parser.add_argument("--research-context", default=DEFAULT_RESEARCH_CONTEXT, help="DeepScientist 研究目标上下文")
    parser.add_argument("--baseline-path", default=None, help="可选：DeepScientist 要使用的本地 baseline 代码路径")
    args = parser.parse_args()

    manifest = export_bundle(
        top_n=args.top_n,
        out_dir=args.out_dir,
        research_context=args.research_context,
        baseline_path=args.baseline_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
