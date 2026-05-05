"""smart-literature-agent 一键流水线，受 start.py 启动器统一调度。

用法（推荐通过 python start.py 交互式运行）：
  python src/run.py                           # 一次性增量运行（默认 --max-read 10）
  python src/run.py --max-read 20             # 本轮多读几篇
  python src/run.py --retry-failed            # 重试 failed_ids 里未 ingest 的论文
  python src/run.py --skip-search             # 跳过 search，复用最新 candidates（省 API）
  python src/run.py --no-llm                  # 只 search + read，不调 LLM 省 token
  python src/run.py --field knowledge_distillation  # 只对指定领域生成综述
  python src/run.py --historical              # 一次性回溯 5 年建历史池，之后每次自动取 30% 历史论文
  python src/run.py --historical-ratio 0.5    # 本轮从历史池取 50%（覆盖默认 30%）

六阶段：
  [1] search    ：遍历 6 领域 × N 关键词，落盘 data/candidates_<YYYYMMDD>.json
  [2] read      ：按 seen_ids 增量 + --max-read 上限，逐篇 full_read（4 种策略自动）
  [3] formulas  ：从 arXiv 源码提取数学公式
  [4] summarize ：对所有 output/papers/*.json 里没 summary 的补摘要 + 领域综述
  [5] report    ：综合评分 → TOP10 合并报告 + HTML 渲染
  [6] synthesis ：对 TOP 论文做跨论文交叉对比，输出模块融合创新分析
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from formula_handler import extract as extract_formulas, save_formulas
from reader import FAILED_IDS_PATH, full_read
from reporter import generate_weekly_top10, render_html_all
from searcher import build_historical_pool, search_all_fields
from summarizer import (
    PAPERS_DIR,
    generate_field_report,
    load_paper,
    summarize_single_paper,
)
from synthesizer import synthesize_top_papers
from utils import DATA_DIR, get_logger, load_keywords, load_seen_ids

log = get_logger("runner")

DEFAULT_MAX_READ = 10


def _latest_candidates() -> dict | None:
    files = sorted(DATA_DIR.glob("candidates_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _has_summary(arxiv_id: str) -> bool:
    safe_id = arxiv_id.replace("/", "_")
    return (PAPERS_DIR / f"{safe_id}.summary.md").exists()


def _has_paper_json(arxiv_id: str) -> bool:
    safe_id = arxiv_id.replace("/", "_")
    return (PAPERS_DIR / f"{safe_id}.json").exists()


def _load_failed_ids() -> dict[str, str]:
    if not FAILED_IDS_PATH.exists():
        return {}
    try:
        return json.loads(FAILED_IDS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _remove_from_failed(arxiv_id: str) -> None:
    existing = _load_failed_ids()
    if arxiv_id in existing:
        existing.pop(arxiv_id)
        FAILED_IDS_PATH.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _has_formulas(arxiv_id: str) -> bool:
    safe_id = arxiv_id.replace("/", "_")
    return (PAPERS_DIR / f"{safe_id}.formulas.json").exists()


def _pick_historical(weekly_ids: set[str], count: int) -> list[dict]:
    """从历史论文池中取 count 篇最佳未读论文。

    排除：已在每周候选中的 / 已在 seen_ids 中的。
    返回按 DeepXiv score 降序的论文列表。
    """
    pool_path = DATA_DIR / "candidates_historical_pool.json"
    if not pool_path.exists():
        return []
    seen = load_seen_ids()
    data = json.loads(pool_path.read_text(encoding="utf-8"))
    candidates = []
    for p in data.get("papers", []):
        aid = p.get("arxiv_id")
        if not aid:
            continue
        if aid in weekly_ids or aid in seen:
            continue
        candidates.append(p)
    # 按 DeepXiv score 降序
    candidates.sort(key=lambda p: float(p.get("score", 0) or 0), reverse=True)
    picks = candidates[:count]
    if picks:
        log.info(
            "从历史池取 %d/%d 篇（池总 %d，已排除 seen/weekly，剩余 %d 未读）",
            len(picks), count, data.get("total", 0),
            len(candidates) - len(picks),
        )
    else:
        log.info("历史池无可取论文（全部已读或在本次候选内）")
    return picks


def pipeline_run(
    max_read: int = DEFAULT_MAX_READ,
    retry_failed: bool = False,
    skip_search: bool = False,
    skip_llm: bool = False,
    field_filter: str | None = None,
    top_n: int = 10,
    auto_open: bool = True,
    skip_report: bool = False,
    skip_formulas: bool = False,
    skip_synthesis: bool = False,
    regen_summary: bool = False,
    historical_ratio: float | None = None,
) -> dict:
    """一次完整的增量运行，返回统计信息。"""
    started = datetime.now()
    log.info("=" * 70)
    log.info("PIPELINE RUN START @ %s", started.isoformat(timespec="seconds"))
    log.info(
        "  max_read=%d retry_failed=%s skip_search=%s skip_llm=%s skip_synthesis=%s regen_summary=%s field=%s hist_ratio=%s",
        max_read, retry_failed, skip_search, skip_llm, skip_synthesis, regen_summary, field_filter, historical_ratio,
    )
    stats: dict = {"started_at": started.isoformat(timespec="seconds")}

    # ---------- 1/6 search ----------
    if skip_search:
        data = _latest_candidates()
        if not data:
            log.error("  没有历史 candidates 可复用，--skip-search 无法继续")
            return stats
        log.info("[1/6 search] 跳过，复用最新 candidates（total=%d）", data.get("total", 0))
    else:
        log.info("[1/6 search] 批量检索 6 领域 × 多关键词")
        search_all_fields(save=True)
        data = _latest_candidates()

    all_papers = data.get("papers", [])

    # --- 历史池混入 ---
    hist_ratio = historical_ratio
    if hist_ratio is None:
        hist_ratio = float(
            load_keywords().get("search_config", {}).get("historical_read_ratio", 0)
        )
    hist_count = int(max_read * hist_ratio)
    if hist_count > 0:
        weekly_ids = {p["arxiv_id"] for p in all_papers}
        hist_picks = _pick_historical(weekly_ids, hist_count)
        if hist_picks:
            # 给历史论文标记来源
            for p in hist_picks:
                p.setdefault("_fields", [])
                p.setdefault("_matched_keywords", [])
                p["_from_historical"] = True
            all_papers = all_papers + hist_picks
            stats["historical_picks"] = len(hist_picks)
        else:
            stats["historical_picks"] = 0
    else:
        stats["historical_picks"] = 0

    field_index: dict[str, list[str]] = {}
    for p in all_papers:
        for f in p.get("_fields", []):
            field_index.setdefault(f, []).append(p["arxiv_id"])
    stats["search_total"] = len(all_papers)

    # ---------- 2/6 read ----------
    log.info("[2/6 read] 计算增量")
    seen = load_seen_ids()
    failed = _load_failed_ids()

    to_read: list[str] = []
    skipped_seen = skipped_failed = 0
    for p in all_papers:
        aid = p["arxiv_id"]
        if _has_paper_json(aid) and aid in seen:
            skipped_seen += 1
            continue
        if aid in failed and not retry_failed:
            skipped_failed += 1
            continue
        to_read.append(aid)

    clipped = len(to_read) > max_read
    to_read = to_read[:max_read]
    log.info(
        "  跳过 seen=%d / 跳过 failed=%d / 本轮 read=%d%s",
        skipped_seen, skipped_failed, len(to_read),
        f"（达到 {max_read} 篇上限，本次余下候选留待下次处理）" if clipped else "",
    )

    read_success = read_metaonly = read_failed = 0
    for aid in to_read:
        try:
            result = full_read(aid)
            strategy = result.get("strategy")
            if strategy == "failed":
                read_failed += 1
                log.warning("  %s 读取失败，已记入 failed_ids", aid)
            elif strategy == "metadata_only":
                read_metaonly += 1
                log.info("  %s metadata_only（DeepXiv 未 ingest 全文）", aid)
            else:
                read_success += 1
                _remove_from_failed(aid)
        except Exception as e:
            read_failed += 1
            log.exception("  %s 读取异常：%s", aid, e)

    log.info(
        "  本轮 read：成功=%d / metadata_only=%d / 失败=%d",
        read_success, read_metaonly, read_failed,
    )
    stats.update(
        read_success=read_success,
        read_metaonly=read_metaonly,
        read_failed=read_failed,
        skipped_seen=skipped_seen,
        skipped_failed=skipped_failed,
    )

    # ---------- 3/6 formulas：先提公式，摘要阶段才能解释关键公式 ----------
    if skip_formulas:
        log.info("[3/6 formulas] 跳过（--skip-formulas）")
        stats["formulas_extracted"] = 0
    else:
        log.info("[3/6 formulas] 从 arXiv 源码提取公式")
        formulas_done = 0
        formulas_fail = 0
        for paper_json in sorted(PAPERS_DIR.glob("*.json")):
            # 只处理原始精读 JSON（跳过 .formulas.json 和 .summary.md）
            if paper_json.name.endswith(".formulas.json"):
                continue
            aid = paper_json.stem
            if _has_formulas(aid):
                continue
            try:
                fs = extract_formulas(aid)
                save_formulas(aid, fs, {"type": "arxiv_latex"})
                formulas_done += 1
            except NotImplementedError:
                pass  # Phase 2 stubs
            except Exception as e:
                formulas_fail += 1
                log.warning("  formula extract(%s) 失败：%s", aid, e)
        log.info("  本轮新提取公式：%d 篇（失败 %d）", formulas_done, formulas_fail)
        stats["formulas_extracted"] = formulas_done
        stats["formulas_failed"] = formulas_fail

    if skip_llm:
        log.info("[4/6 llm] 跳过（--no-llm）")
        stats["summarized"] = 0
        stats["reports_generated"] = 0
    else:
        # ---------- 4/6 summarize + field_report ----------
        _run_llm_stage(stats, field_index, field_filter, regen_summary=regen_summary)

    # ---------- 5/6 report：TOP10 + HTML + 自动打开 ----------
    r = None
    if skip_report:
        log.info("[5/6 report] 跳过（--skip-report）")
    else:
        log.info("[5/6 report] 综合评分 + TOP%d + HTML 渲染", top_n)
        try:
            r = generate_weekly_top10(top_n=top_n)
            stats["top_n_generated"] = r["top_n"]
        except RuntimeError as e:
            log.warning("  weekly_top%d 跳过：%s", top_n, e)
            r = None
            stats["top_n_generated"] = 0
        try:
            index_html = render_html_all(open_browser=auto_open)
            stats["index_html"] = str(index_html)
        except Exception as e:
            log.warning("  HTML 渲染失败：%s", e)

    # ---------- 6/6 synthesis：跨论文综合创新分析 ----------
    if skip_synthesis or skip_llm:
        log.info("[6/6 synthesis] 跳过（%s）", "--skip-synthesis" if skip_synthesis else "--no-llm")
        stats["synthesis_generated"] = False
    elif stats.get("top_n_generated", 0) >= 2 and r is not None:
        log.info("[6/6 synthesis] 对 TOP%d 篇论文做跨论文交叉对比", r["top_n"])
        try:
            fields_config = load_keywords().get("fields", {})
            syn = synthesize_top_papers(
                arxiv_ids=r["arxiv_ids"],
                scores=r["scores"],
                fields_config=fields_config,
            )
            stats["synthesis_saved_to"] = syn.get("saved_to")
            stats["synthesis_analyzed_count"] = syn["analyzed_count"]
            stats["synthesis_generated"] = True
        except Exception as e:
            log.warning("  synthesis 跳过：%s", e)
            stats["synthesis_generated"] = False
    else:
        log.info("[6/6 synthesis] 跳过（TOP 论文不足 2 篇）")
        stats["synthesis_generated"] = False

    ended = datetime.now()
    stats["ended_at"] = ended.isoformat(timespec="seconds")
    stats["elapsed_sec"] = (ended - started).seconds
    log.info("PIPELINE RUN END @ %s 耗时 %ss", stats["ended_at"], stats["elapsed_sec"])
    log.info("=" * 70)
    return stats


def _run_llm_stage(
    stats: dict,
    field_index: dict[str, list[str]],
    field_filter: str | None,
    regen_summary: bool = False,
) -> None:
    """原 pipeline 摘要/综述阶段，现已为 [4/6] 阶段。"""
    log.info("[4/6 llm] 生成摘要和综述%s", "（强制重写摘要）" if regen_summary else "")

    # 3a: 对所有已 read 但没 summary 的论文补摘要
    summarized = 0
    for paper_json in sorted(PAPERS_DIR.glob("*.json")):
        aid = paper_json.stem
        if _has_summary(aid) and not regen_summary:
            continue
        try:
            paper = load_paper(aid)
            summarize_single_paper(paper)
            summarized += 1
        except Exception as e:
            log.warning("  summarize(%s) 失败：%s", aid, e)
    log.info("  本轮新生成摘要：%d 篇", summarized)
    stats["summarized"] = summarized

    # 3b: 对每个领域生成综述（至少 2 篇可用摘要）
    cfg = load_keywords()
    fields_all = list(cfg.get("fields", {}).keys())
    field_keys = [field_filter] if field_filter else fields_all
    reports_generated = 0
    for fk in field_keys:
        if fk not in fields_all:
            log.warning("  未知 field: %s", fk)
            continue
        fids = [aid for aid in field_index.get(fk, []) if _has_summary(aid)]
        if len(fids) < 2:
            log.info("  [%s] 可用摘要=%d，跳过综述（需 ≥ 2）", fk, len(fids))
            continue
        try:
            generate_field_report(fk, fids)
            reports_generated += 1
        except Exception as e:
            log.warning("  field_report(%s) 失败：%s", fk, e)
    log.info("  本轮生成领域报告：%d 份", reports_generated)
    stats["reports_generated"] = reports_generated


def main():
    parser = argparse.ArgumentParser(description="smart-literature-agent 一键流水线")
    parser.add_argument("--max-read", type=int, default=DEFAULT_MAX_READ,
                        help=f"本轮最多精读多少篇（默认 {DEFAULT_MAX_READ}）")
    parser.add_argument("--retry-failed", action="store_true",
                        help="重试 failed_ids 里的论文（默认跳过）")
    parser.add_argument("--skip-search", action="store_true",
                        help="跳过 search，复用最新 candidates")
    parser.add_argument("--no-llm", action="store_true",
                        help="只 search + read，不调 LLM")
    parser.add_argument("--field", dest="field_filter", default=None,
                        help="只对指定领域生成综述")
    parser.add_argument("--top-n", type=int, default=10,
                        help="TOP N 合并报告的 N（默认 10）")
    parser.add_argument("--no-open", action="store_true",
                        help="跑完后不自动打开浏览器")
    parser.add_argument("--skip-report", action="store_true",
                        help="跳过 TOP10 + HTML 渲染阶段（只跑 search/read/summary）")
    parser.add_argument("--skip-formulas", action="store_true",
                        help="跳过从 arXiv 源码提取公式的阶段")
    parser.add_argument("--skip-synthesis", action="store_true",
                        help="跳过跨论文综合创新分析阶段")
    parser.add_argument("--regen-summary", action="store_true",
                        help="强制重写已有 .summary.md（会重新调用 LLM）")
    parser.add_argument("--historical", action="store_true",
                        help="一次性回溯 5 年构建历史论文池（建完继续正常 pipeline）")
    parser.add_argument("--historical-only", action="store_true",
                        help="只构建历史池，不跑后续 pipeline")
    parser.add_argument("--historical-ratio", type=float, default=None,
                        help="本轮从历史池取的比例（0-1，默认从 keywords.yaml 读取 0.3）")
    args = parser.parse_args()

    # --historical：先构建历史池
    if args.historical:
        log.info("开始构建历史论文池（回溯 %d 年）...",
                 load_keywords().get("search_config", {}).get("historical_lookback_years", 5))
        build_historical_pool()
        if args.historical_only:
            log.info("历史池构建完成，--historical-only 模式退出")
            return

    kwargs = dict(
        max_read=args.max_read,
        retry_failed=args.retry_failed,
        skip_search=args.skip_search,
        skip_llm=args.no_llm,
        field_filter=args.field_filter,
        top_n=args.top_n,
        auto_open=not args.no_open,
        skip_report=args.skip_report,
        skip_formulas=args.skip_formulas,
        skip_synthesis=args.skip_synthesis,
        regen_summary=args.regen_summary,
        historical_ratio=args.historical_ratio,
    )

    pipeline_run(**kwargs)


if __name__ == "__main__":
    main()
