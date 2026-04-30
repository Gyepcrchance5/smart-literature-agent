"""smart-literature-agent 一键流水线 + 每周一定时调度。

用法：
  python src/run.py                           # 一次性增量运行（默认 --max-read 10）
  python src/run.py --max-read 20             # 本轮多读几篇
  python src/run.py --retry-failed            # 重试 failed_ids 里未 ingest 的论文
  python src/run.py --skip-search             # 跳过 search，复用最新 candidates（省 API）
  python src/run.py --no-llm                  # 只 search + read，不调 LLM 省 token
  python src/run.py --field knowledge_distillation  # 只对指定领域生成综述
  python src/run.py --daemon                  # 常驻，每周一 08:07 自动运行
  python src/run.py --daemon --no-initial-run # 常驻但不立即跑，等到周一

三阶段：
  [1/3] search：遍历 6 领域 × N 关键词，落盘 data/candidates_<YYYYMMDD>.json
  [2/3] read  ：按 seen_ids 增量 + --max-read 上限，逐篇 full_read（4 种策略自动）
  [3/3] llm   ：对所有 output/papers/*.json 里没 summary 的补摘要，再对每领域（≥2 篇）综述
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime

import schedule

from reader import FAILED_IDS_PATH, full_read
from reporter import generate_weekly_top10, render_html_all
from searcher import search_all_fields
from summarizer import (
    PAPERS_DIR,
    generate_field_report,
    load_paper,
    summarize_single_paper,
)
from utils import DATA_DIR, get_logger, load_keywords, load_seen_ids

log = get_logger("runner")

DEFAULT_MAX_READ = 10
DEFAULT_SCHEDULE_TIME = "08:07"  # 周一早上 8:07，避开整点


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


def pipeline_run(
    max_read: int = DEFAULT_MAX_READ,
    retry_failed: bool = False,
    skip_search: bool = False,
    skip_llm: bool = False,
    field_filter: str | None = None,
    top_n: int = 10,
    auto_open: bool = True,
    skip_report: bool = False,
) -> dict:
    """一次完整的增量运行，返回统计信息。"""
    started = datetime.now()
    log.info("=" * 70)
    log.info("PIPELINE RUN START @ %s", started.isoformat(timespec="seconds"))
    log.info(
        "  max_read=%d retry_failed=%s skip_search=%s skip_llm=%s field=%s",
        max_read, retry_failed, skip_search, skip_llm, field_filter,
    )
    stats: dict = {"started_at": started.isoformat(timespec="seconds")}

    # ---------- 1/3 search ----------
    if skip_search:
        data = _latest_candidates()
        if not data:
            log.error("  没有历史 candidates 可复用，--skip-search 无法继续")
            return stats
        log.info("[1/3 search] 跳过，复用最新 candidates（total=%d）", data.get("total", 0))
    else:
        log.info("[1/3 search] 批量检索 6 领域 × 多关键词")
        search_all_fields(save=True)
        data = _latest_candidates()

    all_papers = data.get("papers", [])
    field_index: dict[str, list[str]] = {}
    for p in all_papers:
        for f in p.get("_fields", []):
            field_index.setdefault(f, []).append(p["arxiv_id"])
    stats["search_total"] = len(all_papers)

    # ---------- 2/3 read ----------
    log.info("[2/3 read] 计算增量")
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
        f"（{max_read} 上限后，剩 {skipped_failed + skipped_seen} 未读）" if clipped else "",
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

    if skip_llm:
        log.info("[3/4 llm] 跳过（--no-llm）")
        stats["summarized"] = 0
        stats["reports_generated"] = 0
    else:
        # ---------- 3/4 summarize + field_report ----------
        _run_llm_stage(stats, field_index, field_filter)

    # ---------- 4/4 report：TOP10 + HTML + 自动打开 ----------
    if skip_report:
        log.info("[4/4 report] 跳过（--skip-report）")
    else:
        log.info("[4/4 report] 综合评分 + TOP%d + HTML 渲染", top_n)
        try:
            r = generate_weekly_top10(top_n=top_n)
            stats["top_n_generated"] = r["top_n"]
        except RuntimeError as e:
            log.warning("  weekly_top%d 跳过：%s", top_n, e)
            stats["top_n_generated"] = 0
        try:
            index_html = render_html_all(open_browser=auto_open)
            stats["index_html"] = str(index_html)
        except Exception as e:
            log.warning("  HTML 渲染失败：%s", e)

    ended = datetime.now()
    stats["ended_at"] = ended.isoformat(timespec="seconds")
    stats["elapsed_sec"] = (ended - started).seconds
    log.info("PIPELINE RUN END @ %s 耗时 %ss", stats["ended_at"], stats["elapsed_sec"])
    log.info("=" * 70)
    return stats


def _run_llm_stage(stats: dict, field_index: dict[str, list[str]], field_filter: str | None) -> None:
    """原 3/3 阶段抽成函数，便于 skip_llm 分支整洁。"""
    log.info("[3/4 llm] 生成摘要和综述")

    # 3a: 对所有已 read 但没 summary 的论文补摘要
    summarized = 0
    for paper_json in sorted(PAPERS_DIR.glob("*.json")):
        aid = paper_json.stem
        if _has_summary(aid):
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


def run_daemon(initial_run: bool = True, **pipeline_kwargs) -> None:
    """常驻进程：每周一 08:07 跑一次。"""
    def job():
        try:
            pipeline_run(**pipeline_kwargs)
        except Exception as e:
            log.exception("定时任务出错：%s", e)

    schedule.every().monday.at(DEFAULT_SCHEDULE_TIME).do(job)
    log.info("已注册定时任务：每周一 %s（本地时间）", DEFAULT_SCHEDULE_TIME)
    next_run = schedule.next_run()
    if next_run:
        log.info("下次触发：%s", next_run.isoformat(timespec="seconds"))

    if initial_run:
        log.info("常驻启动，立即跑一次")
        job()

    log.info("进入常驻循环，Ctrl+C 退出...")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("收到 Ctrl+C，退出常驻模式")


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
    parser.add_argument("--daemon", action="store_true",
                        help="常驻，每周一 08:07 自动运行")
    parser.add_argument("--no-initial-run", action="store_true",
                        help="daemon 启动时不立即跑一次（默认立即跑）")
    parser.add_argument("--top-n", type=int, default=10,
                        help="TOP N 合并报告的 N（默认 10）")
    parser.add_argument("--no-open", action="store_true",
                        help="跑完后不自动打开浏览器")
    parser.add_argument("--skip-report", action="store_true",
                        help="跳过 TOP10 + HTML 渲染阶段（只跑 search/read/summary）")
    parser.add_argument("--verify-schedule", action="store_true",
                        help=argparse.SUPPRESS)  # 内部：注册调度后立即退出，用于冒烟
    args = parser.parse_args()

    kwargs = dict(
        max_read=args.max_read,
        retry_failed=args.retry_failed,
        skip_search=args.skip_search,
        skip_llm=args.no_llm,
        field_filter=args.field_filter,
        top_n=args.top_n,
        auto_open=not args.no_open,
        skip_report=args.skip_report,
    )

    if args.verify_schedule:
        # 只注册一下，报告下次执行时间后退出
        schedule.every().monday.at(DEFAULT_SCHEDULE_TIME).do(lambda: None)
        nr = schedule.next_run()
        print(f"schedule registered OK; next run at {nr}")
        return

    if args.daemon:
        run_daemon(initial_run=not args.no_initial_run, **kwargs)
    else:
        pipeline_run(**kwargs)


if __name__ == "__main__":
    main()
