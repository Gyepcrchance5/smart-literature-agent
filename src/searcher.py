"""论文检索模块：遍历关键词配置，调用 DeepXiv CLI 抓取候选论文。

DeepXiv search JSON 返回结构：
  { "status", "total_count", "result": [
      { "arxiv_id", "score", "title", "tldr", "abstract",
        "authors": [{"name", "orgs"}],
        "url", "date", "citation_count", "categories" }
  ]}

策略：
- 搜索级别 **不**用 seen_ids 去重（只做本轮内去重），避免遮蔽已知但未精读的论文
- 一轮内按 arxiv_id 去重：同一篇命中多个关键词时，把命中的关键词合并
- 按 min_relevance_score 过滤 DeepXiv 返回的浮点 score
- 按 lookback_days 计算 --date-from（YYYY-MM-DD）
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from utils import DATA_DIR, get_logger, load_keywords, run_deepxiv

log = get_logger("searcher")


def search_keyword(
    keyword: str,
    max_results: int = 10,
    date_from: str | None = None,
) -> list[dict]:
    """对单个关键词调用 deepxiv search，返回论文列表。"""
    args = ["search", keyword, "--limit", str(max_results)]
    if date_from:
        args += ["--date-from", date_from]
    data = run_deepxiv(args, parse_json=True)
    return data.get("result", []) if isinstance(data, dict) else []


def _calc_date_from(lookback_days: int) -> str:
    """根据回溯天数计算 --date-from 参数值（YYYY-MM-DD）。"""
    return (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")


def search_all_fields(save: bool = True) -> dict[str, list[dict]]:
    """遍历所有领域关键词，返回 {field_name: [papers]}。

    每篇论文会被增加两个额外字段：
      _fields: [命中的领域 key 列表]
      _matched_keywords: [命中的关键词列表]

    save=True 时落盘到 data/candidates_<YYYYMMDD>.json，便于增量调试。
    """
    cfg = load_keywords()
    search_cfg = cfg.get("search_config", {})
    max_per_kw = int(search_cfg.get("max_results_per_keyword", 10))
    lookback_days = int(search_cfg.get("lookback_days", 7))
    min_score = float(search_cfg.get("min_relevance_score", 0))
    date_from = _calc_date_from(lookback_days)

    log.info(
        "开始批量检索：领域=%d，每关键词上限=%d，回溯=%d 天（%s 起），分数阈值=%.2f",
        len(cfg["fields"]),
        max_per_kw,
        lookback_days,
        date_from,
        min_score,
    )

    # 全局 arxiv_id -> paper（首次命中时收录，再命中时合并 _fields/_matched_keywords）
    merged: dict[str, dict] = {}
    # field_name -> [arxiv_id]，记录每个领域命中了哪些
    field_index: dict[str, list[str]] = {}

    for field_key, field_conf in cfg["fields"].items():
        field_index.setdefault(field_key, [])
        for kw in field_conf.get("keywords", []):
            try:
                papers = search_keyword(kw, max_results=max_per_kw, date_from=date_from)
            except Exception as e:
                log.warning("搜索 [%s / %s] 失败：%s", field_key, kw, e)
                continue

            kept = 0
            for p in papers:
                score = float(p.get("score", 0) or 0)
                if score < min_score:
                    continue
                aid = p.get("arxiv_id")
                if not aid:
                    continue
                if aid not in merged:
                    p["_fields"] = [field_key]
                    p["_matched_keywords"] = [kw]
                    merged[aid] = p
                    field_index[field_key].append(aid)
                    kept += 1
                else:
                    if field_key not in merged[aid]["_fields"]:
                        merged[aid]["_fields"].append(field_key)
                        if aid not in field_index[field_key]:
                            field_index[field_key].append(aid)
                    if kw not in merged[aid]["_matched_keywords"]:
                        merged[aid]["_matched_keywords"].append(kw)
            log.info("  [%s / %s] 原始=%d，通过阈值=%d", field_key, kw, len(papers), kept)

    # 组装 field → papers 的返回结构
    results: dict[str, list[dict]] = {
        f: [merged[aid] for aid in ids] for f, ids in field_index.items()
    }

    log.info(
        "批量检索完成：去重后共 %d 篇，分布=%s",
        len(merged),
        {f: len(ps) for f, ps in results.items()},
    )

    if save:
        out = DATA_DIR / f"candidates_{datetime.now():%Y%m%d}.json"
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "date_from": date_from,
            "total": len(merged),
            "by_field": {f: len(ps) for f, ps in results.items()},
            "papers": list(merged.values()),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("候选结果已保存：%s", out)

    return results


if __name__ == "__main__":
    # 默认冒烟：单关键词
    # 真实全量跑：python src/searcher.py --all
    import sys

    if "--all" in sys.argv:
        search_all_fields(save=True)
    else:
        papers = search_keyword("knowledge distillation", max_results=2)
        log.info("冒烟测试：单关键词检索命中 %d 篇", len(papers))
        for p in papers:
            log.info("  - [%s] score=%.2f %s", p.get("arxiv_id"), p.get("score", 0), p.get("title"))
