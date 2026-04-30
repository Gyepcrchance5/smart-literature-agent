"""OpenAlex 增强：对 arXiv 候选调 OpenAlex API 补充 venue / citation / 作者信号。

OpenAlex (https://openalex.org) 是 Microsoft Academic 停服后的开放学术数据库，
免费、免 API key、覆盖 2.5 亿+ 论文。我们用它回答一个核心问题：
  "这篇 arXiv 论文是否已被 peer-reviewed venue 接收？venue 档次如何？"

查询方式：arXiv ID → OpenAlex work 记录
  URL: https://api.openalex.org/works/doi:10.48550/arXiv.<arxiv_id>
  arXiv 2022 以后的论文都有 10.48550/arXiv.xxx 这个 DOI。

提取的关键字段：
  venue_type            journal / conference / repository（repository 表示仅预印本）
  venue_name            发表 venue 名字
  venue_h_index         venue 的 h-index（越高越顶）
  venue_works_count     venue 发文总数（辅助信号）
  cited_by_count        更新的引用数（DeepXiv 的可能延迟）
  authors               作者名列表（前 5）
  openalex_id           OpenAlex work ID

Rate limit：匿名 10 req/s；带 mailto 进 polite pool（更稳，建议设置）。
"""
from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

from utils import DATA_DIR, get_logger

log = get_logger("enricher")

_API_BASE = "https://api.openalex.org/works"
CACHE_PATH = DATA_DIR / "openalex_cache.json"
DEFAULT_SLEEP_MS = 100  # 每次调用后 sleep，polite

# 单例 cache（只在第一次 load）
_CACHE: dict | None = None


def _load_cache() -> dict:
    global _CACHE
    if _CACHE is None:
        if CACHE_PATH.exists():
            try:
                _CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("openalex_cache 损坏，重建空 cache")
                _CACHE = {}
        else:
            _CACHE = {}
    return _CACHE


def _save_cache() -> None:
    if _CACHE is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(_CACHE, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _request_params() -> dict:
    """OpenAlex 的 polite pool 建议带 mailto。从 env 读，未设置就走普通池。"""
    mailto = os.environ.get("OPENALEX_MAILTO")
    return {"mailto": mailto} if mailto else {}


def _fetch_by_arxiv(arxiv_id: str, timeout: int = 15) -> dict | None:
    """用 arXiv DOI 查 OpenAlex。404 返回 None，其他错误 raise。

    注意：arXiv DOI（10.48550/arXiv.xxx）是 2022 后才注册的，老论文查不到。
    查不到时调用方应 fallback 到 title search。
    """
    doi = f"10.48550/arXiv.{arxiv_id}"
    url = f"{_API_BASE}/doi:{doi}"
    headers = {"User-Agent": "smart-literature-agent (https://github.com/)"}
    r = requests.get(url, params=_request_params(), headers=headers, timeout=timeout)
    if r.status_code == 404:
        return None
    if r.status_code == 429:
        # 被限流，等 5 秒再 raise，让上层重试或跳过
        time.sleep(5)
        r.raise_for_status()
    r.raise_for_status()
    return r.json()


def _normalize_title(s: str) -> str:
    """把 title 压平：小写、去标点、压多空格，便于相似度比对。"""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _search_by_title(title: str, expected_title: str | None = None, per_page: int = 5, timeout: int = 15) -> list[dict]:
    """用 title 搜 OpenAlex，返回所有 title 相似度 ≥ 0.8 的 works。"""
    if not title:
        return []
    params = {
        **_request_params(),
        "search": title[:200],
        "per-page": per_page,
    }
    headers = {"User-Agent": "smart-literature-agent (https://github.com/)"}
    r = requests.get(_API_BASE, params=params, headers=headers, timeout=timeout)
    if r.status_code == 429:
        time.sleep(5)
        r.raise_for_status()
    r.raise_for_status()
    data = r.json()
    candidates = data.get("results") or []
    ref = _normalize_title(expected_title or title)
    matched = []
    for c in candidates:
        cand_title = _normalize_title(c.get("display_name") or "")
        score = SequenceMatcher(None, ref, cand_title).ratio()
        if score >= 0.8:
            matched.append((score, c))
    matched.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in matched]


def _fetch_by_title(title: str, expected_title: str | None = None) -> dict | None:
    """用 title 找最佳匹配的单条 work（兼容旧接口）。"""
    candidates = _search_by_title(title, expected_title)
    return candidates[0] if candidates else None


def _find_published_sibling(title: str, expected_title: str | None = None) -> dict | None:
    """给定一篇论文的 title，在 OpenAlex 里找同一论文的"会议/期刊版本"work。

    背景：OpenAlex 把 arXiv preprint 和 NeurIPS/ICLR 发表版当作两条独立 works
    （DOI 不同）。查 arXiv DOI 只拿到 preprint work（venue_type=repository），
    同一 title search 能额外找到 type=conference/journal 的 sibling work。
    """
    if not title:
        return None
    candidates = _search_by_title(title, expected_title, per_page=5)
    for c in candidates:
        src = (c.get("primary_location") or {}).get("source") or {}
        if src.get("type") in _PUBLISHED_TYPES:
            return c
    return None


_PUBLISHED_TYPES = ("journal", "conference", "book series", "ebook platform")


def _pick_published_location(work: dict) -> dict | None:
    """从 work.locations[] 中挑出"已发表"location（优先 conference/journal，避开 repository）。

    OpenAlex 对 arXiv 论文常把 primary_location 设成 arXiv 预印本，
    而把真正的 NeurIPS/CVPR/Nature 等发表记录藏在 locations 数组的其他项里。
    我们要找到那个真正能代表"peer-reviewed venue"的 location。
    """
    locs = work.get("locations") or []
    # 优先 is_published=True 的
    for loc in locs:
        src = loc.get("source") or {}
        if src.get("type") in _PUBLISHED_TYPES and loc.get("is_published"):
            return loc
    # 次选：任何 type 是已发表类型的（不看 is_published flag）
    for loc in locs:
        src = loc.get("source") or {}
        if src.get("type") in _PUBLISHED_TYPES:
            return loc
    return None


def _extract_signals(work: dict | None, venue_work: dict | None = None) -> dict:
    """从 OpenAlex work JSON 提取我们评分要用的字段。

    citation/authors/topic 始终来自主 work（arXiv preprint 那条，信号最完整）。
    venue 信息优先级：
      1. venue_work（外部传入的"发表版 sibling"work 的 primary_location）
      2. 主 work.locations[] 里的 published location（同一 work 有多个 location）
      3. 主 work.primary_location（通常是 arXiv repository）
    """
    if not work:
        return {"hit": False}
    sig: dict[str, Any] = {"hit": True}
    sig["openalex_id"] = work.get("id")
    sig["cited_by_count"] = int(work.get("cited_by_count") or 0)

    venue_src = None
    if venue_work:
        venue_src = (venue_work.get("primary_location") or {}).get("source") or {}
    if not venue_src or not venue_src.get("type"):
        published = _pick_published_location(work)
        if published:
            venue_src = published.get("source") or {}
    if not venue_src or not venue_src.get("type"):
        venue_src = (work.get("primary_location") or {}).get("source") or {}
    src = venue_src or {}

    sig["venue_name"] = src.get("display_name")
    sig["venue_type"] = src.get("type")  # journal / conference / repository / ...
    sig["venue_host"] = src.get("host_organization_name")
    stats = src.get("summary_stats") or {}
    sig["venue_h_index"] = stats.get("h_index")
    sig["venue_works_count"] = src.get("works_count")
    # 作者名字（不查各自 h-index，太贵）
    auths = work.get("authorships") or []
    sig["authors"] = [
        (a.get("author") or {}).get("display_name")
        for a in auths[:8]
        if (a.get("author") or {}).get("display_name")
    ]
    # 主题（OpenAlex 自动分类，跟我们关键词不一定对上，做参考）
    pt = work.get("primary_topic") or {}
    sig["topic"] = pt.get("display_name")
    sig["topic_score"] = pt.get("score")
    return sig


def enrich_one(arxiv_id: str, title: str | None = None, use_cache: bool = True) -> dict:
    """查询单篇 arXiv 论文的 OpenAlex 元数据。

    策略：
    1. 先试 DOI `10.48550/arXiv.<arxiv_id>`（2022 后论文都有）
    2. 若 DOI 查不到且传了 title，用 title search + 相似度 ≥ 0.8 校验
    3. 主 work 如果 primary 是 repository（arXiv preprint），额外 title search
       找"发表版 sibling"（type=conference/journal 的 work），拿到真正 venue
    """
    cache = _load_cache()
    if use_cache and arxiv_id in cache:
        return cache[arxiv_id]
    try:
        work = _fetch_by_arxiv(arxiv_id)
        if work is None and title:
            log.info("  %s DOI 查不到，fallback 到 title search", arxiv_id)
            work = _fetch_by_title(title, expected_title=title)

        venue_sibling = None
        if work and title:
            primary_src = (work.get("primary_location") or {}).get("source") or {}
            if primary_src.get("type") == "repository":
                # 主 work 是 arXiv preprint，找同题 conference/journal 版本
                venue_sibling = _find_published_sibling(title, expected_title=title)
    except Exception as e:
        log.warning("OpenAlex query failed for %s: %s", arxiv_id, e)
        return {"hit": False, "error": str(e)}
    sig = _extract_signals(work, venue_work=venue_sibling)
    cache[arxiv_id] = sig
    return sig


def enrich_all(papers: list[dict], sleep_ms: int = DEFAULT_SLEEP_MS) -> dict:
    """批量对 papers 列表做 OpenAlex 增强。

    - 命中 cache 不走网络
    - 新查的每次 sleep_ms 控制 rate
    - 就地把 _openalex 字段写回每个 paper
    - 返回统计 {total, cache_hits, new_queries, hit_rate}
    """
    cache = _load_cache()
    total = len(papers)
    cache_hits = 0
    new_queries = 0
    hit_rate_num = 0  # OpenAlex 返回非空的次数

    for i, p in enumerate(papers, 1):
        aid = p.get("arxiv_id")
        if not aid:
            continue
        if aid in cache:
            sig = cache[aid]
            cache_hits += 1
        else:
            sig = enrich_one(aid, title=p.get("title"), use_cache=False)
            new_queries += 1
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000)
            if new_queries % 20 == 0:
                log.info("  OpenAlex 进度 %d/%d（new=%d, cache=%d）",
                         i, total, new_queries, cache_hits)
                _save_cache()  # 中途定期存盘，断电也不丢

        if sig.get("hit"):
            hit_rate_num += 1
        p["_openalex"] = sig

    _save_cache()
    stats = {
        "total": total,
        "cache_hits": cache_hits,
        "new_queries": new_queries,
        "openalex_hit_rate": f"{hit_rate_num}/{total}",
    }
    log.info("OpenAlex 增强完成：%s", stats)
    return stats


# ============================================================
# Venue prestige scoring (供 reporter.composite_score 调用)
# ============================================================

# 预印本（repository）→ 低基线分
# 学术期刊 / 会议 → 按 h-index 归一化到 0-100
# 典型 h-index 参考：NeurIPS/ICML/CVPR 的 venue h-index 在 200+，中流会议 50-100，小众 <30
_VENUE_H_INDEX_CAP = 250


_VENUE_PUBLISHED_BASELINE = 50.0  # OpenAlex 大多数 venue 的 h_index 字段为 null，
                                   # 给 conference/journal 一个基线分避免被跟 repository 混为一谈


def venue_prestige_score(openalex_sig: dict | None) -> float:
    """从 OpenAlex signal 计算 venue prestige 分数（0-100）。

    打分梯度：
      - 未命中 OpenAlex / 无 venue 信息                → 0
      - repository（纯 arXiv 预印本，从未被发表）      → 10
      - conference/journal 但 h_index 字段缺失（常见） → 50
      - conference/journal 有 h_index                  → max(50, h/250*100)
      - 顶刊顶会（h ≥ 250）                            → 100 封顶

    公式说明：OpenAlex 的 source.summary_stats.h_index 字段在很多 venue 上缺失，
    我们不能把 "无 h_index 的 NeurIPS/ICLR/Nature" 错误地降到 0 分。
    """
    if not openalex_sig or not openalex_sig.get("hit"):
        return 0.0
    venue_type = openalex_sig.get("venue_type")
    if venue_type == "repository":
        return 10.0
    if venue_type in _PUBLISHED_TYPES:
        h = openalex_sig.get("venue_h_index")
        if not h:
            return _VENUE_PUBLISHED_BASELINE
        return max(_VENUE_PUBLISHED_BASELINE, min(h, _VENUE_H_INDEX_CAP) / _VENUE_H_INDEX_CAP * 100)
    return 0.0


if __name__ == "__main__":
    # 冒烟：对几个 arxiv id 跑一下
    import sys

    if len(sys.argv) > 1:
        for aid in sys.argv[1:]:
            sig = enrich_one(aid, use_cache=False)
            _save_cache()
            print(f"\n=== {aid} ===")
            print(json.dumps(sig, ensure_ascii=False, indent=2))
    else:
        # 默认测试一个知名 arxiv id（Attention is All You Need）
        sig = enrich_one("1706.03762", use_cache=False)
        _save_cache()
        print("=== 1706.03762 (Attention is All You Need) ===")
        print(json.dumps(sig, ensure_ascii=False, indent=2))
        print(f"\nVenue prestige score: {venue_prestige_score(sig):.1f}")
