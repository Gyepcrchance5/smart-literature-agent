"""论文精读模块：调用 DeepXiv CLI 逐章节阅读论文并组装为结构化数据。

DeepXiv paper 支持的关键 flag：
  --brief / -b    : 简报（title, TLDR, keywords, citations, GitHub URL）
  --head          : 元数据 + sections 列表（每个 section 有独立 TLDR 和 token_count）
  --preview / -p  : 预览（~10k 字符）
  --raw           : 完整原文 markdown
  --section NAME  : 指定章节正文（NAME 必须是 sections 列表里的真实名字）

full_read 的 token 预算策略：
  token_count <= 8000           → 全文 --raw
  8000 < token_count <= 20000   → 匹配关键 section（Introduction / Method / Experiments / Conclusion）
  token_count > 20000           → --preview（~10k 字符）+ brief
"""
from __future__ import annotations

import json

from utils import (
    DATA_DIR,
    OUTPUT_DIR,
    get_logger,
    load_seen_ids,
    run_deepxiv,
    save_seen_ids,
)

log = get_logger("reader")

PAPERS_DIR = OUTPUT_DIR / "papers"
FAILED_IDS_PATH = DATA_DIR / "failed_ids.json"

_SMALL_TOKEN_THRESHOLD = 8000
_MEDIUM_TOKEN_THRESHOLD = 20000

# 选关键章节时用的子串匹配（按优先级；不区分大小写）
_KEY_SECTION_GROUPS: list[list[str]] = [
    ["introduction"],
    ["method", "approach", "proposed"],
    ["experiment", "evaluation", "result"],
    ["conclusion", "discussion"],
]


def get_brief(arxiv_id: str) -> dict:
    """获取论文简报（title、TLDR、关键词、引用数）。"""
    return run_deepxiv(["paper", arxiv_id, "--brief"], parse_json=True)


def get_head(arxiv_id: str) -> dict:
    """获取论文元数据与 sections 结构。

    返回字段包含：arxiv_id, title, abstract, authors, token_count, venue,
    journal_name, citations, sections: [{name, idx, tldr, token_count}],
    categories, publish_at, keywords, tldr
    """
    return run_deepxiv(["paper", arxiv_id, "--head"], parse_json=True)


def read_section(arxiv_id: str, section_name: str) -> str:
    """读取指定章节正文（markdown）。section_name 必须是 head.sections[].name 的真实值。"""
    return run_deepxiv(["paper", arxiv_id, "--section", section_name], parse_json=False)


def _record_failed(arxiv_id: str, error: str) -> None:
    """把 ingest 失败的论文记到 data/failed_ids.json，方便后续重试。"""
    FAILED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(FAILED_IDS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}
    existing[arxiv_id] = error
    FAILED_IDS_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def _pick_key_sections(sections_meta: list[dict]) -> list[str]:
    """从 head 的 sections 里挑 Introduction/Method/Experiments/Conclusion 各一个。"""
    picked: list[str] = []
    picked_groups: set[int] = set()
    for s in sections_meta:
        name = s.get("name", "")
        name_l = name.lower()
        if any(skip in name_l for skip in ("appendix", "reference", "acknowledg")):
            continue
        for i, keywords in enumerate(_KEY_SECTION_GROUPS):
            if i in picked_groups:
                continue
            if any(k in name_l for k in keywords):
                picked.append(name)
                picked_groups.add(i)
                break
    return picked


def full_read(arxiv_id: str, save: bool = True, mark_seen: bool = True) -> dict:
    """完整精读：head → 按 token_count 选策略 → 组装结构化 dict。

    save=True：落盘到 output/papers/<arxiv_id>.json
    mark_seen=True：把 arxiv_id 加入 seen_ids.json（下游用此判断是否跳过）
    """
    log.info("精读论文：%s", arxiv_id)
    try:
        head = get_head(arxiv_id)
    except Exception as e:
        # DeepXiv 后端对部分论文只有元数据，没有全文 ingest；或论文刚上传还没处理完
        log.warning("get_head(%s) 失败（可能未被 DeepXiv 完整 ingest）：%s", arxiv_id, e)
        _record_failed(arxiv_id, str(e))
        return {"arxiv_id": arxiv_id, "strategy": "failed", "error": str(e)}

    if not isinstance(head, dict):
        err = f"get_head({arxiv_id}) 返回异常：{head!r}"
        _record_failed(arxiv_id, err)
        return {"arxiv_id": arxiv_id, "strategy": "failed", "error": err}

    token_count = int(head.get("token_count") or 0)
    sections_meta = head.get("sections", []) or []

    result: dict = {
        "arxiv_id": arxiv_id,
        "title": head.get("title"),
        "abstract": head.get("abstract"),
        "keywords": head.get("keywords"),
        "tldr": head.get("tldr"),
        "citations": head.get("citations"),
        "authors": head.get("authors"),
        "categories": head.get("categories"),
        "publish_at": head.get("publish_at"),
        "venue": head.get("venue"),
        "journal_name": head.get("journal_name"),
        "token_count": token_count,
        "section_meta": sections_meta,
        "strategy": None,
        "sections": {},
        "preview": None,
        "raw": None,
    }

    if token_count == 0:
        # head 有但全文未 ingest，跳过正文下载
        result["strategy"] = "metadata_only"
        log.info("  token_count=0，DeepXiv 尚未 ingest 全文，仅保存元数据")
    elif token_count <= _SMALL_TOKEN_THRESHOLD:
        result["strategy"] = "raw"
        try:
            result["raw"] = run_deepxiv(["paper", arxiv_id, "--raw"], parse_json=False)
        except Exception as e:
            log.warning("取 --raw 失败，尝试 --preview：%s", e)
            try:
                result["preview"] = run_deepxiv(["paper", arxiv_id, "--preview"], parse_json=False)
                result["strategy"] = "preview"
            except Exception as e2:
                log.warning("取 --preview 也失败，只保留元数据：%s", e2)
                result["strategy"] = "metadata_only"
    elif token_count <= _MEDIUM_TOKEN_THRESHOLD:
        result["strategy"] = "selected"
        picked = _pick_key_sections(sections_meta)
        log.info("  选中章节：%s", picked)
        for name in picked:
            try:
                result["sections"][name] = read_section(arxiv_id, name)
            except Exception as e:
                log.warning("  读章节 [%s] 失败：%s", name, e)
        if not result["sections"]:
            log.warning("  所有章节读取失败，降级为 metadata_only")
            result["strategy"] = "metadata_only"
    else:
        result["strategy"] = "preview"
        try:
            result["preview"] = run_deepxiv(["paper", arxiv_id, "--preview"], parse_json=False)
        except Exception as e:
            log.warning("取 --preview 失败，只保留元数据：%s", e)
            result["strategy"] = "metadata_only"

    log.info(
        "  策略=%s, token_count=%d, 章节数=%d",
        result["strategy"],
        token_count,
        len(result["sections"]),
    )

    if save and result["strategy"] != "failed":
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = arxiv_id.replace("/", "_")
        out = PAPERS_DIR / f"{safe_id}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("  已保存：%s", out)

    # metadata_only 不标 seen（下次重试可能能拿到全文）
    # failed 也不标 seen
    if mark_seen and result["strategy"] not in ("failed", "metadata_only"):
        seen = load_seen_ids()
        seen.add(arxiv_id)
        save_seen_ids(seen)

    return result


if __name__ == "__main__":
    import sys

    # 默认跑一篇 demo；自定义 ID：python src/reader.py 2411.11707
    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2411.11707"
    result = full_read(arxiv_id)
    strategy = result.get("strategy")
    if strategy == "failed":
        log.error("%s 精读失败：%s", arxiv_id, result.get("error"))
        sys.exit(1)
    log.info(
        "冒烟测试：%s 标题=%s token=%s 策略=%s",
        arxiv_id,
        (result.get("title") or "")[:60],
        result.get("token_count"),
        strategy,
    )
