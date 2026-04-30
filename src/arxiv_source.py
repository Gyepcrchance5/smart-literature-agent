"""arXiv 源码下载与 .tex 主文件定位。

arXiv 对每篇论文都提供 LaTeX 源码：
  URL: https://arxiv.org/e-print/<arxiv_id>
  返回体可能是：
    - tar.gz（大多数论文）
    - gzip 单文件（少数只有一个 .tex 的论文）
    - 纯文本 .tex（更少见）

下载后解压到 data/arxiv_src/<arxiv_id>/，并返回主 .tex 文件路径。
主 .tex 的判定：找含有 `\documentclass` 的文件；若有多个，优先含 `\begin{document}` 的。

arXiv rate limit：约 4 req/s；设 polite sleep 300ms + User-Agent。
"""
from __future__ import annotations

import gzip
import io
import re
import tarfile
import time
from pathlib import Path

import requests

from utils import DATA_DIR, get_logger

log = get_logger("arxiv_source")

_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
_SRC_DIR = DATA_DIR / "arxiv_src"
_POLITE_SLEEP_MS = 300
_USER_AGENT = "smart-literature-agent (https://github.com/Gyepcrchance5/smart-literature-agent; arxiv source fetcher)"


def _cache_dir(arxiv_id: str) -> Path:
    safe = arxiv_id.replace("/", "_")
    return _SRC_DIR / safe


def _download_eprint(arxiv_id: str, timeout: int = 30) -> bytes:
    """下载 arXiv e-print，返回原始字节。"""
    url = _EPRINT_URL.format(arxiv_id=arxiv_id)
    log.info("下载 arXiv 源码：%s", url)
    r = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"arxiv e-print {arxiv_id} HTTP {r.status_code}")
    content = r.content
    time.sleep(_POLITE_SLEEP_MS / 1000)
    return content


def _extract(content: bytes, dest: Path) -> None:
    """解压下载体到 dest/。自动识别 tar.gz / gzip / 纯 tex 三种形式。"""
    dest.mkdir(parents=True, exist_ok=True)
    # 尝试 tarfile（可兼容 tar.gz / tar）
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tf:
            for member in tf.getmembers():
                # 安全：阻止 path traversal
                if member.name.startswith(("/", "..")) or ".." in member.name:
                    continue
                if not member.isfile():
                    continue
                tf.extract(member, dest, filter="data")
        return
    except tarfile.TarError:
        pass
    # 尝试 gzip 单文件
    try:
        raw = gzip.decompress(content)
        # 猜它是 .tex 文件
        (dest / "main.tex").write_bytes(raw)
        return
    except (OSError, gzip.BadGzipFile):
        pass
    # 视作纯文本 .tex
    (dest / "main.tex").write_bytes(content)


def _find_main_tex(dir_: Path) -> Path | None:
    """在解压出的 .tex 文件里找主文件。
    策略：含 \\documentclass 的 = 候选；含 \\begin{document} 的 = 强候选。
    """
    candidates = []
    strong = []
    for p in dir_.rglob("*.tex"):
        try:
            head = p.read_text(encoding="utf-8", errors="ignore")[:20000]
        except OSError:
            continue
        if r"\documentclass" in head:
            candidates.append(p)
            if r"\begin{document}" in head:
                strong.append(p)
    if strong:
        # 有多个强候选时取最短路径（避免选到 subsubmission）
        return min(strong, key=lambda p: (len(p.parts), len(str(p))))
    if candidates:
        return min(candidates, key=lambda p: (len(p.parts), len(str(p))))
    # 没 documentclass 时 fallback：任意 .tex 文件里最大的一个
    all_tex = list(dir_.rglob("*.tex"))
    if all_tex:
        return max(all_tex, key=lambda p: p.stat().st_size)
    return None


def _resolve_inputs(main_path: Path, depth: int = 3) -> str:
    """把 \\input{foo} 和 \\include{foo} 内联展开（递归 depth 层）。
    如果找不到子文件就原样保留。
    """
    try:
        text = main_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    if depth <= 0:
        return text
    base_dir = main_path.parent

    def _sub(m: re.Match) -> str:
        rel = m.group(1).strip()
        if not rel.endswith(".tex"):
            rel_try = rel + ".tex"
        else:
            rel_try = rel
        sub_path = base_dir / rel_try
        if sub_path.exists():
            return "\n% <<< included: " + rel + " >>>\n" + _resolve_inputs(sub_path, depth - 1) + "\n"
        return m.group(0)

    text = re.sub(r"\\(?:input|include)\{([^}]+)\}", _sub, text)
    return text


def fetch_latex(arxiv_id: str, use_cache: bool = True) -> dict:
    """下载 arXiv 源码并返回合并后的主 .tex 文本。

    返回：
      {
        "arxiv_id": str,
        "main_tex_path": str,          # 主 .tex 绝对路径
        "main_tex": str,               # 主 .tex 内容（已内联 \\input / \\include）
        "cached": bool,                # 是否走缓存
        "tex_files_count": int,        # 解压出的 .tex 文件数
      }
    """
    cache = _cache_dir(arxiv_id)
    if use_cache and cache.exists() and any(cache.rglob("*.tex")):
        cached_hit = True
    else:
        cached_hit = False
        content = _download_eprint(arxiv_id)
        _extract(content, cache)

    main_path = _find_main_tex(cache)
    if not main_path:
        raise RuntimeError(f"{arxiv_id}: 解压后未找到 .tex 文件")

    tex_body = _resolve_inputs(main_path)
    tex_count = sum(1 for _ in cache.rglob("*.tex"))

    return {
        "arxiv_id": arxiv_id,
        "main_tex_path": str(main_path),
        "main_tex": tex_body,
        "cached": cached_hit,
        "tex_files_count": tex_count,
    }


if __name__ == "__main__":
    import sys

    aid = sys.argv[1] if len(sys.argv) > 1 else "2411.11707"
    result = fetch_latex(aid)
    print(f"arxiv_id:      {result['arxiv_id']}")
    print(f"cached:        {result['cached']}")
    print(f"main_tex_path: {result['main_tex_path']}")
    print(f"tex_files:     {result['tex_files_count']}")
    print(f"main_tex size: {len(result['main_tex'])} chars")
    print()
    print("--- first 500 chars ---")
    print(result["main_tex"][:500])
