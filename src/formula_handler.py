"""公式处理模块：从 LaTeX 源码中提取数学公式，带编号、label、上下文。

支持的数学环境 / 定界符：
  \\begin{equation}...\\end{equation}         (带编号)
  \\begin{equation*}...\\end{equation*}       (无编号)
  \\begin{align}...\\end{align}               (带编号)
  \\begin{align*}...\\end{align*}             (无编号)
  \\begin{eqnarray}...\\end{eqnarray}         (带编号，旧风格)
  \\begin{gather}...\\end{gather}             (带编号)
  \\begin{multline}...\\end{multline}         (带编号)
  \\[ ... \\]                                 (display math, 无编号)
  $$ ... $$                                   (display math, 无编号)
  \\( ... \\)                                 (inline math)
  $ ... $                                     (inline math)

路由：
  extract(source) 根据 source 类型分派到：
    - arxiv_id        → arxiv_source.fetch_latex + extract_from_latex
    - .pdf 路径       → pdf_handler.extract_from_pdf (Phase 2 stub)
    - URL             → html_handler.extract_from_html (Phase 2 stub)
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from utils import OUTPUT_DIR, get_logger

log = get_logger("formula_handler")

PAPERS_DIR = OUTPUT_DIR / "papers"

FormulaType = Literal["display", "inline"]


@dataclass
class Formula:
    id: str                               # 全文内唯一 id（f1, f2, ...）
    type: FormulaType                     # display / inline
    env: str                              # 所在环境：equation / align / eqnarray / $$ / $ / \[ / \(
    latex: str                            # 公式 LaTeX 源码（去除定界符后的纯公式体）
    numbered: bool                        # 是否带编号（作者原意）
    eq_num: int | None = None             # 自然顺序的编号（从 1 起）；无编号式为 None
    label: str | None = None              # \label{eq:xxx} 里的字符串
    context_before: str = ""              # 公式前 ~150 字符文本
    context_after: str = ""               # 公式后 ~150 字符文本
    char_start: int = 0                   # 在原 tex 里的起始位置（调试用）


# --------- 正则表达式 ---------

# 环境型：\begin{env}...\end{env}
_ENV_NAMES = ("equation", "align", "eqnarray", "gather", "multline")
_ENV_NUMBERED = {n: True for n in _ENV_NAMES}
_ENV_UNNUMBERED = {n + "*": False for n in _ENV_NAMES}
_ENV_ALL = {**_ENV_NUMBERED, **_ENV_UNNUMBERED}


def _env_pattern() -> re.Pattern:
    env_alt = "|".join(re.escape(k) for k in _ENV_ALL.keys())
    # non-greedy body; DOTALL 让 . 跨行
    return re.compile(
        r"\\begin\{(" + env_alt + r")\}(.*?)\\end\{\1\}",
        re.DOTALL,
    )


_RE_ENV = _env_pattern()

# display math: \[ ... \]   或   $$ ... $$
_RE_BRACKET_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_RE_DOLLAR_DISPLAY = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$", re.DOTALL)

# inline math:  \( ... \)
_RE_PAREN_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
# inline math:  $ ... $（最棘手：要避开 $$）
# 前不是 $ 或 \，后不是 $，中间不含换行、不含 $
_RE_DOLLAR_INLINE = re.compile(r"(?<!\\)(?<!\$)\$([^\$\n]{1,400})(?<!\\)\$(?!\$)")

# \label{...} 在公式体内
_RE_LABEL = re.compile(r"\\label\{([^}]+)\}")

# 注释行：% ... 到行尾（避免把注释里的 $ 当公式）
_RE_COMMENT = re.compile(r"(?<!\\)%[^\n]*")

# 最小过滤：不把单符号（如 $\bullet$、$x$ 字符 <= 2 的）当有用的公式
_MIN_INLINE_BODY_LEN = 3


# --------- 工具函数 ---------


def _strip_comments(tex: str) -> str:
    """按行逐个去除 LaTeX 注释。保留 % 转义的 \\% 不动。"""
    return _RE_COMMENT.sub("", tex)


def _extract_label(body: str) -> tuple[str | None, str]:
    """从公式体里提 label，并返回 (label, body去掉label指令)。"""
    labels = _RE_LABEL.findall(body)
    if not labels:
        return None, body
    clean = _RE_LABEL.sub("", body)
    return labels[0].strip(), clean


def _clean_body(body: str) -> str:
    """去前后空白；压缩多余空白但保留换行结构。"""
    body = body.strip()
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n\s*\n", "\n", body)
    return body


def _slice_context(tex: str, start: int, end: int, width: int = 150) -> tuple[str, str]:
    before = tex[max(0, start - width) : start]
    after = tex[end : end + width]
    # 净化一下：去 LaTeX 控制符和多余换行
    before = re.sub(r"\s+", " ", before).strip()
    after = re.sub(r"\s+", " ", after).strip()
    return before, after


# --------- 核心：从 LaTeX 提取公式 ---------


def extract_from_latex(tex: str, context_chars: int = 150) -> list[Formula]:
    """把 LaTeX 源码里所有公式提出来。顺序按出现位置。
    注：这里 eq_num 只是"按出现顺序的自然编号"（1, 2, 3...），
    不完全等价于作者原 \\arabic 计数器（\\nonumber 等情况难完美还原）。
    """
    tex = _strip_comments(tex)
    hits: list[tuple[int, int, Formula]] = []  # (start, end, formula)

    # 1) 数学环境（最优先，内部的 $...$ 不能再被 inline 捕获）
    for m in _RE_ENV.finditer(tex):
        env = m.group(1)
        body = m.group(2)
        label, body_clean = _extract_label(body)
        body_clean = _clean_body(body_clean)
        hits.append(
            (
                m.start(),
                m.end(),
                Formula(
                    id="",
                    type="display",
                    env=env,
                    latex=body_clean,
                    numbered=_ENV_ALL.get(env, True),
                    label=label,
                    char_start=m.start(),
                ),
            )
        )

    # 为避免 delimiter 型的匹配落入已占据的区间（如环境内部的 $）：
    # 先记录已被环境覆盖的区间
    covered: list[tuple[int, int]] = [(s, e) for s, e, _ in hits]

    def _not_covered(start: int, end: int) -> bool:
        for cs, ce in covered:
            if start >= cs and end <= ce:
                return False
        return True

    # 2) \[ ... \] display
    for m in _RE_BRACKET_DISPLAY.finditer(tex):
        if not _not_covered(m.start(), m.end()):
            continue
        body = _clean_body(m.group(1))
        if len(body) < _MIN_INLINE_BODY_LEN:
            continue
        hits.append(
            (m.start(), m.end(), Formula(id="", type="display", env=r"\[", latex=body, numbered=False, char_start=m.start()))
        )

    # 3) $$ ... $$ display
    for m in _RE_DOLLAR_DISPLAY.finditer(tex):
        if not _not_covered(m.start(), m.end()):
            continue
        body = _clean_body(m.group(1))
        if len(body) < _MIN_INLINE_BODY_LEN:
            continue
        hits.append(
            (m.start(), m.end(), Formula(id="", type="display", env="$$", latex=body, numbered=False, char_start=m.start()))
        )

    # 重新汇总 covered（加上新 display 范围）
    covered = [(s, e) for s, e, _ in hits]

    # 4) \( ... \) inline
    for m in _RE_PAREN_INLINE.finditer(tex):
        if not _not_covered(m.start(), m.end()):
            continue
        body = _clean_body(m.group(1))
        if len(body) < _MIN_INLINE_BODY_LEN:
            continue
        hits.append(
            (m.start(), m.end(), Formula(id="", type="inline", env=r"\(", latex=body, numbered=False, char_start=m.start()))
        )

    # 5) $ ... $ inline（最后，跳过已覆盖区间）
    for m in _RE_DOLLAR_INLINE.finditer(tex):
        if not _not_covered(m.start(), m.end()):
            continue
        body = _clean_body(m.group(1))
        if len(body) < _MIN_INLINE_BODY_LEN:
            continue
        # 过滤噪音：没有 \ 命令且没有下划线/上标的"纯文本夹在 $ 中间"
        if "\\" not in body and not re.search(r"[_^{}]", body):
            continue
        hits.append(
            (m.start(), m.end(), Formula(id="", type="inline", env="$", latex=body, numbered=False, char_start=m.start()))
        )

    # 按位置排序，分配 id & eq_num
    hits.sort(key=lambda x: x[0])
    display_counter = 0
    for i, (start, end, f) in enumerate(hits, 1):
        f.id = f"f{i}"
        if f.type == "display" and f.numbered:
            display_counter += 1
            f.eq_num = display_counter
        before, after = _slice_context(tex, start, end, width=context_chars)
        f.context_before = before
        f.context_after = after

    return [f for _, _, f in hits]


# --------- 路由：source → 公式列表 ---------


def extract(source: str, **kwargs) -> list[Formula]:
    """统一入口。根据 source 类型分派到对应 handler。

    - arXiv ID（匹配 \\d+\\.\\d+ 或 旧风格 cs/xxx）→ arxiv 路线（Phase 1 实装）
    - 以 .pdf 结尾的路径                        → pdf_handler（Phase 2 stub）
    - http(s):// 开头的 URL                     → html_handler（Phase 2 stub）
    """
    # arXiv ID pattern：新 ID 如 2411.11707，旧 ID 如 cs/0501001
    if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", source) or re.fullmatch(r"[a-z\-]+/\d{7}", source):
        from arxiv_source import fetch_latex

        res = fetch_latex(source)
        return extract_from_latex(res["main_tex"], **kwargs)

    if source.lower().endswith(".pdf"):
        from pdf_handler import extract_from_pdf

        return extract_from_pdf(source, **kwargs)

    if re.match(r"https?://", source):
        from html_handler import extract_from_html

        return extract_from_html(source, **kwargs)

    raise ValueError(
        f"无法识别的 source：{source!r}。"
        f"支持 arXiv ID（如 '2411.11707'）、PDF 路径（Phase 2）、URL（Phase 2）。"
    )


# --------- 产物：保存 JSON + Markdown ---------


def save_formulas(arxiv_id: str, formulas: list[Formula], source_info: dict | None = None) -> dict:
    """保存 formulas 到 output/papers/<id>.formulas.{json,md}。"""
    safe_id = arxiv_id.replace("/", "_")
    json_path = PAPERS_DIR / f"{safe_id}.formulas.json"
    md_path = PAPERS_DIR / f"{safe_id}.formulas.md"
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = {
        "arxiv_id": arxiv_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source_info or {"type": "arxiv_latex"},
        "counts": {
            "total": len(formulas),
            "display": sum(1 for f in formulas if f.type == "display"),
            "inline": sum(1 for f in formulas if f.type == "inline"),
            "numbered": sum(1 for f in formulas if f.numbered),
            "with_label": sum(1 for f in formulas if f.label),
        },
        "formulas": [asdict(f) for f in formulas],
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Markdown：按 display / inline 分类，display 每条完整列上下文，inline 汇总表
    lines = [
        f"# {arxiv_id} 公式速览",
        "",
        f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}  ",
        f"> 共 {payload['counts']['total']} 个公式："
        f"{payload['counts']['display']} display + {payload['counts']['inline']} inline；"
        f"{payload['counts']['numbered']} 带编号、{payload['counts']['with_label']} 带 label",
        "",
        "---",
        "",
        "## Display 公式（核心）",
        "",
    ]
    display_fs = [f for f in formulas if f.type == "display"]
    if not display_fs:
        lines.append("_（本论文未使用 display math 环境）_")
        lines.append("")
    for f in display_fs:
        title = f"### {f.id}"
        meta_parts = [f"`{f.env}`"]
        if f.eq_num is not None:
            meta_parts.append(f"**Eq.{f.eq_num}**")
        if f.label:
            meta_parts.append(f"label=`{f.label}`")
        title += " · " + " · ".join(meta_parts)
        lines.append(title)
        lines.append("")
        # 上下文前
        if f.context_before:
            lines.append(f"> …{f.context_before[-120:]}")
            lines.append("")
        # LaTeX code block + display math（MathJax 会在 HTML 里渲染）
        lines.append("```latex")
        lines.append(f.latex)
        lines.append("```")
        lines.append("")
        lines.append(f"$$\n{f.latex}\n$$")
        lines.append("")
        # 上下文后
        if f.context_after:
            lines.append(f"> {f.context_after[:120]}…")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Inline 公式汇总")
    lines.append("")
    inline_fs = [f for f in formulas if f.type == "inline"]
    if not inline_fs:
        lines.append("_（无）_")
    else:
        lines.append("| id | latex | 上下文 |")
        lines.append("| :--- | :--- | :--- |")
        for f in inline_fs:
            latex_cell = f.latex.replace("|", r"\|").replace("\n", " ")
            ctx = (f.context_before[-50:] + " **" + f.latex + "** " + f.context_after[:50]).replace("|", r"\|")
            ctx = re.sub(r"\s+", " ", ctx).strip()
            lines.append(f"| {f.id} | `${latex_cell}$` | {ctx[:160]} |")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    log.info(
        "已保存公式产物：%s (display=%d, inline=%d)",
        md_path.name, payload["counts"]["display"], payload["counts"]["inline"],
    )

    return {
        "json": str(json_path),
        "md": str(md_path),
        "counts": payload["counts"],
    }


if __name__ == "__main__":
    import sys

    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2411.11707"
    formulas = extract(arxiv_id)
    print(f"\n{arxiv_id} 提取到 {len(formulas)} 个公式：")
    print(f"  display = {sum(1 for f in formulas if f.type == 'display')}")
    print(f"  inline  = {sum(1 for f in formulas if f.type == 'inline')}")
    print(f"  numbered = {sum(1 for f in formulas if f.numbered)}")
    print(f"  with label = {sum(1 for f in formulas if f.label)}")
    print()
    print("前 3 个 display 公式：")
    for f in [x for x in formulas if x.type == "display"][:3]:
        print(f"  [{f.id}] env={f.env} eq_num={f.eq_num} label={f.label}")
        print(f"    latex: {f.latex[:120]}")
        print()
    save_info = save_formulas(arxiv_id, formulas, {"type": "arxiv_latex"})
    print(f"产物：\n  {save_info['json']}\n  {save_info['md']}")
