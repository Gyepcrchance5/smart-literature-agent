"""一次性探测脚本：看 DeepXiv 返回的论文正文里公式是什么格式。"""
import json
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))


def probe(arxiv_id: str) -> None:
    p = PROJECT / "output" / "papers" / f"{arxiv_id}.json"
    if not p.exists():
        print(f"skip {arxiv_id}: no JSON")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    strategy = d.get("strategy")
    # 拿可读正文
    if strategy == "raw":
        text = d.get("raw") or ""
    elif strategy == "selected":
        text = "\n\n".join(d.get("sections", {}).values())
    elif strategy == "preview":
        text = d.get("preview") or ""
    else:
        text = d.get("abstract") or ""

    print(f"\n========== {arxiv_id} (strategy={strategy}) 正文 {len(text)} 字符 ==========")

    # 各种 LaTeX 公式标记
    patterns = [
        (r"\$\$[^\$]{1,500}\$\$", "display math  $$...$$"),
        (r"(?<!\$)\$[^\$\n]{2,200}\$(?!\$)", "inline math   $...$"),
        (r"\\begin\{equation\*?\}", "env: equation"),
        (r"\\begin\{align\*?\}", "env: align"),
        (r"\\begin\{eqnarray\*?\}", "env: eqnarray"),
        (r"\\\[.+?\\\]", "\\[...\\]"),
        (r"\\\(.+?\\\)", "\\(...\\)"),
        (r"!\[[^\]]*\]\([^\)]*\)", "markdown image"),
    ]
    for pat, label in patterns:
        hits = re.findall(pat, text, flags=re.DOTALL)
        print(f"  {label:30s} : {len(hits):4d} 次")

    # 打印前 3 个 display math 的样本
    for i, m in enumerate(re.findall(r"\$\$[^\$]{1,500}\$\$", text)[:3]):
        print(f"  sample display [{i+1}]: {m[:180]!r}")

    # 打印前 3 个 inline math 的样本
    for i, m in enumerate(re.findall(r"(?<!\$)\$[^\$\n]{2,200}\$(?!\$)", text)[:3]):
        print(f"  sample inline  [{i+1}]: {m[:120]!r}")

    # 打印 'Eq.' 附近的上下文（看是否有"公式编号"）
    eq_refs = list(re.finditer(r"[Ee]q(?:uation)?\s*[.\(]?\s*\(?\d+\)?", text))[:3]
    for i, m in enumerate(eq_refs):
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        print(f"  ref ctx [{i+1}]: ...{text[start:end]!r}...")


if __name__ == "__main__":
    for aid in ["2411.11707", "2604.22529", "2604.22338", "2604.22432"]:
        probe(aid)
