"""文献知识库问答模块：基于 enrichment JSON + summary 的 RAG 问答系统。

用法：
  交互模式：python src/qa_agent.py --project-path <项目路径>
  单次提问：python src/qa_agent.py --ask "你的问题" [--project-path <路径>]
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from project_analyzer import _build_project_summary, scan_project
from summarizer import Anthropic, DEFAULT_MODEL, load_enrichment
from utils import (
    OUTPUT_DIR,
    PAPERS_DIR,
    classify_llm_error,
    create_llm_message,
    get_llm_config,
    get_logger,
)

log = get_logger("qa_agent")

QA_MAX_TOKENS = 4000
QA_DIR = OUTPUT_DIR / "qa"


# ================================================================
# 知识库加载
# ================================================================


def load_knowledge_base() -> list[dict]:
    """加载所有 enrichment JSON + 对应的 summary，构建本地知识库。"""
    kb = []
    for path in sorted(PAPERS_DIR.glob("*.enrichment.json")):
        try:
            enrichment = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        arxiv_id = enrichment.get("arxiv_id", "")
        if not arxiv_id:
            # 从文件名推断 (e.g. "2107.04689.enrichment.json" → "2107.04689")
            arxiv_id = path.stem.replace(".enrichment", "")

        safe_id = arxiv_id.replace("/", "_")

        # 加载对应 summary
        summary = ""
        summary_path = PAPERS_DIR / f"{safe_id}.summary.md"
        if summary_path.exists():
            try:
                summary = summary_path.read_text(encoding="utf-8")
            except OSError:
                pass

        # 加载公式
        formulas = []
        formula_path = PAPERS_DIR / f"{safe_id}.formulas.json"
        if formula_path.exists():
            try:
                fj = json.loads(formula_path.read_text(encoding="utf-8"))
                for f in fj.get("formulas", []):
                    if isinstance(f, dict) and f.get("type") == "display" and f.get("latex"):
                        formulas.append({
                            "id": f.get("id"),
                            "latex": f.get("latex"),
                            "label": f.get("label"),
                            "eq_num": f.get("eq_num"),
                        })
            except (OSError, json.JSONDecodeError):
                pass

        kb.append({
            "arxiv_id": arxiv_id,
            "enrichment": enrichment,
            "summary": summary,
            "formulas": formulas,
        })

    log.info("知识库加载完成：%d 篇论文（%d 有 enrichment）", len(kb), len(kb))
    return kb


# ================================================================
# 检索
# ================================================================

# 中文关键词 → enrichment 字段的映射
_KEYWORD_EXPANSION = {
    "蒸馏": ["knowledge_distillation", "distillation", "teacher", "student"],
    "剪枝": ["pruning", "prune", "sparsity", "sparse"],
    "量化": ["quantization", "quantize", "quantum"],
    "压缩": ["compression", "compact", "lightweight"],
    "故障": ["fault", "diagnosis", "bearing", "vibration"],
    "轴承": ["bearing", "fault", "vibration", "CWRU", "Paderborn"],
    "边缘": ["edge", "deploy", "inference", "onnx", "tensorrt", "mobile"],
    "频域": ["frequency", "spectral", "FFT", "fourier"],
    "注意力": ["attention", "SE", "channel", "spatial"],
    "损失": ["loss", "objective", "function"],
    "跨域": ["cross-domain", "transfer", "domain adaptation", "generalization"],
    "困难样本": ["hard example", "hard sample", "curriculum", "focal"],
    "公式": ["formula", "equation", "mathematical"],
    "模型": ["model", "architecture", "network", "backbone"],
    "特征": ["feature", "representation", "embedding"],
    "训练": ["train", "optimization", "learning rate", "schedule"],
    "推理": ["inference", "predict", "runtime"],
    "部署": ["deploy", "serve", "production", "edge"],
    "可解释": ["interpretab", "explainab", "attention map", "visualization"],
    "鲁棒": ["robust", "noise", "adversarial", "out-of-distribution"],
    "数据增强": ["augment", "mixup", "cutmix", "crop"],
    "多尺度": ["multi-scale", "multiscale", "pyramid"],
    "残差": ["residual", "skip connection", "resnet"],
    "归一化": ["normaliz", "batch norm", "layer norm", "rms norm"],
}


def _extract_keywords(question: str) -> list[str]:
    """从问题中提取关键词（中英文混合）。"""
    keywords = []

    # 英文单词（3+ 字符）
    for w in re.findall(r"[a-zA-Z_]{3,}", question.lower()):
        keywords.append(w)

    # 中文关键词匹配
    for cn_key, en_expansions in _KEYWORD_EXPANSION.items():
        if cn_key in question:
            keywords.extend(en_expansions)

    # arxiv_id 模式
    for aid in re.findall(r"\d{4}\.\d{4,5}", question):
        keywords.append(aid)

    return list(set(keywords))


def _score_paper(entry: dict, keywords: list[str], question: str) -> float:
    """对单篇论文计算与问题的相关性分数。"""
    enrichment = entry.get("enrichment", {})
    summary = entry.get("summary", "")
    score = 0.0

    # 1. method_type 匹配（权重 3）
    method_type = enrichment.get("method_profile", {}).get("method_type", "")
    for kw in keywords:
        if kw in method_type:
            score += 3.0
            break

    # 2. applicable_scenarios 匹配（权重 2）
    scenarios = enrichment.get("transferability", {}).get("applicable_scenarios", [])
    for kw in keywords:
        for s in scenarios:
            if kw in s.lower():
                score += 2.0
                break

    # 3. field_relevance.fault_diagnosis（权重 1.5）
    fr = enrichment.get("field_relevance", {})
    fd = fr.get("fault_diagnosis", {})
    if isinstance(fd, dict):
        score += fd.get("score", 0) * 0.3

    # 4. one_line_summary + innovation_claim 关键词匹配（权重 1）
    text_pool = " ".join([
        enrichment.get("method_profile", {}).get("one_line_summary", ""),
        enrichment.get("method_profile", {}).get("innovation_claim", ""),
        enrichment.get("method_profile", {}).get("method_name", ""),
    ]).lower()
    for kw in keywords:
        if kw in text_pool:
            score += 1.0

    # 5. summary 全文关键词匹配（权重 0.3）
    summary_lower = summary.lower()
    for kw in keywords:
        if kw in summary_lower:
            score += 0.3

    # 6. applicable_scenarios 全文匹配（权重 0.5）
    tech = enrichment.get("technical_spec", {})
    modules_text = " ".join(
        m.get("name", "") + " " + m.get("role", "")
        for m in tech.get("core_modules", [])
        if isinstance(m, dict)
    ).lower()
    for kw in keywords:
        if kw in modules_text:
            score += 0.5

    # 7. 问题直接提到 arxiv_id
    arxiv_id = entry.get("arxiv_id", "")
    if arxiv_id in question:
        score += 10.0

    return score


def retrieve(question: str, kb: list[dict], top_k: int = 5) -> list[dict]:
    """从知识库中检索最相关的论文。"""
    keywords = _extract_keywords(question)
    log.info("问题关键词：%s", keywords[:10])

    scored = []
    for entry in kb:
        s = _score_paper(entry, keywords, question)
        if s > 0:
            scored.append((s, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for s, entry in scored[:top_k]:
        results.append({
            "arxiv_id": entry["arxiv_id"],
            "relevance_score": round(s, 2),
            "enrichment": entry["enrichment"],
            "summary": entry["summary"],
            "formulas": entry["formulas"],
        })

    log.info("检索完成：%d 篇相关（top score=%.1f）", len(results), results[0]["relevance_score"] if results else 0)
    return results


# ================================================================
# 问答生成
# ================================================================


def _build_qa_prompt(
    question: str,
    retrieved: list[dict],
    project_summary: str | None = None,
) -> str:
    """构建问答 prompt。"""
    papers_block = []
    for i, r in enumerate(retrieved, 1):
        aid = r["arxiv_id"]
        enrichment = r["enrichment"]
        summary = r["summary"]
        formulas = r["formulas"]

        mp = enrichment.get("method_profile", {})
        tech = enrichment.get("technical_spec", {})
        trans = enrichment.get("transferability", {})
        results = enrichment.get("empirical_results", {})
        limits = enrichment.get("limitations", {})

        paper_text = f"""### 论文 {i}：[{aid}] {mp.get('method_name', '')}
- 方法类型：{mp.get('method_type', '')}
- 一句话：{mp.get('one_line_summary', '')}
- 创新点：{mp.get('innovation_claim', '')}
- 检索相关度：{r['relevance_score']}

**核心模块：**"""
        for m in tech.get("core_modules", []):
            if isinstance(m, dict):
                paper_text += f"\n  - {m.get('name', '')}：{m.get('role', '')}（输入：{m.get('input', '?')} → 输出：{m.get('output', '?')}）"

        paper_text += "\n\n**损失函数：**"
        for l in tech.get("loss_functions", []):
            if isinstance(l, dict):
                paper_text += f"\n  - {l.get('name', '')}：{l.get('description', '')}（角色：{l.get('role', '')}）"

        if trans.get("transferable_components"):
            paper_text += "\n\n**可迁移组件：**"
            for c in trans["transferable_components"]:
                if isinstance(c, dict):
                    paper_text += f"\n  - {c.get('name', '')}（工作量：{c.get('effort', '?')}）：{c.get('description', '')}"
                    paper_text += f"\n    集成方式：{c.get('modification', '')}"

        if trans.get("integration_points"):
            paper_text += "\n\n**集成点：**"
            for p in trans["integration_points"]:
                paper_text += f"\n  - {p}"

        if results.get("key_metrics"):
            paper_text += "\n\n**关键指标：**"
            for m in results["key_metrics"]:
                if isinstance(m, dict):
                    paper_text += f"\n  - {m.get('metric', '')}: {m.get('value', '')} {m.get('unit', '')} ({m.get('vs_baseline', '')})"

        if limits.get("transfer_risks"):
            paper_text += f"\n\n**迁移风险：** {'; '.join(limits['transfer_risks'])}"

        # 公式
        if formulas:
            paper_text += "\n\n**关键公式：**"
            for f in formulas[:3]:
                paper_text += f"\n  - [{f.get('id', '')}] {f.get('label') or '公式'}: $${f.get('latex', '')[:200]}$$"

        # 详细摘要（截断）
        if summary:
            trunc = summary[:3000] + ("..." if len(summary) > 3000 else "")
            paper_text += f"\n\n**详细解读：**\n{trunc}"

        papers_block.append(paper_text)

    papers_section = "\n\n---\n\n".join(papers_block)

    project_section = ""
    if project_summary:
        project_section = f"""

【用户项目现状】
{project_summary}
"""

    return f"""你是轴承故障诊断 + 模型压缩 + 知识蒸馏领域的资深研究助手。
用户正在做一个科研项目，遇到了问题。请基于知识库中最相关的论文，给出具体、可操作的解决方案。

【用户问题】
{question}
{project_section}
【知识库中最相关的 {len(retrieved)} 篇论文】
{papers_section}

请按以下格式回答：

## 问题分析
（简要分析用户的问题属于什么技术挑战）

## 解决方案
针对每篇相关论文，给出具体的迁移建议：
1. **方案名**（来自 [arxiv_id] 论文名）
   - 核心思路：一句话
   - 具体做法：如何集成到用户项目（指明文件、模块、函数）
   - 预期效果：能解决什么问题
   - 工作量和风险

## 推荐优先级
按"投入产出比"排序，推荐最佳的 1-2 个方案优先尝试。

## 创新点提示
基于以上论文的交叉分析，有没有可能组合多篇论文的技术做出创新？
"""


def ask(
    question: str,
    kb: list[dict] | None = None,
    project_path: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    top_k: int = 5,
    save: bool = True,
) -> dict:
    """单次问答入口。"""
    if kb is None:
        kb = load_knowledge_base()

    if not kb:
        return {"error": "知识库为空，请先运行 pipeline 生成 enrichment JSON"}

    # 检索
    retrieved = retrieve(question, kb, top_k=top_k)
    if not retrieved:
        return {"error": "未找到相关论文，请尝试换个问法"}

    # 项目上下文（可选）
    project_summary = None
    if project_path:
        try:
            profile = scan_project(project_path)
            project_summary = _build_project_summary(profile)
        except Exception as e:
            log.warning("项目扫描失败（不影响问答）：%s", e)

    # 生成回答
    prompt = _build_qa_prompt(question, retrieved, project_summary)
    log.info("生成回答（prompt=%d 字符，%d 篇论文）", len(prompt), len(retrieved))

    llm = get_llm_config()
    client = Anthropic(_llm_config=llm)
    try:
        msg = create_llm_message(
            client,
            config=llm,
            model=model,
            max_tokens=QA_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        error_info = classify_llm_error(exc, llm)
        log.warning("问答 LLM 调用失败：%s", error_info["message"])
        return {
            "error": error_info["message"],
            "error_category": error_info["category"],
            "error_info": error_info,
        }
    answer = "".join(
        blk.text for blk in msg.content if getattr(blk, "type", None) == "text"
    )

    usage = getattr(msg, "usage", None)
    log.info(
        "回答生成完成：%d 字符（input=%s, output=%s）",
        len(answer),
        getattr(usage, "input_tokens", "?"),
        getattr(usage, "output_tokens", "?"),
    )

    result = {
        "question": question,
        "answer": answer,
        "retrieved_papers": [r["arxiv_id"] for r in retrieved],
        "retrieved_scores": [r["relevance_score"] for r in retrieved],
        "model": model,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    if save:
        QA_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 保存完整结果
        json_path = QA_DIR / f"qa_{date_str}.json"
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 保存可读 markdown
        md_path = QA_DIR / f"qa_{date_str}.md"
        md_content = f"# 问答记录\n\n**问题**：{question}\n\n{answer}\n\n---\n\n**引用论文**：{', '.join(result['retrieved_papers'])}\n\n**生成时间**：{result['generated_at']}\n"
        md_path.write_text(md_content, encoding="utf-8")
        log.info("问答记录已保存：%s", md_path)
        result["saved_to"] = str(md_path)

    return result


# ================================================================
# 交互模式
# ================================================================


def interactive_loop(project_path: str | Path | None = None) -> None:
    """交互式问答循环。"""
    print("=" * 60)
    print("  smart-literature-agent 文献知识库问答")
    print("=" * 60)

    kb = load_knowledge_base()
    if not kb:
        print("知识库为空！请先运行 python src/run.py 积累文献数据。")
        return

    print(f"已加载 {len(kb)} 篇论文的知识库。")
    if project_path:
        print(f"项目上下文：{project_path}")
    print("输入问题开始对话，输入 q/quit/exit 退出。\n")

    while True:
        try:
            question = input("你的问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question or question.lower() in ("q", "quit", "exit"):
            print("再见！")
            break

        result = ask(question, kb=kb, project_path=project_path, save=True)

        if "error" in result:
            print(f"  错误：{result['error']}\n")
            continue

        print()
        print(result["answer"])
        print()
        print(f"  参考论文：{', '.join(result['retrieved_papers'])}")
        if result.get("saved_to"):
            print(f"  记录已保存：{result['saved_to']}")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="文献知识库问答")
    parser.add_argument("--ask", default=None, help="单次提问（非交互模式）")
    parser.add_argument("--project-path", default=None, help="科研项目路径（提供上下文）")
    parser.add_argument("--top-k", type=int, default=5, help="检索 top K 篇论文")
    parser.add_argument("--model", default=None, help="LLM 模型")
    args = parser.parse_args()

    model = args.model or DEFAULT_MODEL

    if args.ask:
        result = ask(args.ask, project_path=args.project_path, top_k=args.top_k, model=model)
        if "error" in result:
            print(f"错误：{result['error']}")
            sys.exit(1)
        print(result["answer"])
    else:
        interactive_loop(project_path=args.project_path)
