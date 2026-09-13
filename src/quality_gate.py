"""论文质量审查与证据卡。

这里的评分是可解释的初筛，不是对论文科学正确性的自动证明。它把论文分为
核心参考、创新线索和背景/待核查三类，并明确缺失的实验证据。
"""
from __future__ import annotations

import json
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any

from research_session import has_research_context
from utils import DATA_DIR, OUTPUT_DIR, PAPERS_DATA_DIR, PAPERS_DIR, get_logger, try_write_text


log = get_logger("quality_gate")


@lru_cache(maxsize=1)
def _candidate_metadata() -> dict[str, dict[str, Any]]:
    """读取候选池里的 OpenAlex/搜索信号，供质量审查复用。"""
    index: dict[str, dict[str, Any]] = {}
    for candidate_path in sorted(DATA_DIR.glob("candidates_*.json")):
        try:
            data = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in data.get("papers", []) or []:
            arxiv_id = item.get("arxiv_id")
            if arxiv_id:
                index[arxiv_id] = item
    return index


def enrich_quality_metadata(paper: dict[str, Any]) -> dict[str, Any]:
    """将候选池的公开质量信号合并到精读 artifact，不覆盖正文数据。"""
    arxiv_id = paper.get("arxiv_id")
    candidate = _candidate_metadata().get(arxiv_id, {})
    if not candidate:
        return paper
    merged = dict(paper)
    for key in ("score", "citation_count", "_openalex"):
        if key in candidate and key not in merged:
            merged[key] = candidate[key]
    return merged


def _paper_text(paper: dict[str, Any]) -> str:
    parts = [paper.get("title"), paper.get("abstract"), paper.get("tldr")]
    parts.extend([paper.get("raw"), paper.get("preview")])
    sections = paper.get("sections") or {}
    if isinstance(sections, dict):
        parts.extend(sections.values())
    return "\n".join(str(part) for part in parts if part).lower()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term.lower() in text for term in terms)


def _as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _field_relevance(enrichment: dict[str, Any]) -> float:
    scores: list[float] = []
    for item in (enrichment.get("field_relevance") or {}).values():
        raw = item.get("score") if isinstance(item, dict) else item
        score = _as_number(raw)
        if score is not None:
            scores.append(max(0.0, min(5.0, score)))
    return max(scores) / 5 * 100 if scores else 0.0


def _context_relevance(paper: dict[str, Any], context: dict[str, Any] | None) -> float:
    if not has_research_context(context):
        return 0.0
    text = " ".join(
        str(paper.get(key) or "") for key in ("title", "abstract", "keywords", "tldr")
    ).lower()
    terms: list[str] = []
    for key in ("problem", "bottleneck", "symptoms", "constraints", "target_fields"):
        value = (context or {}).get(key)
        values = value if isinstance(value, list) else [value]
        terms.extend(str(item).lower() for item in values if item)
    terms = [term for term in terms if len(term) >= 3]
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if term in text)
    return min(100.0, hits / len(terms) * 100)


def assess_paper(
    paper: dict[str, Any],
    enrichment: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """对论文做确定性的质量初筛，返回可解释的质量结果。"""
    enrichment = enrichment or {}
    text = _paper_text(paper)
    strategy = paper.get("strategy")
    empirical = enrichment.get("empirical_results") or {}
    technical = enrichment.get("technical_spec") or {}
    method = enrichment.get("method_profile") or {}
    transfer = enrichment.get("transferability") or {}

    datasets = empirical.get("datasets") or []
    baselines = empirical.get("baselines") or []
    metrics = empirical.get("key_metrics") or []
    modules = technical.get("core_modules") or []
    has_method = bool(method.get("innovation_claim") or modules) or _contains_any(
        text, ("method", "approach", "proposed", "framework", "architecture")
    )
    has_dataset = bool(datasets) or _contains_any(
        text, ("dataset", "benchmark", "cwru", "pu bearing", "test set", "training set")
    )
    has_baseline = bool(baselines) or _contains_any(
        text, ("baseline", "compared with", "state-of-the-art", "comparison")
    )
    has_metric = bool(metrics) or _contains_any(
        text, ("accuracy", "f1", "precision", "recall", "auc", "rmse", "error rate")
    )
    has_ablation = bool(empirical.get("ablation_highlights")) or _contains_any(
        text, ("ablation", "without", "effect of", "component analysis")
    )
    has_limitations = bool((enrichment.get("limitations") or {}).get("paper_self_reported")) or _contains_any(
        text, ("limitation", "future work", "cannot", "fails", "failure")
    )
    has_repro = _contains_any(text, ("github", "code is available", "open source", "dataset is available"))
    full_text_score = {"raw": 20, "selected": 16, "preview": 9, "metadata_only": 0}.get(strategy, 0)
    metadata_score = sum(
        bool(paper.get(key)) for key in ("title", "abstract", "authors", "publish_at")
    ) / 4 * 10

    publication_signal = bool(
        paper.get("journal_name")
        or (paper.get("venue") and str(paper.get("venue")).lower() not in {"arxiv", "arxiv.org"})
        or (paper.get("_openalex") or {}).get("venue_type") in {"journal", "conference"}
    )
    score_parts = {
        "metadata": round(metadata_score, 1),
        "full_text": full_text_score,
        "method": 15 if has_method else 0,
        "dataset": 10 if has_dataset else 0,
        "baselines": 8 if has_baseline else 0,
        "metrics": 10 if has_metric else 0,
        "ablation": 7 if has_ablation else 0,
        "reproducibility": 10 if has_repro else 0,
        "publication_signal": 10 if publication_signal else 0,
    }
    score = round(sum(score_parts.values()), 1)
    core_evidence = all((has_method, has_dataset, has_baseline, has_metric))
    if strategy == "metadata_only":
        level = "C"
    elif strategy == "preview" or not core_evidence:
        level = "B" if score >= 50 else "C"
    elif score >= 75:
        level = "A"
    elif score >= 50:
        level = "B"
    else:
        level = "C"

    missing: list[str] = []
    for present, label in (
        (has_method, "方法/机制"),
        (has_dataset, "数据集或任务"),
        (has_baseline, "对比基线"),
        (has_metric, "量化指标"),
        (has_ablation, "消融或组件分析"),
    ):
        if not present:
            missing.append(label)
    if strategy in {"preview", "metadata_only"}:
        missing.append("完整正文（当前只读取了预览或元数据）")
    if not publication_signal:
        missing.append("正式发表状态（当前更像 arXiv 预印本或未知）")

    field_score = _field_relevance(enrichment)
    context_score = _context_relevance(paper, context)
    components = transfer.get("transferable_components") or []
    scenarios = transfer.get("applicable_scenarios") or []
    transfer_score = round(
        min(
            100.0,
            field_score * 0.45
            + context_score * 0.35
            + min(len(components), 3) / 3 * 12
            + min(len(scenarios), 3) / 3 * 8,
        ),
        1,
    )
    if level == "A" and (not has_research_context(context) or transfer_score >= 55):
        decision = "core_reference"
    elif level in {"A", "B"}:
        decision = "inspiration"
    else:
        decision = "lead_or_background"

    return {
        "schema_version": "0.1",
        "arxiv_id": paper.get("arxiv_id"),
        "quality_score": score,
        "quality_level": level,
        "decision": decision,
        "score_parts": score_parts,
        "evidence_checks": {
            "has_method": has_method,
            "has_dataset": has_dataset,
            "has_baseline": has_baseline,
            "has_metric": has_metric,
            "has_ablation": has_ablation,
            "has_limitations": has_limitations,
            "has_reproducibility_signal": has_repro,
            "publication_signal": publication_signal,
        },
        "relevance_score": round(max(field_score, context_score), 1),
        "transferability_score": transfer_score if has_research_context(context) else None,
        "missing_evidence": missing,
        "requires_human_verification": True,
        "caveat": "这是可解释的文献初筛，不是对论文科学正确性的自动证明。",
    }


def build_evidence_card(
    paper: dict[str, Any],
    enrichment: dict[str, Any] | None,
    assessment: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把论文内容、结构化提取和质量结果合并成 Agent 可消费的证据卡。"""
    enrichment = enrichment or {}
    empirical = enrichment.get("empirical_results") or {}
    method = enrichment.get("method_profile") or {}
    limitations = enrichment.get("limitations") or {}
    return {
        "schema_version": "0.1",
        "arxiv_id": paper.get("arxiv_id"),
        "title": paper.get("title"),
        "source": {
            "type": "arxiv",
            "url": f"https://arxiv.org/abs/{paper.get('arxiv_id')}",
            "read_strategy": paper.get("strategy"),
            "published_hint": paper.get("journal_name") or paper.get("venue"),
            "search_score": paper.get("score"),
            "openalex": paper.get("_openalex") or {},
        },
        "quality": assessment,
        "evidence": {
            "core_claim": method.get("innovation_claim"),
            "method_name": method.get("method_name"),
            "datasets": empirical.get("datasets") or [],
            "baselines": empirical.get("baselines") or [],
            "key_metrics": empirical.get("key_metrics") or [],
            "ablation": empirical.get("ablation_highlights"),
            "limitations": limitations,
            "transferability": enrichment.get("transferability") or {},
            "evidence_sections": enrichment.get("evidence_sections") or [],
            "evidence_gaps": enrichment.get("evidence_gaps") or assessment.get("missing_evidence", []),
            "verification_note": "字段可能来自 LLM 结构化提取；正式引用或实验决策前必须回看原文证据。",
        },
        "research_context": context or {},
        "provenance": {
            "paper_artifact": str(PAPERS_DATA_DIR / f"{str(paper.get('arxiv_id')).replace('/', '_')}.json"),
            "summary_artifact": str(PAPERS_DIR / f"{str(paper.get('arxiv_id')).replace('/', '_')}.summary.md"),
            "enrichment_artifact": str(PAPERS_DIR / f"{str(paper.get('arxiv_id')).replace('/', '_')}.enrichment.json"),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def save_evidence_card(card: dict[str, Any], path: str | Path | None = None) -> Path | None:
    """保存单篇证据卡，默认与 summary/enrichment 放在同一目录。"""
    arxiv_id = str(card.get("arxiv_id") or "unknown").replace("/", "_")
    out = Path(path) if path else PAPERS_DIR / f"{arxiv_id}.evidence.json"
    return try_write_text(
        out,
        json.dumps(card, ensure_ascii=False, indent=2),
        logger=log,
    )


def audit_library(context: dict[str, Any] | None = None, save: bool = True) -> dict[str, Any]:
    """对已有 data/papers 重新审查，不调用 LLM。"""
    audited = 0
    counts = {"A": 0, "B": 0, "C": 0}
    saved_paths: list[str] = []
    for paper_path in sorted(PAPERS_DATA_DIR.glob("*.json")):
        try:
            paper = json.loads(paper_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("跳过损坏的论文 artifact %s: %s", paper_path, exc)
            continue
        arxiv_id = paper.get("arxiv_id") or paper_path.stem
        enrichment_path = PAPERS_DIR / f"{str(arxiv_id).replace('/', '_')}.enrichment.json"
        enrichment: dict[str, Any] = {}
        if enrichment_path.exists():
            try:
                enrichment = json.loads(enrichment_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        paper = enrich_quality_metadata(paper)
        assessment = assess_paper(paper, enrichment, context)
        card = build_evidence_card(paper, enrichment, assessment, context)
        counts[assessment["quality_level"]] += 1
        audited += 1
        if save:
            evidence_path = save_evidence_card(card)
            if evidence_path:
                saved_paths.append(str(evidence_path))

    result = {
        "audited": audited,
        "quality_counts": counts,
        "context_configured": has_research_context(context),
        "saved_cards": len(saved_paths),
    }
    if save:
        audit_path = OUTPUT_DIR / "quality_audit.json"
        saved = try_write_text(
            audit_path,
            json.dumps(result, ensure_ascii=False, indent=2),
            logger=log,
        )
        if saved:
            result["saved_to"] = str(saved)
        else:
            result["save_error"] = f"无法写入 {audit_path}"
    return result
