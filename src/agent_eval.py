"""Agent 运行时的离线评测与 trace 审计。

评测分成两层：

1. ``evaluate_planner`` 不调用 LLM，检查任务意图和工具覆盖率，适合每次提交运行；
2. ``audit_agent_runs`` 读取本地 Agent trace，统计工具成功率、引用可追溯率、反思通过率
   和预算遵守情况，帮助定位“能跑但不可靠”的链路。

这里的指标是工程质量信号，不是对答案科学正确性的证明。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from agent_runtime import AGENT_RUNS_DIR, TaskPlanner, extract_arxiv_ids, plan_progress, reflect_answer
from utils import CONFIG_DIR, OUTPUT_DIR, get_logger

log = get_logger("agent_eval")

DEFAULT_CASES_PATH = CONFIG_DIR / "agent_eval_cases.yaml"

_BUILTIN_CASES: list[dict[str, Any]] = [
    {
        "id": "discover",
        "query": "这周有什么关于模型压缩的新论文？",
        "expected_intent": "discovery",
        "expected_tools": ["search_candidates", "retrieve_literature"],
    },
    {
        "id": "paper-review",
        "query": "论文 2411.11707 值得作为研究依据吗？",
        "expected_intent": "paper_review",
        "expected_tools": ["read_paper", "audit_literature"],
    },
    {
        "id": "project-match",
        "query": "结合我的项目瓶颈，判断剪枝方法能否迁移。",
        "expected_intent": "project_match",
        "expected_tools": ["inspect_project", "retrieve_literature"],
    },
    {
        "id": "quality",
        "query": "帮我审查知识库里哪些论文证据更完整。",
        "expected_intent": "quality_audit",
        "expected_tools": ["retrieve_literature", "audit_literature"],
    },
    {
        "id": "comparison",
        "query": "比较 2411.11707 和 2501.00001 的方法差异。",
        "expected_intent": "synthesis",
        "expected_tools": ["compare_papers"],
    },
]


def load_eval_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    """加载评测用例；配置缺失时使用内置最小集。"""
    case_path = Path(path) if path else DEFAULT_CASES_PATH
    if not case_path.exists():
        return list(_BUILTIN_CASES)
    try:
        raw = yaml.safe_load(case_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("评测用例读取失败，使用内置用例：%s", exc)
        return list(_BUILTIN_CASES)
    cases = raw.get("cases", []) if isinstance(raw, dict) else []
    valid = [
        case for case in cases
        if isinstance(case, dict) and case.get("id") and case.get("query")
    ]
    return valid or list(_BUILTIN_CASES)


def evaluate_planner(
    cases: Iterable[dict[str, Any]] | None = None,
    planner: TaskPlanner | None = None,
) -> dict[str, Any]:
    """离线检查规划器的意图识别和工具覆盖率。"""
    planner = planner or TaskPlanner()
    selected = list(cases) if cases is not None else load_eval_cases()
    results: list[dict[str, Any]] = []
    intent_hits = 0
    coverage_values: list[float] = []

    for case in selected:
        plan = planner.plan(str(case.get("query", "")))
        expected_intent = case.get("expected_intent")
        expected_tools = set(case.get("expected_tools") or [])
        actual_tools = set(plan.tool_names)
        intent_ok = not expected_intent or plan.intent == expected_intent
        coverage = (
            len(expected_tools & actual_tools) / len(expected_tools)
            if expected_tools else 1.0
        )
        intent_hits += int(intent_ok)
        coverage_values.append(coverage)
        results.append(
            {
                "id": case.get("id"),
                "query": case.get("query"),
                "expected_intent": expected_intent,
                "actual_intent": plan.intent,
                "expected_tools": sorted(expected_tools),
                "actual_tools": plan.tool_names,
                "intent_ok": intent_ok,
                "tool_coverage": round(coverage, 3),
            }
        )

    total = len(selected)
    return {
        "schema_version": "0.1",
        "evaluation": "planner",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": total,
        "intent_accuracy": round(intent_hits / total, 3) if total else 0.0,
        "tool_coverage": round(sum(coverage_values) / total, 3) if total else 0.0,
        "planning_pass_rate": round(
            sum(item["intent_ok"] and item["tool_coverage"] >= 1 for item in results) / total,
            3,
        ) if total else 0.0,
        "cases": results,
    }


def evaluate_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """从单次运行 trace 计算可观测质量指标。"""
    calls = [call for call in trace.get("tool_calls", []) if isinstance(call, dict)]
    successes = sum(bool(call.get("ok")) and not call.get("policy_blocked", False) for call in calls)
    answer = str(trace.get("answer") or "")
    reflection = reflect_answer(answer, trace)
    source_ids = set(reflection["evidence_ids"])
    cited_ids = set(extract_arxiv_ids(answer))
    citation_grounding = (
        None if not cited_ids else len(cited_ids & source_ids) / len(cited_ids)
    )
    budget = trace.get("budget") or {}
    max_steps = int(budget.get("max_steps") or 0)
    within_budget = len(calls) <= max_steps if max_steps else True
    plan_steps = (trace.get("plan") or {}).get("steps") or []
    planned_tools = {
        str(step.get("tool_hint"))
        for step in plan_steps
        if isinstance(step, dict) and step.get("tool_hint")
    }
    called_tools = {str(call.get("tool")) for call in calls
                    if call.get("tool") and call.get("ok") and not call.get("policy_blocked")}
    plan_tool_coverage = (
        len(planned_tools & called_tools) / len(planned_tools)
        if planned_tools else None
    )
    policy_violations = sum(bool(call.get("policy_blocked")) for call in calls)
    policy_compliant = policy_violations == 0
    max_output_budget = int(budget.get("max_total_output_tokens") or 0)
    output_tokens = (trace.get("usage") or {}).get("output_tokens", 0)
    within_output_budget = (
        int(output_tokens or 0) <= max_output_budget
        if max_output_budget else True
    )
    progress = plan_progress(trace)
    successful_run = (
        trace.get("status") == "completed" and bool(reflection.get("passed"))
        and progress["complete"] and within_budget and within_output_budget and policy_compliant
        and trace.get("task_status", "completed") == "completed"
        and (not reflection["evidence_required"] or bool(source_ids))
    )
    return {
        "run_id": trace.get("run_id"),
        "status": trace.get("status"),
        "tool_call_count": len(calls),
        "tool_success_count": successes,
        "tool_success_rate": round(successes / len(calls), 3) if calls else None,
        "citation_count": len(cited_ids),
        "source_count": len(source_ids),
        "grounded_citation_count": len(cited_ids & source_ids),
        "citation_grounding": round(citation_grounding, 3) if citation_grounding is not None else None,
        "reflection_passed": bool(reflection.get("passed")),
        "recorded_reflection_passed": (trace.get("reflection") or {}).get("passed"),
        "reflection_score": reflection.get("score", 0),
        "within_step_budget": within_budget,
        "plan_tool_coverage": round(plan_tool_coverage, 3) if plan_tool_coverage is not None else None,
        "plan_step_coverage": progress["coverage"],
        "required_step_count": progress["required"],
        "completed_step_count": progress["completed"],
        "missing_plan_steps": progress["missing_steps"],
        "task_status": trace.get("task_status"),
        "policy_compliant": policy_compliant,
        "policy_violation_count": policy_violations,
        "successful_run": successful_run,
        "input_tokens": (trace.get("usage") or {}).get("input_tokens", 0),
        "output_tokens": output_tokens,
        "within_output_budget": within_output_budget,
    }


def audit_agent_runs(
    run_dir: str | Path | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """审计已经保存的 Agent trace，不重新调用 LLM。"""
    directory = Path(run_dir) if run_dir else AGENT_RUNS_DIR
    traces: list[dict[str, Any]] = []
    invalid = 0
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                invalid += 1
                continue
            if isinstance(data, dict):
                traces.append(data)

    metrics = [evaluate_trace(trace) for trace in traces]
    n = len(metrics)

    def average(key: str) -> float | None:
        values = [float(item[key]) for item in metrics if item.get(key) is not None]
        return round(sum(values) / len(values), 3) if values else None

    result: dict[str, Any] = {
        "schema_version": "0.1",
        "evaluation": "agent_trace_audit",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(directory),
        "run_count": n,
        "invalid_trace_count": invalid,
        "sample_counts": {
            key: sum(item.get(key) is not None for item in metrics)
            for key in ("tool_success_rate", "citation_grounding", "plan_tool_coverage", "plan_step_coverage")
        },
        "averages": {
            "tool_success_rate": average("tool_success_rate"),
            "citation_grounding": average("citation_grounding"),
            "reflection_pass_rate": average("reflection_passed"),
            "plan_tool_coverage": average("plan_tool_coverage"),
            "plan_step_coverage": average("plan_step_coverage"),
            "policy_compliance_rate": average("policy_compliant"),
            "within_step_budget_rate": average("within_step_budget"),
            "within_output_budget_rate": average("within_output_budget"),
            "successful_run_rate": average("successful_run"),
        },
        "runs": metrics,
    }

    if save:
        out = OUTPUT_DIR / "reports" / f"agent_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["saved_to"] = str(out)
        except OSError as exc:
            result["save_error"] = str(exc)
            log.warning("Agent trace 评测结果保存失败：%s", exc)
    return result


def run_evaluation(audit_runs: bool = False, save: bool = True) -> dict[str, Any]:
    if audit_runs:
        return audit_agent_runs(save=save)

    result = evaluate_planner()
    if save:
        out = OUTPUT_DIR / "reports" / f"agent_planner_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["saved_to"] = str(out)
        except OSError as exc:
            result["save_error"] = str(exc)
            log.warning("规划评测结果保存失败：%s", exc)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent 运行时离线评测")
    parser.add_argument("--audit-runs", action="store_true", help="审计 output/agent_runs 中的历史 trace")
    args = parser.parse_args()
    print(json.dumps(run_evaluation(audit_runs=args.audit_runs), ensure_ascii=False, indent=2))
