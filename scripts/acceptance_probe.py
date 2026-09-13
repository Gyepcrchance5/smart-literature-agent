"""Acceptance probes; no model/network calls unless --live is specified.

Writes only to the explicitly selected output directory. Synthetic traces are
kept separate from real-model traces and never added to production memory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_runtime import AgentRuntime, MemoryStore, ToolRegistry, ToolSpec, build_default_registry
from agent_eval import evaluate_trace
from qa_agent import load_knowledge_base


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = self

    def create(self, **kwargs):
        return {"content": next(self.responses), "usage": {"input_tokens": 0, "output_tokens": 0}}


def text_block(text):
    return {"type": "text", "text": text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--project-path", type=Path, help="Optional local read-only project scan (offline only)")
    args = parser.parse_args()
    if args.live and args.project_path:
        parser.error("Project scans are offline only; do not send private project content in live acceptance")
    args.out.mkdir(parents=True, exist_ok=True)
    # A real write/read check, before spending any model budget.
    probe = args.out / "write_probe.txt"
    probe.write_text("acceptance", encoding="utf-8")
    assert probe.read_text(encoding="utf-8") == "acceptance"
    kb = load_knowledge_base()
    ids = [p["arxiv_id"] for p in kb if (ROOT / "data" / "papers" / (p["arxiv_id"].replace("/", "_") + ".json")).exists()]
    results = {"mode": "live" if args.live else "offline", "library_count": len(kb), "cases": []}
    if args.live:
        if len(ids) < 2:
            raise SystemExit("Live comparison requires two local papers with enrichment and reading artifacts")
        from summarizer import Anthropic
        client = Anthropic(timeout=45.0, max_retries=0)
        queries = [
            "从本地知识库中找知识蒸馏方法，列出两篇论文及证据局限。",
            f"比较 {ids[0]} 和 {ids[1]} 的方法差异；没有的指标请明确说未知。",
        ]
        for i, query in enumerate(queries):
            start = time.monotonic()
            trace = AgentRuntime(client=client, memory_store=MemoryStore(persist=False), save_trace=False,
                                 max_steps=3, max_llm_turns=4, max_output_tokens=1600,
                                 max_total_output_tokens=4800).run(query, session_id=f"acceptance-live-{i}")
            trace["acceptance_mode"] = "live"
            trace["wall_seconds"] = round(time.monotonic() - start, 2)
            # Raw error text can contain provider details; do not publish traces.
            (args.out / f"live_{i}.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
            results["cases"].append({"id": f"live-{i}", **evaluate_trace(trace), "wall_seconds": trace["wall_seconds"],
                                     "tools": [c["tool"] for c in trace["tool_calls"]],
                                     "error_type": str(trace.get("error", "")).split(":", 1)[0]})
            print(json.dumps(results["cases"][-1], ensure_ascii=False), flush=True)
            if trace["status"] in {"configuration_error", "llm_error"}:
                break
    else:
        spec = lambda name: ToolSpec(name, name, {"type": "object", "properties": {}, "additionalProperties": False},
                                    lambda _: {"arxiv_id": "2411.11707", "title": "Synthetic evidence: no experimental metrics"})
        registry = ToolRegistry([spec("read_paper"), spec("audit_literature"), spec("retrieve_literature")])
        cases = [
            ("no_tools", "知识蒸馏有什么方法？", [[text_block("知识蒸馏能提升模型表现。")]]),
            ("skipped_audit", "论文 2411.11707 值得作为研究依据吗？", [
                [{"type": "tool_use", "id": "t1", "name": "read_paper", "input": {}}],
                [text_block("根据 [2411.11707]，值得作为研究依据。")]]),
            ("unsupported_claim_valid_id", "知识蒸馏有什么方法？", [
                [{"type": "tool_use", "id": "t1", "name": "retrieve_literature", "input": {}}],
                [text_block("[2411.11707] 在 CWRU 上达到 99.99% 准确率。")]]),
        ]
        for name, query, responses in cases:
            trace = AgentRuntime(registry=registry, client=ScriptedClient(responses),
                                 memory_store=MemoryStore(persist=False), save_trace=False,
                                 repair_failed_reflection=False).run(query, name)
            results["cases"].append({"id": name, "synthetic": True, **evaluate_trace(trace),
                                     "completed_plan_steps": trace["completed_plan_steps"]})
        default = build_default_registry()
        for name, params in [
            ("retrieve_literature", {"question": "知识蒸馏", "top_k": 3}),
            ("read_paper", {"arxiv_id": ids[0]}),
            ("compare_papers", {"arxiv_ids": ids[:2]}),
            ("read_paper", {"arxiv_id": "9999.99999"}),
        ]:
            result = default.execute(name, params, "acceptance")
            results["cases"].append({"id": name, "params": params, "ok": result["ok"],
                                     "output": result.get("output"), "error": result.get("error")})
        memory_root = args.out / "synthetic_memory"
        memory = MemoryStore(root=memory_root)
        saved = memory.append_turn("acceptance", "synthetic question", "verified synthetic answer",
                                   ["2411.11707"], [], True, "completed", "synthetic-1")
        memory.append_turn("acceptance", "failed synthetic question", "must not be reused",
                           [], [], False, "llm_error", "synthetic-2")
        restored = MemoryStore(root=memory_root)
        context = restored.context("acceptance")
        results["cases"].append({"id": "memory_disk_roundtrip", "synthetic": True,
                                 "ok": bool(saved) and len(context) == 2
                                 and context[0]["answer"] == "verified synthetic answer"
                                 and context[1]["answer"] == ""
                                 and restored.context("different-session") == []})
        if args.project_path:
            from project_analyzer import scan_project
            profile = scan_project(args.project_path)
            (args.out / "project_profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
            results["cases"].append({"id": "project_scan", "local_only": True,
                                     "model_definitions": len(profile["models"]),
                                     "loss_definitions": len(profile["losses"]),
                                     "data_loader_definitions": len(profile["data_loaders"]),
                                     "file_count": len(profile["key_files"]),
                                     "scan_scope": profile["scan_scope"],
                                     "has_official_protocol": bool(profile["official_protocol"])})
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"mode": results["mode"], "cases": len(results["cases"]), "saved_to": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
