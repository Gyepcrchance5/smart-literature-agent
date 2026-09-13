from __future__ import annotations

import sys
import unittest
import anthropic
import httpx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_runtime import (  # noqa: E402
    AgentRuntime,
    MemoryStore,
    TaskPlanner,
    ToolRegistry,
    ToolSpec,
    extract_arxiv_ids,
    reflect_answer,
    plan_progress,
)


class _Block:
    def __init__(self, block_type: str, **values):
        self.type = block_type
        for key, value in values.items():
            setattr(self, key, value)


class _Message:
    def __init__(self, blocks, input_tokens=10, output_tokens=5):
        self.content = blocks
        self.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}


class _FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("fake client responses exhausted")
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


class _FailingMessages:
    def __init__(self, error):
        self.error = error

    def create(self, **kwargs):
        raise self.error


class _FailingClient:
    def __init__(self, error):
        self.messages = _FailingMessages(error)


class AgentRuntimeTests(unittest.TestCase):
    def _run_script(self, query, responses, output=None, **runtime_options):
        def handler(args):
            if isinstance(output, Exception):
                raise output
            return output if output is not None else {"arxiv_id": "2411.11707"}

        registry = ToolRegistry([
            ToolSpec(name, name, {"type": "object", "properties": {}, "additionalProperties": False},
                     handler)
            for name in ("read_paper", "audit_literature", "retrieve_literature")
        ])
        memory = MemoryStore(persist=False)
        client = _FakeClient(responses)
        trace = AgentRuntime(registry=registry, client=client, memory_store=memory,
                             save_trace=False, **runtime_options).run(query, "regression")
        return trace, memory.context("regression"), client.messages.calls

    def test_no_tools_cannot_complete_task_or_enter_verified_memory(self):
        trace, memory, calls = self._run_script("知识蒸馏有什么方法？", [
            _Message([_Block("text", text="知识蒸馏可以提升表现。")])])
        self.assertEqual(trace["status"], "completed")
        self.assertEqual(trace["task_status"], "incomplete")
        self.assertFalse(trace["reflection"]["passed"])
        self.assertEqual(memory[0]["answer"], "")
        self.assertEqual(len(calls), 1)  # Text repair cannot execute missing tools.

    def test_skipped_audit_does_not_become_success_via_text_repair(self):
        trace, memory, calls = self._run_script("论文 2411.11707 值得作为研究依据吗？", [
            _Message([_Block("tool_use", id="t1", name="read_paper", input={})]),
            _Message([_Block("text", text="根据 [2411.11707]，值得参考。")])])
        self.assertEqual(trace["task_status"], "incomplete")
        self.assertEqual(trace["plan_progress"]["missing_steps"], ["audit"])
        self.assertFalse(trace["reflection"]["passed"])
        self.assertEqual(memory[0]["citations"], "")
        self.assertEqual(len(calls), 2)

    def test_completed_review_enters_verified_memory(self):
        trace, memory, _ = self._run_script("论文 2411.11707 值得作为研究依据吗？", [
            _Message([_Block("tool_use", id="t1", name="read_paper", input={})]),
            _Message([_Block("tool_use", id="t2", name="audit_literature", input={})]),
            _Message([_Block("text", text="根据 [2411.11707]，仍需人工复核。")])])
        self.assertEqual(trace["task_status"], "completed")
        self.assertTrue(trace["reflection"]["passed"])
        self.assertEqual(memory[0]["verified"], "yes")

    def test_empty_retrieval_is_insufficient_evidence_not_verified_memory(self):
        trace, memory, _ = self._run_script("知识蒸馏有什么方法？", [
            _Message([_Block("tool_use", id="t1", name="retrieve_literature", input={})]),
            _Message([_Block("text", text="证据不足，无法确认。")])], output={"papers": []})
        self.assertTrue(trace["plan_progress"]["complete"])
        self.assertEqual(trace["task_status"], "insufficient_evidence")
        self.assertEqual(memory[0]["verified"], "no")

    def test_failed_tools_cannot_supply_citations(self):
        result = reflect_answer("参考 [2411.11707]，证据不足。", {
            "status": "completed", "tool_calls": [{"ok": False, "output": {"arxiv_id": "2411.11707"}}]})
        self.assertEqual(result["evidence_ids"], [])
        self.assertFalse(result["checks"]["citations_grounded"])

    def test_tool_exception_leaves_incomplete_task_and_empty_memory(self):
        trace, memory, _ = self._run_script("知识蒸馏有什么方法？", [
            _Message([_Block("tool_use", id="t1", name="retrieve_literature", input={})]),
            _Message([_Block("text", text="工具失败，证据不足。")])], output=TimeoutError("synthetic timeout"))
        self.assertEqual(trace["task_status"], "incomplete")
        self.assertFalse(trace["tool_calls"][0]["ok"])
        self.assertEqual(memory[0]["answer"], "")

    def test_duplicate_call_is_not_a_verified_task(self):
        trace, memory, _ = self._run_script("知识蒸馏有什么方法？", [
            _Message([_Block("tool_use", id="t1", name="retrieve_literature", input={})]),
            _Message([_Block("tool_use", id="t2", name="retrieve_literature", input={})]),
            _Message([_Block("text", text="参考 [2411.11707]，存在风险。")])])
        self.assertTrue(trace["plan_progress"]["complete"])
        self.assertTrue(trace["tool_calls"][1]["policy_blocked"])
        self.assertEqual(trace["task_status"], "needs_review")
        self.assertEqual(memory[0]["answer"], "")

    def test_model_error_does_not_complete_task(self):
        trace, memory, _ = self._run_script("知识蒸馏有什么方法？", [])
        self.assertEqual(trace["status"], "llm_error")
        self.assertEqual(trace["task_status"], "failed")
        self.assertEqual(memory[0]["answer"], "")

    def test_401_is_recorded_as_non_retryable_authentication_failure(self):
        request = httpx.Request("POST", "https://provider.invalid/anthropic")
        response = httpx.Response(401, request=request)
        error = anthropic.AuthenticationError(
            "invalid credential",
            response=response,
            body={"error": {"message": "invalid credential"}},
        )
        registry = ToolRegistry([
            ToolSpec(name, name, {"type": "object", "properties": {}, "additionalProperties": False},
                     lambda args: {"arxiv_id": "2411.11707"})
            for name in ("read_paper", "audit_literature", "retrieve_literature")
        ])
        memory = MemoryStore(persist=False)
        runtime = AgentRuntime(
            registry=registry,
            client=_FailingClient(error),
            memory_store=memory,
            save_trace=False,
            repair_failed_reflection=False,
        )
        trace = runtime.run("知识蒸馏有什么方法？", "auth-error")
        self.assertEqual(trace["status"], "llm_error")
        self.assertEqual(trace["error_category"], "authentication")
        self.assertEqual(trace["error_info"]["status_code"], 401)
        self.assertFalse(trace["error_info"]["retryable"])
        self.assertFalse(trace["error_info"]["fallback_attempted"])
        self.assertEqual(memory.context("auth-error")[0]["answer"], "")

    def test_output_budget_exhaustion_does_not_reuse_answer(self):
        trace, memory, _ = self._run_script("知识蒸馏有什么方法？", [
            _Message([_Block("text", text="忽略超额输出")], output_tokens=300)],
            max_output_tokens=256, max_total_output_tokens=256)
        self.assertEqual(trace["termination_reason"], "output_token_budget_exhausted")
        self.assertEqual(trace["task_status"], "failed")
        self.assertEqual(memory[0]["answer"], "")

    def test_plan_progress_rejects_cached_completion_and_wrong_execution_order(self):
        trace = {"plan": {"steps": [
            {"id": "read", "tool_hint": "read_paper", "depends_on": []},
            {"id": "audit", "tool_hint": "audit_literature", "depends_on": ["read"]}]},
            "completed_plan_steps": ["read", "audit"], "tool_calls": [
                {"ok": True, "tool": "audit_literature", "plan_step_id": "audit"},
                {"ok": True, "tool": "read_paper", "plan_step_id": "read"}]}
        self.assertEqual(plan_progress(trace)["missing_steps"], ["audit"])
        trace["tool_calls"].reverse()
        self.assertTrue(plan_progress(trace)["complete"])

    def test_legacy_memory_and_failed_status_cannot_be_reused(self):
        memory = MemoryStore(persist=False)
        memory.append_turn("old", "q", "legacy answer", [], [], True, "completed", "r1")
        del memory.load("old")["turns"][0]["task_status"]
        self.assertEqual(memory.context("old")[0]["answer"], "")
        memory.append_turn("failed", "q", "bad answer", [], [], True, "llm_error", "r2")
        self.assertEqual(memory.context("failed")[0]["answer"], "")

    def test_planner_routes_core_intents(self):
        planner = TaskPlanner()
        self.assertEqual(planner.plan("这周有什么新论文？").intent, "discovery")
        self.assertEqual(planner.plan("结合我的项目瓶颈判断迁移方案").intent, "project_match")
        self.assertEqual(planner.plan("2411.11707 值得作为依据吗？").intent, "paper_review")
        self.assertEqual(planner.plan("哪些论文证据更完整？").intent, "quality_audit")
        comparison = planner.plan("比较 2411.11707 和 2501.00001 的方法差异")
        self.assertEqual(comparison.intent, "synthesis")
        self.assertEqual(comparison.tool_names, ["compare_papers"])

    def test_registry_validates_and_isolates_tool_errors(self):
        registry = ToolRegistry(
            [
                ToolSpec(
                    "echo",
                    "echo",
                    {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                    lambda args: {"echo": args["text"]},
                )
            ]
        )
        ok = registry.execute("echo", {"text": "hi"}, "t1")
        self.assertTrue(ok["ok"])
        bad = registry.execute("echo", {"text": 3}, "t2")
        self.assertFalse(bad["ok"])
        unknown = registry.execute("missing", {}, "t3")
        self.assertFalse(unknown["ok"])

        blocked = registry.execute(
            "echo", {"text": "hi"}, "t4", allowed_tools={"other"}
        )
        self.assertFalse(blocked["ok"])
        self.assertEqual(blocked["policy_reason"], "tool_not_in_plan")

        limited = registry.execute("echo", {"text": "hi"}, "t5", prior_calls=2)
        self.assertFalse(limited["ok"])
        self.assertEqual(limited["policy_reason"], "tool_call_limit")

    def test_memory_is_session_scoped_and_truncated(self):
        store = MemoryStore(max_turns=2, max_chars=30, persist=False)
        path = store.append_turn("demo/session", "q" * 100, "a" * 100, ["2411.11707"], ["echo"], True, "completed", "r1")
        self.assertEqual(path, "memory://demo-session")
        data = store.load("demo/session")
        self.assertEqual(data["session_id"], "demo-session")
        self.assertLessEqual(len(data["turns"][0]["query"]), 31)
        self.assertEqual(store.context("demo/session")[0]["citations"], "2411.11707")

        store.append_turn(
            "demo/session", "bad", "未经验证的结论", ["2501.00001"], [], False, "llm_error", "r2"
        )
        latest = store.context("demo/session")[-1]
        self.assertEqual(latest["answer"], "")
        self.assertEqual(latest["citations"], "")
        self.assertEqual(latest["verified"], "no")

    def test_runtime_executes_tool_then_answers_with_grounded_citation(self):
        registry = ToolRegistry(
            [
                ToolSpec(
                    "retrieve_literature",
                    "retrieve",
                    {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                    lambda args: {"papers": [{"arxiv_id": "2411.11707", "title": "Demo"}]},
                )
            ]
        )
        fake = _FakeClient(
            [
                _Message([_Block("tool_use", id="tool-1", name="retrieve_literature", input={"question": "剪枝"})]),
                _Message([_Block("text", text="基于证据 [2411.11707]，建议先做小规模验证。")]),
            ]
        )
        runtime = AgentRuntime(
            registry=registry,
            client=fake,
            memory_store=MemoryStore(persist=False),
            save_trace=False,
            repair_failed_reflection=False,
        )
        trace = runtime.run("帮我找剪枝方法", session_id="demo")
        self.assertEqual(trace["status"], "completed")
        self.assertEqual(len(trace["tool_calls"]), 1)
        self.assertTrue(trace["tool_calls"][0]["ok"])
        self.assertTrue(trace["reflection"]["passed"])
        self.assertEqual(trace["usage"]["llm_turns"], 2)
        self.assertEqual(len(fake.messages.calls), 2)

    def test_reflection_marks_unsupported_citation(self):
        trace = {"tool_calls": [{"ok": True, "output": {"arxiv_id": "2411.11707"}}]}
        result = reflect_answer("结论来自 [2501.00001]。", trace)
        self.assertFalse(result["passed"])
        self.assertEqual(result["unsupported_citations"], ["2501.00001"])
        self.assertEqual(extract_arxiv_ids("见 2411.11707v2"), ["2411.11707v2"])

    def test_runtime_enforces_plan_dependencies(self):
        registry = ToolRegistry(
            [
                ToolSpec(
                    "read_paper",
                    "read",
                    {"type": "object", "properties": {}, "additionalProperties": False},
                    lambda args: {"arxiv_id": "2411.11707"},
                ),
                ToolSpec(
                    "audit_literature",
                    "audit",
                    {"type": "object", "properties": {}, "additionalProperties": False},
                    lambda args: {"quality": "A"},
                ),
            ]
        )
        fake = _FakeClient(
            [
                _Message([_Block("tool_use", id="tool-1", name="audit_literature", input={})]),
                _Message([_Block("text", text="前置步骤未完成，证据不足，请人工复核。")]),
            ]
        )
        runtime = AgentRuntime(
            registry=registry,
            client=fake,
            memory_store=MemoryStore(persist=False),
            save_trace=False,
            repair_failed_reflection=False,
        )
        trace = runtime.run("论文 2411.11707 值得作为研究依据吗？", session_id="dependency")
        self.assertEqual(trace["status"], "completed")
        self.assertEqual(trace["tool_calls"][0]["policy_reason"], "plan_dependency")
        self.assertEqual(trace["completed_plan_steps"], [])
        self.assertFalse(trace["reflection"]["passed"])

    def test_runtime_finalizes_after_tool_budget_without_new_tools(self):
        registry = ToolRegistry(
            [
                ToolSpec(
                    "retrieve_literature",
                    "retrieve",
                    {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                        "additionalProperties": False,
                    },
                    lambda args: {"papers": [{"arxiv_id": "2411.11707"}]},
                )
            ]
        )
        fake = _FakeClient(
            [
                _Message([_Block("tool_use", id="tool-1", name="retrieve_literature", input={"question": "剪枝"})]),
                _Message([_Block("tool_use", id="tool-2", name="retrieve_literature", input={"question": "蒸馏"})]),
                _Message([_Block("text", text="基于已检索证据 [2411.11707]，建议先做小规模验证。")]),
            ]
        )
        runtime = AgentRuntime(
            registry=registry,
            client=fake,
            memory_store=MemoryStore(persist=False),
            max_steps=1,
            max_llm_turns=2,
            save_trace=False,
            repair_failed_reflection=False,
        )
        trace = runtime.run("帮我找剪枝方法", session_id="finalize")
        self.assertEqual(trace["status"], "completed")
        self.assertEqual(trace["termination_reason"], "text_finalization")
        self.assertTrue(trace["finalization_attempted"])
        self.assertTrue(trace["reflection"]["passed"])
        self.assertEqual(len(trace["tool_calls"]), 1)


if __name__ == "__main__":
    unittest.main()
