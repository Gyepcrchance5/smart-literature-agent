from __future__ import annotations

import sys
import unittest
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_eval import evaluate_planner, evaluate_trace, audit_agent_runs  # noqa: E402


class AgentEvalTests(unittest.TestCase):
    def test_builtin_planner_cases_pass(self):
        result = evaluate_planner()
        self.assertGreaterEqual(result["intent_accuracy"], 0.75)
        self.assertGreaterEqual(result["tool_coverage"], 0.75)

    def test_trace_metrics_are_bounded(self):
        result = evaluate_trace(
            {
                "run_id": "demo",
                "status": "completed",
                "plan": {"steps": [{"id": "retrieve", "tool_hint": "retrieve_literature"}]},
                "tool_calls": [{"ok": True, "tool": "retrieve_literature", "plan_step_id": "retrieve",
                                "output": {"arxiv_id": "2411.11707"}}],
                "source_ids": ["2411.11707"],
                "answer": "参考 [2411.11707]。",
                "reflection": {"passed": True, "score": 100},
                "budget": {"max_steps": 2, "max_total_output_tokens": 10},
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )
        self.assertEqual(result["tool_success_rate"], 1.0)
        self.assertEqual(result["citation_grounding"], 1.0)
        self.assertTrue(result["within_step_budget"])
        self.assertEqual(result["plan_tool_coverage"], 1.0)
        self.assertTrue(result["policy_compliant"])
        self.assertTrue(result["within_output_budget"])
        self.assertTrue(result["successful_run"])

    def test_no_samples_are_null_not_perfect_scores(self):
        result = evaluate_trace({"status": "llm_error", "tool_calls": []})
        self.assertIsNone(result["tool_success_rate"])
        self.assertIsNone(result["citation_grounding"])
        self.assertIsNone(result["plan_step_coverage"])
        self.assertFalse(result["successful_run"])

    def test_historical_success_flag_does_not_override_missing_plan(self):
        result = evaluate_trace({"status": "completed", "reflection": {"passed": True}})
        self.assertFalse(result["successful_run"])

    def test_failed_tool_is_not_plan_coverage(self):
        result = evaluate_trace({"status": "completed", "reflection": {"passed": True},
            "plan": {"steps": [{"id": "r", "tool_hint": "read_paper"}]},
            "tool_calls": [{"tool": "read_paper", "plan_step_id": "r", "ok": False}]})
        self.assertEqual(result["plan_tool_coverage"], 0)
        self.assertEqual(result["completed_step_count"], 0)
        self.assertFalse(result["successful_run"])

    def test_aggregate_excludes_null_and_preserves_sample_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "empty.json").write_text(json.dumps({"tool_calls": []}), encoding="utf-8")
            (root / "success.json").write_text(json.dumps({"tool_calls": [{"ok": True}]}), encoding="utf-8")
            result = audit_agent_runs(root, save=False)
        self.assertEqual(result["averages"]["tool_success_rate"], 1.0)
        self.assertEqual(result["sample_counts"]["tool_success_rate"], 1)
        self.assertIsNone(result["averages"]["citation_grounding"])
        self.assertEqual(result["sample_counts"]["citation_grounding"], 0)

    def test_empty_audit_has_no_success_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            result = audit_agent_runs(directory, save=False)
        self.assertEqual(result["run_count"], 0)
        self.assertIsNone(result["averages"]["successful_run_rate"])

    def test_audit_recomputes_old_reflection_and_ignores_failed_sources(self):
        result = evaluate_trace({"status": "completed", "answer": "参考 [2411.11707]。",
            "reflection": {"passed": True}, "source_ids": ["2411.11707"],
            "tool_calls": [{"ok": False, "output": {"arxiv_id": "2411.11707"}}]})
        self.assertTrue(result["recorded_reflection_passed"])
        self.assertFalse(result["reflection_passed"])
        self.assertEqual(result["citation_grounding"], 0)


if __name__ == "__main__":
    unittest.main()
