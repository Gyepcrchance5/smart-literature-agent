from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project_analyzer


class ProjectAnalyzerTests(unittest.TestCase):
    def test_migrated_layout_respects_scope_and_distinguishes_inventory_from_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = root / "work/mainline/code/core"
            core.mkdir(parents=True)
            (core / "models.py").write_text("class Student:\n    pass\n", encoding="utf-8")
            (core / "loss_functions.py").write_text("class DisabledLoss:\n    pass\n", encoding="utf-8")
            (root / "README.md").write_text("Frozen bearing diagnosis study", encoding="utf-8")
            (root / "NOW.md").write_text("当前瓶颈：未验证外部泛化", encoding="utf-8")
            config = root / "work/mainline/configs"
            config.mkdir()
            (config / "official_mainline.json").write_text(json.dumps({
                "dataset": "synthetic bearing dataset", "distillation": {"disabled": ["DisabledLoss"]},
                "pruning": {"official": {"name": "matched_taylor", "gamma_risk": 0.0}},
                "private_checkpoint": "must not be copied"}), encoding="utf-8")
            for name in ("assets", "archive", "work/mainline/runs", "work/explorations"):
                excluded = root / name
                excluded.mkdir(parents=True, exist_ok=True)
                (excluded / "models.py").write_text("raise RuntimeError('never read or execute')", encoding="utf-8")
            read = project_analyzer._safe_read
            with patch.object(project_analyzer, "_safe_read", wraps=read) as reader:
                result = project_analyzer.scan_project(root)
            paths = [call.args[0].relative_to(root.resolve()).as_posix() for call in reader.call_args_list]
            self.assertFalse(any(path.startswith(("assets/", "archive/", "work/mainline/runs/", "work/explorations/")) for path in paths))
            self.assertEqual([item["name"] for item in result["models"]], ["Student"])
            self.assertEqual(result["official_protocol"]["distillation"]["disabled"], ["DisabledLoss"])
            self.assertNotIn("private_checkpoint", result["official_protocol"])
            self.assertIn("未验证外部泛化", result["bottlenecks"])
            self.assertFalse(result["scan_scope"]["recursive"])
            self.assertIn("未读取 run", result["evidence_boundary"])
            summary = project_analyzer._build_project_summary(result)
            self.assertIn('"disabled"', summary)
            prompt = project_analyzer._build_match_prompt(summary, {})
            self.assertNotIn("项目已有 KD+ARKD+CLD+CenterLoss", prompt)
            self.assertIn("正式配置里的启用/禁用声明优先", prompt)

    def test_legacy_core_layout_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "core").mkdir()
            (root / "core/models.py").write_text("class OldStudent:\n    pass\n", encoding="utf-8")
            profile = project_analyzer.scan_project(root)
            self.assertEqual(profile["models"][0]["name"], "OldStudent")
            self.assertEqual(profile["scan_scope"]["code_directories"], ["core"])

    def test_malformed_protocol_does_not_break_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "work/mainline/configs"
            config.mkdir(parents=True)
            (config / "official_mainline.json").write_text("[]", encoding="utf-8")
            result = project_analyzer.scan_project(root)
            self.assertIn("protocol_error", result)
            self.assertEqual(result["official_protocol"], {})


if __name__ == "__main__":
    unittest.main()
