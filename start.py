#!/usr/bin/env python3
"""smart-literature-agent 启动器 —— 你的「开始按钮」。

在项目根目录运行：
    python start.py

无需记任何 CLI flag，所有操作从菜单选择。
"""

from __future__ import annotations

import json
import os
import sys

# Windows GBK terminal workaround：强制 stdout/stderr 用 UTF-8
if os.name == "nt":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _count_files(pattern: str) -> int:
    return len(list((PROJECT_ROOT / pattern).parent.glob(
        Path(pattern).name
    )))


# ── status checks ──────────────────────────────────────────────


def _check_env() -> tuple[bool, str]:
    """Check if API key is configured."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return True, f"已配置 (sk-...{key[-8:]})"
    # check .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return True, f"已配置 (.env, sk-...{val[-8:]})"
    return False, "未配置 — 请编辑 .env 文件"


def _check_historical_pool() -> tuple[bool, int, int]:
    """(has_pool, total, remaining_unread)."""
    path = PROJECT_ROOT / "data" / "candidates_historical_pool.json"
    data = _read_json(path)
    if not data:
        return False, 0, 0
    total = data.get("total", len(data.get("papers", [])))
    seen = set(json.loads(
        (PROJECT_ROOT / "data" / "seen_ids.json").read_text(encoding="utf-8")
    )) if (PROJECT_ROOT / "data" / "seen_ids.json").exists() else set()
    papers = data.get("papers", [])
    remaining = sum(1 for p in papers if p.get("arxiv_id") not in seen)
    return True, total, remaining


def _check_last_run() -> str:
    logs = sorted((PROJECT_ROOT / "logs").glob("*.log"))
    if not logs:
        return "从未运行"
    log = logs[-1]
    try:
        text = log.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = log.read_text(encoding="gbk")
        except UnicodeDecodeError:
            return "有日志但编码无法识别"
    for line in reversed(text.splitlines()):
        if "PIPELINE RUN START" in line:
            try:
                ts = line.split("@")[1].strip()
                dt = datetime.fromisoformat(ts)
                return dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, IndexError):
                pass
    return "有日志但无法解析"


def _gather_status() -> dict:
    """Collect all status info for the dashboard."""
    env_ok, env_msg = _check_env()
    has_hist, hist_total, hist_remaining = _check_historical_pool()

    latest_candidates = sorted(
        (PROJECT_ROOT / "data").glob("candidates_20*.json")
    )
    candidates_info = ""
    if latest_candidates:
        data = _read_json(latest_candidates[-1])
        if data:
            candidates_info = f"{data.get('total', '?')} 篇 ({latest_candidates[-1].name})"

    # LLM provider info
    from utils import get_llm_config
    llm_cfg = get_llm_config()
    llm_info = f"{llm_cfg['provider_label']} / {llm_cfg['model']}"

    return {
        "env_ok": env_ok,
        "env_msg": env_msg,
        "llm_info": llm_info,
        "has_hist": has_hist,
        "hist_total": hist_total,
        "hist_remaining": hist_remaining,
        "last_run": _check_last_run(),
        "candidates_info": candidates_info,
        "seen_count": len(json.loads(
            (PROJECT_ROOT / "data" / "seen_ids.json").read_text(encoding="utf-8")
        )) if (PROJECT_ROOT / "data" / "seen_ids.json").exists() else 0,
        "summary_count": _count_files("output/papers/*.summary.md"),
        "formula_count": _count_files("output/papers/*.formulas.json"),
        "field_report_count": _count_files("output/reports/*.md"),
        "html_exists": (PROJECT_ROOT / "output" / "html" / "index.html").exists(),
    }


# ── display ────────────────────────────────────────────────────


def _show_dashboard(s: dict) -> None:
    print()
    print("=" * 56)
    print("  smart-literature-agent -- 科研文献智能体")
    print("=" * 56)
    print(f"  API Key    : {s['env_msg']}")
    print(f"  LLM        : {s['llm_info']}")
    print(f"  上次运行   : {s['last_run']}")
    print(f"  已精读论文 : {s['seen_count']} 篇")
    print(f"  单篇摘要   : {s['summary_count']} 篇")
    print(f"  公式提取   : {s['formula_count']} 篇")
    print(f"  领域综述   : {s['field_report_count']} 份")
    if s["candidates_info"]:
        print(f"  本周候选   : {s['candidates_info']}")
    if s["has_hist"]:
        warn = " [!] 池子快空了，建议重建" if s["hist_remaining"] < 20 else ""
        print(f"  历史池     : {s['hist_total']} 篇总量, {s['hist_remaining']} 篇未读{warn}")
    else:
        print(f"  历史池     : 未创建（菜单选项 3 可创建）")
    if s["html_exists"]:
        print(f"  报告       : output/html/index.html 可用")
    print("=" * 56)
    print()


def _show_menu() -> None:
    print("请选择操作：")
    print()
    print("  [1] > 运行完整流水线")
    print("       search → read → formulas → summarize → report → synthesis")
    print()
    print("  [2] > 快速运行（跳过公式提取和综合创新）")
    print("       search → read → summarize → report（快 ~2 分钟）")
    print()
    print("  [3] [H] 构建/重建历史论文池")
    print("       一次性回溯 5 年，之后每次自动取 30% 混入阅读")
    print()
    print("  [4] [R] 仅重新生成报告（不搜不读不调 LLM）")
    print("       TOP10 + 综合创新分析 + HTML 重渲染")
    print()
    print("  [5] [O] 打开最新报告（浏览器）")
    print()
    print("  [6] [C]  调整运行参数（max_read / 历史比例 / top_n）")
    print()
    print("  [0] 退出")
    print()


def _show_params(params: dict) -> None:
    print()
    print("当前运行参数：")
    print(f"  每次精读上限 (max_read) : {params['max_read']}")
    print(f"  历史论文比例            : {params['historical_ratio']}")
    print(f"  TOP N 合并数            : {params['top_n']}")
    print(f"  自动打开浏览器          : {'是' if params['auto_open'] else '否'}")
    print()


# ── actions ────────────────────────────────────────────────────


def _run_full_pipeline(params: dict) -> None:
    from run import pipeline_run
    print()
    print("=" * 60)
    print("  启动完整流水线...")
    _show_params(params)
    print("=" * 60)
    pipeline_run(
        max_read=params["max_read"],
        historical_ratio=params["historical_ratio"],
        top_n=params["top_n"],
        auto_open=params["auto_open"],
    )
    print()
    print("[OK] 流水线完成。运行 python start.py 查看最新状态。")


def _run_quick(params: dict) -> None:
    from run import pipeline_run
    print()
    print("=" * 60)
    print("  启动快速运行（跳过公式 + 综合创新）...")
    _show_params(params)
    print("=" * 60)
    pipeline_run(
        max_read=params["max_read"],
        historical_ratio=params["historical_ratio"],
        top_n=params["top_n"],
        auto_open=params["auto_open"],
        skip_formulas=True,
        skip_synthesis=True,
    )


def _run_historical() -> None:
    from searcher import build_historical_pool
    print()
    print("=" * 60)
    print("  构建历史论文池（5 年回溯）...")
    print("  这需要调 42 个关键词 × 20 条结果，预计 2-4 分钟")
    print("=" * 60)
    build_historical_pool()
    print()
    print("[OK] 历史池构建完成。之后每次运行会自动从中取论文。")


def _run_report_only(params: dict) -> None:
    from run import pipeline_run
    print()
    print("=" * 60)
    print("  仅重新生成报告...")
    print("=" * 60)
    pipeline_run(
        max_read=0,
        skip_search=True,
        skip_llm=True,
        top_n=params["top_n"],
        auto_open=params["auto_open"],
    )


def _open_report() -> None:
    import os as _os
    html_path = PROJECT_ROOT / "output" / "html" / "index.html"
    if not html_path.exists():
        print("[X] 还没有生成报告，先运行流水线。")
        return
    if _os.name == "nt":
        _os.startfile(str(html_path))
    else:
        import webbrowser
        webbrowser.open(html_path.resolve().as_uri())
    print("已在浏览器打开报告。")


def _switch_provider() -> None:
    from utils import get_llm_config, load_keywords
    providers = load_keywords().get("providers", {})
    if not providers:
        print("keywords.yaml 未配置 providers，无法切换。")
        return

    current = get_llm_config()["provider_key"]
    print()
    print("可用 LLM 提供商：")
    keys = list(providers.keys())
    for i, (k, p) in enumerate(providers.items(), 1):
        marker = " ← 当前" if k == current else ""
        print(f"  [{i}] {p['label']} ({p.get('models', [])}){marker}")

    try:
        choice = input("选择提供商 (数字，回车返回): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not choice:
        return
    try:
        idx = int(choice) - 1
        new_key = keys[idx]
    except (IndexError, ValueError):
        print("无效选择")
        return

    # Write LLM_PROVIDER to .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        found = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("LLM_PROVIDER="):
                new_lines.append(f"LLM_PROVIDER={new_key}")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"LLM_PROVIDER={new_key}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    # Also set in current process
    os.environ["LLM_PROVIDER"] = new_key
    print(f"LLM 提供商已切换为: {providers[new_key]['label']}")
    print("(模型已自动选择默认值，下次运行 start.py 生效)")


def _switch_model() -> None:
    from utils import get_llm_config, load_keywords
    llm = get_llm_config()
    provider_key = llm["provider_key"]
    providers = load_keywords().get("providers", {})
    p = providers.get(provider_key) if provider_key else None
    if not p:
        print("当前使用自定义配置，无模型列表可切换。请在 .env 中设置 LLM_PROVIDER。")
        return

    models = p.get("models", [])
    if not models:
        print("当前提供商无预设模型列表。")
        return

    print()
    print(f"{p['label']} 可用模型：")
    for i, m in enumerate(models, 1):
        marker = " ← 当前" if m == llm["model"] else ""
        print(f"  [{i}] {m}{marker}")

    try:
        choice = input("选择模型 (数字，回车返回): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not choice:
        return
    try:
        idx = int(choice) - 1
        new_model = models[idx]
    except (IndexError, ValueError):
        print("无效选择")
        return

    # Write LLM_MODEL to .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
        found = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("LLM_MODEL="):
                new_lines.append(f"LLM_MODEL={new_model}")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"LLM_MODEL={new_model}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    os.environ["LLM_MODEL"] = new_model
    print(f"LLM 模型已切换为: {new_model}")


def _configure_params(params: dict) -> dict:
    while True:
        from utils import get_llm_config
        llm = get_llm_config()
        print()
        _show_params(params)
        print(f"  LLM Provider : {llm['provider_label']} ({llm['provider_key'] or 'custom'})")
        print(f"  LLM Model    : {llm['model']}")
        print()
        print("  [1] 修改运行参数   [2] 切换 LLM 提供商   [3] 切换 LLM 模型   [0] 返回")
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "1":
            print()
            print("输入新值（直接回车保留当前值）：")
            for key, label, cast in [
                ("max_read", "每次精读上限", int),
                ("historical_ratio", "历史论文比例 (0-1)", float),
                ("top_n", "TOP N 合并数", int),
                ("auto_open", "自动打开浏览器 (1=是/0=否)", lambda x: bool(int(x))),
            ]:
                current = params[key]
                current_display = "是" if key == "auto_open" and current else str(current) if key != "auto_open" else "否"
                if key == "auto_open":
                    current_display = "是" if current else "否"
                else:
                    current_display = str(current)
                val = input(f"  {label} [{current_display}]: ").strip()
                if val:
                    try:
                        params[key] = cast(val)
                    except (ValueError, TypeError):
                        print(f"    输入无效，保持 {current_display}")
            print("参数已更新。")
        elif choice == "2":
            _switch_provider()
        elif choice == "3":
            _switch_model()
        elif choice == "0":
            break
        else:
            print(f"无效选项: {choice}")
    return params


# ── main ───────────────────────────────────────────────────────


def main() -> None:
    params = {
        "max_read": 10,
        "historical_ratio": 0.3,
        "top_n": 10,
        "auto_open": True,
    }

    while True:
        s = _gather_status()
        _show_dashboard(s)
        _show_menu()

        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 bye")
            break

        if choice == "1":
            _run_full_pipeline(params)
        elif choice == "2":
            _run_quick(params)
        elif choice == "3":
            _run_historical()
        elif choice == "4":
            _run_report_only(params)
        elif choice == "5":
            _open_report()
        elif choice == "6":
            params = _configure_params(params)
        elif choice == "0":
            print("再见 bye")
            break
        else:
            print(f"无效选项: {choice}")

        # Update params from keywords.yaml for next run
        try:
            from utils import load_keywords
            cfg = load_keywords()
            sc = cfg.get("search_config", {})
            params["historical_ratio"] = float(
                sc.get("historical_read_ratio", params["historical_ratio"])
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
