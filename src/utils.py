"""通用工具模块：日志、配置加载、去重状态管理、DeepXiv CLI 调用封装。"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Windows 下 Python 默认 stdout 用 GBK，输出 emoji 会 UnicodeEncodeError。
# DeepXiv CLI 里用了 emoji，所以子进程必须强制 UTF-8。
_UTF8_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def _resolve_deepxiv_cli() -> str:
    """找同 conda env / venv 下的 deepxiv 可执行文件。

    直接 `python src/xxx.py` 运行时，conda env 没激活，PATH 里没有 deepxiv.exe。
    用 sys.executable 所在目录的 Scripts/bin 子目录来定位。
    """
    scripts_dir = Path(sys.executable).parent / ("Scripts" if os.name == "nt" else "bin")
    exe_name = "deepxiv.exe" if os.name == "nt" else "deepxiv"
    candidate = scripts_dir / exe_name
    return str(candidate) if candidate.exists() else "deepxiv"


DEEPXIV_CLI = _resolve_deepxiv_cli()


def get_logger(name: str = "smart-literature-agent") -> logging.Logger:
    """返回一个统一格式的 logger，同时写文件与标准输出。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{datetime.now():%Y%m%d}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_keywords() -> dict[str, Any]:
    """加载 config/keywords.yaml。"""
    path = CONFIG_DIR / "keywords.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_seen_ids() -> set[str]:
    """加载已处理过的论文 ID 集合，用于去重。"""
    path = DATA_DIR / "seen_ids.json"
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return set(json.load(f))


def save_seen_ids(ids: set[str]) -> None:
    """保存已处理过的论文 ID 集合。"""
    path = DATA_DIR / "seen_ids.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, ensure_ascii=False, indent=2)


def ensure_dirs() -> None:
    """确保所有输出目录存在。"""
    for d in [DATA_DIR, LOGS_DIR, OUTPUT_DIR, OUTPUT_DIR / "papers", OUTPUT_DIR / "reports"]:
        d.mkdir(parents=True, exist_ok=True)


def get_anthropic_config() -> dict[str, str | None]:
    """返回 {"api_key", "base_url"}：优先从环境变量读，fallback 到 ~/.claude/settings.json。

    fallback 的动机：某些配置工具（例如 cc-switch）会把 key 注入到 Claude Code 进程的环境
    变量，但这些变量只在那个进程的子进程可见。如果从别的 shell（PowerShell / 独立终端）
    运行本脚本，env 里就没有。此时去读 Claude Code 的持久配置兜底，避免重复配置。
    若你直连 Anthropic 官方 API 并已设好 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL 环境变量，
    此 fallback 不会触发。
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    if not (api_key and base_url):
        settings_path = Path.home() / ".claude" / "settings.json"
        if settings_path.exists():
            try:
                cfg = json.loads(settings_path.read_text(encoding="utf-8"))
                env_section = cfg.get("env", {})
                api_key = api_key or env_section.get("ANTHROPIC_API_KEY")
                base_url = base_url or env_section.get("ANTHROPIC_BASE_URL")
            except (OSError, json.JSONDecodeError):
                pass
    return {"api_key": api_key, "base_url": base_url}


def run_deepxiv(args: list[str], parse_json: bool = False, timeout: int = 60) -> str | dict | list:
    """调用 deepxiv CLI 子进程，强制 UTF-8，避开 Windows GBK 陷阱。

    Args:
        args: 传给 deepxiv 的参数列表，例如 ["search", "knowledge distillation", "--limit", "3"]
        parse_json: True 时自动加 `--format json` 并解析为 dict/list
        timeout: 子进程超时（秒）

    Raises:
        RuntimeError: 子进程退出码非 0
        json.JSONDecodeError: parse_json=True 但输出不是合法 JSON
    """
    cmd = [DEEPXIV_CLI, *args]
    if parse_json and "--format" not in args and "-f" not in args:
        cmd += ["--format", "json"]
    result = subprocess.run(
        cmd,
        env=_UTF8_ENV,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"deepxiv {' '.join(args)} failed (exit {result.returncode}):\n{result.stderr}"
        )
    return json.loads(result.stdout) if parse_json else result.stdout


if __name__ == "__main__":
    # 冒烟测试：验证所有工具函数可用
    ensure_dirs()
    log = get_logger()
    cfg = load_keywords()
    seen = load_seen_ids()
    log.info(
        "utils 自检通过：领域数=%d，已见 ID 数=%d，搜索配置=%s",
        len(cfg["fields"]),
        len(seen),
        cfg["search_config"],
    )
