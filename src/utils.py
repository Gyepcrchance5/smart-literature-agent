"""通用工具模块：日志、配置加载、去重状态管理、DeepXiv CLI 调用封装。"""
from __future__ import annotations

import json
import logging
import os
import re
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
PAPERS_DIR = OUTPUT_DIR / "papers"
REPORTS_DIR = OUTPUT_DIR / "reports"
_FILE_LOGGING_UNAVAILABLE = False


def _load_project_env() -> None:
    """Load repo-local .env values without overriding the active shell."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_project_env()

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
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 日志目录可能在只读容器、文件锁定或受限权限下不可写；这不应阻断
    # search/read/summarize 主流程，至少保留终端日志。一次失败后不再让每个
    # 子模块重复尝试同一个不可写文件，避免污染 Agent 的可读 trace/终端输出。
    global _FILE_LOGGING_UNAVAILABLE
    if not _FILE_LOGGING_UNAVAILABLE:
        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError as exc:
            _FILE_LOGGING_UNAVAILABLE = True
            logger.warning("日志文件不可写，改用终端输出：%s", exc)
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


PAPERS_DATA_DIR = DATA_DIR / "papers"  # 内部精读产物（reader 输出，summarizer 读取）


def ensure_dirs() -> None:
    """确保所有输出目录存在。"""
    for d in [DATA_DIR, LOGS_DIR, OUTPUT_DIR, PAPERS_DIR, REPORTS_DIR, PAPERS_DATA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def try_write_text(path: str | Path, content: str, logger: logging.Logger | None = None) -> Path | None:
    """尽力写入非关键产物；权限/文件锁异常只记录告警，不击穿主流程。"""
    out = Path(path)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        return out
    except OSError as exc:
        target_logger = logger or logging.getLogger("smart-literature-agent")
        target_logger.warning("产物写入失败（继续运行）：%s — %s", out, exc)
        return None


def _read_claude_code_settings() -> dict[str, str]:
    """读取 Claude Code 持久化配置（~/.claude/settings.json 的 env 段）。

    优先级最高：当用户通过 Claude Code 对话时，项目应自动复用同一套凭证，
    无需手动在 .env 中维护 API key。
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    if not settings_path.exists():
        return {}
    try:
        cfg = json.loads(settings_path.read_text(encoding="utf-8"))
        return cfg.get("env", {})
    except (OSError, json.JSONDecodeError):
        return {}


def get_anthropic_config() -> dict[str, str | None]:
    """返回 {"api_key", "base_url"}。

    优先级：
      1. 显式环境变量 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL
      2. Claude Code 配置（ANTHROPIC_AUTH_TOKEN 也视为 api_key）
      3. .env 文件（由 _load_project_env 已注入 os.environ）
    """
    # Claude Code 配置优先（.env 可能有已过期的 key）
    cc = _read_claude_code_settings()
    api_key = cc.get("ANTHROPIC_API_KEY") or cc.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = cc.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")

    return {"api_key": api_key, "base_url": base_url}


def get_llm_config() -> dict:
    """Resolve LLM config from provider preset (keywords.yaml) + env vars.

    优先级：显式环境变量 > provider preset > Claude Code 配置 > 默认值

    Returns:
        {
            "api_key": str,           # resolved API key
            "base_url": str,          # "" = Anthropic SDK default
            "model": str,             # e.g. "deepseek-v4-pro"
            "auth_mode": str,         # "" = SDK default x-api-key, "bearer"
            "provider_label": str,    # human-readable, e.g. "DeepSeek"
            "provider_key": str,      # provider name in config, e.g. "deepseek"
        }
    """
    # Claude Code 配置作为底层 fallback
    cc = _read_claude_code_settings()

    provider_key = os.environ.get("LLM_PROVIDER", "")
    providers = load_keywords().get("providers", {}) if provider_key else {}

    if provider_key in providers:
        p = providers[provider_key]
        # Claude Code 的 AUTH_TOKEN 作为 API key fallback（.env 的 key 可能已过期）
        cc_key = cc.get("ANTHROPIC_AUTH_TOKEN") or cc.get("ANTHROPIC_API_KEY", "")
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
        api_key = env_key or cc_key
        base_url = p.get("base_url", "")
        model = os.environ.get("LLM_MODEL") or p.get("models", [""])[0]
        auth_mode = p.get("auth_mode", "")
        return {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "auth_mode": auth_mode,
            "provider_label": p.get("label", provider_key),
            "provider_key": provider_key,
        }

    # 无 LLM_PROVIDER：直接用 Claude Code 配置
    cfg = get_anthropic_config()
    raw_model = cc.get("ANTHROPIC_MODEL") or os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
    # 去掉 Claude Code 内部的上下文窗口标记（如 mimo-v2.5-pro[1m] → mimo-v2.5-pro）
    model = re.sub(r"\[.*\]$", "", raw_model) if raw_model else raw_model
    return {
        "api_key": cfg["api_key"] or "",
        "base_url": cfg.get("base_url") or "",
        "model": model,
        "auth_mode": "",
        "provider_label": "Claude Code 同源",
        "provider_key": "",
    }


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
