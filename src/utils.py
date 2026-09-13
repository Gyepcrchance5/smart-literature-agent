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
from typing import Any, Iterable
from urllib.parse import urlsplit

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output"
PAPERS_DIR = OUTPUT_DIR / "papers"
REPORTS_DIR = OUTPUT_DIR / "reports"
_FILE_LOGGING_UNAVAILABLE = False
_INITIAL_ENV = dict(os.environ)
_PROJECT_ENV: dict[str, str] = {}


DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_PLACEHOLDER_VALUES = {
    "your_api_key_here",
    "your_model_here",
    "your_provider_here",
}
_SECRET_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "DEEPXIV_TOKEN",
    "OPENAI_API_KEY",
}


class LLMConfigError(RuntimeError):
    """配置无法安全解析时抛出的错误。"""

    category = "configuration"


class LLMRequestError(RuntimeError):
    """已脱敏、可供 CLI/trace 使用的 LLM 请求错误。"""

    category = "provider"

    def __init__(self, info: dict[str, Any]):
        self.info = info
        super().__init__(info.get("message", "LLM 请求失败"))


def _clean_config_value(value: Any) -> str | None:
    """Normalize a config value and ignore empty/template placeholder values."""
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text.lower() in _PLACEHOLDER_VALUES:
        return None
    return text


def _parse_env_file(path: Path) -> dict[str, str]:
    """Read a dotenv-like file without exposing its contents."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def _load_project_env() -> None:
    """Load repo-local .env values without overriding the active shell."""
    env_path = PROJECT_ROOT / ".env"
    _PROJECT_ENV.update(_parse_env_file(env_path))
    for key, value in _PROJECT_ENV.items():
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


def _runtime_env_value(name: str) -> tuple[str | None, str | None]:
    """Return a value from process env or the repo .env, with its source.

    ``_load_project_env`` keeps compatibility with existing code by injecting
    project values into ``os.environ``.  The import-time snapshot and the
    parsed project mapping let us distinguish that injection from a value
    explicitly supplied by the caller.
    """
    current = _clean_config_value(os.environ.get(name))
    project_value = _clean_config_value(_PROJECT_ENV.get(name))
    if current is None:
        return (project_value, "project_env") if project_value else (None, None)
    if name not in _PROJECT_ENV:
        return current, "environment"
    if name in _INITIAL_ENV:
        return current, "environment"
    if current != project_value:
        return current, "environment"
    return current, "project_env"


def _setting_value(settings: dict[str, Any], name: str) -> str | None:
    return _clean_config_value(settings.get(name))


def _credential_for_source(
    source: str,
    cc: dict[str, Any],
    prefer_auth_token: bool = False,
) -> tuple[str | None, str | None]:
    """Pick one credential from exactly one source, never mixing sources."""
    if source == "claude_code":
        api_key = _setting_value(cc, "ANTHROPIC_API_KEY")
        auth_token = _setting_value(cc, "ANTHROPIC_AUTH_TOKEN")
    else:
        api_key, api_source = _runtime_env_value("ANTHROPIC_API_KEY")
        auth_token, auth_source = _runtime_env_value("ANTHROPIC_AUTH_TOKEN")
        api_key = api_key if api_source == source else None
        auth_token = auth_token if auth_source == source else None

    candidates = (
        ((auth_token, "auth_token"), (api_key, "api_key"))
        if prefer_auth_token
        else ((api_key, "api_key"), (auth_token, "auth_token"))
    )
    for value, kind in candidates:
        if value:
            return value, kind
    return None, None


def _resolve_credential(
    cc: dict[str, Any],
    *,
    provider_selected: bool,
    prefer_auth_token: bool,
) -> tuple[str | None, str | None, str | None]:
    """Resolve a credential and its source without cross-provider fallback."""
    # An explicitly selected provider is an explicit binding.  A Claude Code
    # credential may belong to a different endpoint, so it is intentionally
    # not used as a fallback in this branch.
    sources = ("environment", "project_env") if provider_selected else (
        "environment", "claude_code", "project_env"
    )
    for source in sources:
        value, kind = _credential_for_source(
            source, cc, prefer_auth_token=prefer_auth_token
        )
        if value:
            return value, source, kind
    return None, None, None


def _normalise_model(value: str | None) -> str | None:
    if not value:
        return None
    # Claude Code may append an internal context-window marker.
    return re.sub(r"\[.*\]$", "", value).strip() or None


def _safe_endpoint(base_url: Any) -> str:
    """Return a log-safe endpoint without query strings or userinfo."""
    value = _clean_config_value(base_url)
    if not value:
        return "<official>"
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return "<configured>"
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port else ""
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{host}{port}{path}"
    except ValueError:
        return "<configured>"


def redact_secrets(value: Any, secrets: Iterable[str] | None = None) -> str:
    """Redact known credentials and common bearer/API-key forms from text."""
    text = str(value)
    known: list[str] = []
    if secrets is not None:
        known.extend(str(item) for item in secrets if item)
    known.extend(
        str(_PROJECT_ENV[name])
        for name in _SECRET_ENV_NAMES
        if _PROJECT_ENV.get(name)
    )
    known.extend(
        str(os.environ[name])
        for name in _SECRET_ENV_NAMES
        if os.environ.get(name)
    )
    for secret in sorted(set(known), key=len, reverse=True):
        if secret:
            text = text.replace(secret, "<redacted>")
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1<redacted>",
        text,
    )
    text = re.sub(r"(?i)\b(?:sk-ant-|sk-)\w{8,}", "<redacted>", text)
    return text


def _error_status_code(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def classify_llm_error(
    exc: BaseException,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify an LLM failure into a safe, retry-aware error record.

    Authentication failures are deliberately marked non-retryable.  The
    runtime never attempts a different provider or credential after a 401.
    """
    if isinstance(exc, LLMRequestError):
        return exc.info

    cfg = config or {}
    if isinstance(exc, LLMConfigError):
        provider = cfg.get("provider_key") or cfg.get("provider_label") or "<unspecified>"
        message = redact_secrets(str(exc))
        return {
            "category": "configuration",
            "error_type": "configuration_error",
            "reason": "invalid_llm_config",
            "status_code": None,
            "retryable": False,
            "fallback_attempted": False,
            "provider": provider,
            "model": redact_secrets(cfg.get("model") or "<unspecified>"),
            "endpoint": _safe_endpoint(cfg.get("base_url")),
            "message": message,
        }

    status = _error_status_code(exc)
    name = type(exc).__name__.lower()
    if status == 401 or "authenticationerror" in name:
        category = "authentication"
        error_type = "authentication_error"
        retryable = False
        reason = "credentials_rejected"
    elif status == 403:
        category = "authorization"
        error_type = "authorization_error"
        retryable = False
        reason = "permission_denied"
    elif status == 429:
        category = "rate_limit"
        error_type = "rate_limit_error"
        retryable = True
        reason = "rate_limited"
    elif status is not None and status >= 500:
        category = "upstream"
        error_type = "upstream_error"
        retryable = True
        reason = "provider_server_error"
    elif isinstance(exc, (TimeoutError,)) or "timeout" in name:
        category = "timeout"
        error_type = "timeout_error"
        retryable = True
        reason = "request_timeout"
    else:
        category = "provider"
        error_type = "provider_error"
        retryable = False
        reason = "request_failed"

    config_secrets = [cfg.get("api_key"), cfg.get("auth_token")]
    raw_detail = redact_secrets(str(exc), secrets=config_secrets)[:500]
    provider = cfg.get("provider_key") or cfg.get("provider_label") or "<unspecified>"
    model = redact_secrets(cfg.get("model") or "<unspecified>")
    endpoint = _safe_endpoint(cfg.get("base_url"))
    if status == 401 or category == "authentication":
        message = (
            f"LLM 认证失败（HTTP 401）：provider={provider}，model={model}，"
            f"endpoint={endpoint}；当前凭证被服务拒绝，未尝试切换其他 provider 或 key。"
        )
        if raw_detail:
            message += f" 服务信息：{raw_detail}"
    else:
        status_text = f"HTTP {status}" if status is not None else "无 HTTP 状态码"
        message = (
            f"LLM 请求失败（{status_text}，category={category}）："
            f"provider={provider}，model={model}，endpoint={endpoint}"
        )
        if raw_detail:
            message += f"；{raw_detail}"

    return {
        "category": category,
        "error_type": error_type,
        "reason": reason,
        "status_code": status,
        "retryable": retryable,
        "fallback_attempted": False,
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
        "message": redact_secrets(message, secrets=config_secrets),
    }


def create_llm_message(
    client: Any,
    *,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Call ``messages.create`` and convert provider failures to safe errors."""
    try:
        return client.messages.create(**kwargs)
    except Exception as exc:
        safe_config = config or getattr(client, "_smart_lit_llm_config", None)
        if safe_config is None:
            try:
                safe_config = get_llm_config()
            except Exception:
                safe_config = None
        # Suppress the original exception chain: provider SDK exceptions can
        # include request headers or response bodies in an unhandled traceback.
        raise LLMRequestError(classify_llm_error(exc, safe_config)) from None


def get_anthropic_config() -> dict[str, str | None]:
    """Return the resolved API key and endpoint for legacy callers.

    This is a compatibility view over :func:`get_llm_config`; callers that
    need provider, model, auth mode, or source information should use the
    full resolver directly.
    """
    cfg = get_llm_config()
    return {
        "api_key": cfg.get("api_key") or None,
        "base_url": cfg.get("base_url") or None,
    }


def get_llm_config() -> dict:
    """Resolve LLM config from provider preset (keywords.yaml) + env vars.

    Resolution contract:
      * no provider: process env > Claude Code settings > project ``.env``;
      * selected provider: process/project provider config > provider preset;
        Claude Code credentials are not mixed into that provider.
      * process environment values always override project ``.env`` values.

    Returns:
        {
            "api_key": str,           # resolved API key
            "base_url": str,          # "" = Anthropic SDK default
            "model": str,             # e.g. "deepseek-v4-pro"
            "auth_mode": str,         # "" = SDK default x-api-key, "bearer"
            "provider_label": str,    # human-readable, e.g. "DeepSeek"
            "provider_key": str,      # provider name in config, e.g. "deepseek"
            "sources": dict,          # non-secret source for each resolved field
            "credential_kind": str,   # api_key / auth_token / ""
        }
    """
    cc = _read_claude_code_settings()
    provider_raw, provider_source = _runtime_env_value("LLM_PROVIDER")
    provider_key = provider_raw.lower() if provider_raw else ""
    keywords = load_keywords()
    providers = keywords.get("providers", {}) or {}

    if provider_key:
        if provider_key not in providers:
            available = ", ".join(sorted(str(key) for key in providers)) or "无"
            raise LLMConfigError(
                f"未知 LLM_PROVIDER={provider_key!r}；可选 provider：{available}。"
            )
        preset = providers[provider_key]
        preset_models = preset.get("models") or []
        preset_model = _normalise_model(preset_models[0] if preset_models else None)
        if not preset_model:
            raise LLMConfigError(f"provider {provider_key!r} 未配置默认模型。")

        raw_base_url, base_source = _runtime_env_value("ANTHROPIC_BASE_URL")
        base_url = raw_base_url or _clean_config_value(preset.get("base_url")) or ""
        if not raw_base_url:
            base_source = "provider_preset"

        raw_model, model_source = _runtime_env_value("LLM_MODEL")
        model = _normalise_model(raw_model) or preset_model
        if not raw_model:
            model_source = "provider_preset"

        auth_mode = str(preset.get("auth_mode") or "").strip().lower()
        api_key, key_source, credential_kind = _resolve_credential(
            cc,
            provider_selected=True,
            prefer_auth_token=auth_mode == "bearer",
        )
        return {
            "api_key": api_key or "",
            "base_url": base_url,
            "model": model,
            "auth_mode": auth_mode,
            "provider_label": preset.get("label", provider_key),
            "provider_key": provider_key,
            "credential_kind": credential_kind or "",
            "sources": {
                "provider": provider_source or "unknown",
                "api_key": key_source or "unset",
                "base_url": base_source or "unset",
                "model": model_source or "unset",
            },
        }

    api_key, key_source, credential_kind = _resolve_credential(
        cc,
        provider_selected=False,
        prefer_auth_token=False,
    )

    raw_base_url, base_source = _runtime_env_value("ANTHROPIC_BASE_URL")
    cc_base_url = _setting_value(cc, "ANTHROPIC_BASE_URL")
    project_base_url, project_base_source = _runtime_env_value("ANTHROPIC_BASE_URL")
    if base_source == "environment" and raw_base_url:
        base_url = raw_base_url
    elif cc_base_url:
        base_url = cc_base_url
        base_source = "claude_code"
    elif project_base_url:
        base_url = project_base_url
        base_source = project_base_source or "project_env"
    else:
        base_url = ""
        base_source = "default"

    raw_model, model_source = _runtime_env_value("LLM_MODEL")
    cc_model = _normalise_model(
        _setting_value(cc, "ANTHROPIC_MODEL") or _setting_value(cc, "LLM_MODEL")
    )
    project_model = _normalise_model(raw_model) if model_source == "project_env" else None
    if model_source == "environment" and raw_model:
        model = _normalise_model(raw_model) or DEFAULT_ANTHROPIC_MODEL
    elif cc_model:
        model = cc_model
        model_source = "claude_code"
    elif project_model:
        model = project_model
        model_source = "project_env"
    else:
        model = DEFAULT_ANTHROPIC_MODEL
        model_source = "default"

    return {
        "api_key": api_key or "",
        "base_url": base_url,
        "model": model,
        "auth_mode": "bearer" if credential_kind == "auth_token" else "",
        "provider_label": "Claude Code 同源" if key_source == "claude_code" else "Anthropic 兼容端点",
        "provider_key": "",
        "credential_kind": credential_kind or "",
        "sources": {
            "provider": "unset",
            "api_key": key_source or "unset",
            "base_url": base_source,
            "model": model_source,
        },
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
