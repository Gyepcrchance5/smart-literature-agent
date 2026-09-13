"""研究会话上下文：把一次具体的科研瓶颈传给检索、总结和迁移分析。

上下文是可选的。没有配置时，旧的文献跟踪流程仍然可以正常运行；配置后，
本地质量审查会结合研究问题、约束和目标指标判断，默认不把上下文发送给外部 LLM。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from utils import CONFIG_DIR


DEFAULT_CONTEXT_PATH = CONFIG_DIR / "research_context.yaml"
_LIST_FIELDS = {
    "symptoms",
    "constraints",
    "target_metrics",
    "known_attempts",
    "target_fields",
    "non_goals",
}


def load_research_context(path: str | Path | None = None) -> dict[str, Any]:
    """读取研究上下文；文件不存在时返回空上下文，不阻断旧流程。"""
    context_path = Path(path) if path else DEFAULT_CONTEXT_PATH
    if not context_path.exists():
        return {}
    try:
        raw = yaml.safe_load(context_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取研究上下文 {context_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"研究上下文必须是 YAML object: {context_path}")

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _LIST_FIELDS and value is None:
            normalized[key] = []
        elif key in _LIST_FIELDS and isinstance(value, str):
            normalized[key] = [value]
        else:
            normalized[key] = value
    return normalized


def has_research_context(context: dict[str, Any] | None) -> bool:
    """判断上下文是否包含足够信息，避免把空模板注入 prompt。"""
    if not context:
        return False
    meaningful = ("problem", "bottleneck", "baseline", "constraints", "target_metrics")
    return any(context.get(key) for key in meaningful)


def format_research_context(context: dict[str, Any] | None) -> str:
    """将研究上下文格式化为稳定、可审计的 prompt 片段。"""
    if not has_research_context(context):
        return "（未提供具体研究上下文；只做论文自身分析，不推断对用户项目的适配性。）"

    labels = {
        "project": "研究项目",
        "problem": "研究问题",
        "bottleneck": "当前瓶颈",
        "symptoms": "观察到的现象",
        "baseline": "当前基线",
        "constraints": "约束条件",
        "target_metrics": "目标指标",
        "known_attempts": "已经尝试过的方法",
        "target_fields": "关注方向",
        "non_goals": "明确不考虑的方向",
    }
    lines: list[str] = []
    for key, label in labels.items():
        value = context.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = "；".join(str(item) for item in value)
        lines.append(f"- {label}: {value}")
    return "\n".join(lines) or "（未提供具体研究上下文。）"
