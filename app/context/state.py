"""上下文状态存根模块。

   原 context_state 持久化机制已删除，此文件仅保留最小接口满足 runtime.py 的 import。
   所有函数返回空值或 None。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ContextStateData:
    """存根：所有字段返回默认值。"""
    session_id: str = ""
    source_message_count: int = 0
    source_history_fingerprint: str = ""
    compacted_messages: list = field(default_factory=list)
    active_context_summary: str = ""
    active_context_snapshot: dict[str, list[str]] = field(default_factory=dict)
    older_history_summary: str = ""
    resolved_user_preferences: list[str] = field(default_factory=list)
    resolved_project_constraints: list[str] = field(default_factory=list)
    recent_risks: list[str] = field(default_factory=list)
    compaction_level: int = 0
    auto_compact_failure_count: int = 0
    auto_compact_suppressed_until: float = 0.0
    last_microcompact_at: float = 0.0
    compaction_history: list[dict[str, Any]] = field(default_factory=list)
    last_token_stats: dict[str, Any] = field(default_factory=dict)


def build_history_fingerprint(history: list) -> str:
    return ""


def build_token_stats_snapshot(**kwargs) -> dict:
    return {}


def load_context_state(workspace: str, session_id: str) -> ContextStateData | None:
    return None


def save_context_state(workspace: str, state: ContextStateData) -> Path:
    return Path(".")


def merge_context_state_snapshot(workspace: str, session_id: str, overlay_snapshot: dict) -> Path | None:
    return None
