from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.context_manager import ContextStats
from app.types import ChatMessage

CONTEXT_STATE_DIR_NAME = ".context_state"


@dataclass(slots=True)
class ContextStateData:
    """保存一份可恢复的活动上下文状态，不覆盖原始 session 历史。"""

    session_id: str
    source_message_count: int
    source_history_fingerprint: str
    compacted_messages: list[ChatMessage] = field(default_factory=list)
    older_history_summary: str = ""
    compaction_level: int = 0
    compaction_history: list[dict[str, Any]] = field(default_factory=list)
    last_token_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source_message_count": self.source_message_count,
            "source_history_fingerprint": self.source_history_fingerprint,
            "compacted_messages": list(self.compacted_messages),
            "older_history_summary": self.older_history_summary,
            "compaction_level": self.compaction_level,
            "compaction_history": list(self.compaction_history),
            "last_token_stats": dict(self.last_token_stats),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextStateData":
        raw_messages = data.get("compacted_messages", [])
        compacted_messages = [item for item in raw_messages if isinstance(item, dict)]
        raw_history = data.get("compaction_history", [])
        compaction_history = [item for item in raw_history if isinstance(item, dict)]
        raw_stats = data.get("last_token_stats", {})
        last_token_stats = dict(raw_stats) if isinstance(raw_stats, dict) else {}
        return cls(
            session_id=str(data["session_id"]),
            source_message_count=int(data.get("source_message_count", 0)),
            source_history_fingerprint=str(data.get("source_history_fingerprint", "")),
            compacted_messages=compacted_messages,  # type: ignore[arg-type]
            older_history_summary=str(data.get("older_history_summary", "")),
            compaction_level=int(data.get("compaction_level", 0)),
            compaction_history=compaction_history,
            last_token_stats=last_token_stats,
        )


def get_context_state_dir(workspace: str) -> Path:
    """返回活动上下文状态目录。"""
    return Path(workspace).resolve() / CONTEXT_STATE_DIR_NAME


def get_context_state_file_path(workspace: str, session_id: str) -> Path:
    """返回某个会话对应的活动上下文状态文件。"""
    return get_context_state_dir(workspace) / f"{session_id}.json"


def save_context_state(workspace: str, state: ContextStateData) -> Path:
    """把 compact 后的活动上下文安全写入本地状态文件。"""
    state_dir = get_context_state_dir(workspace)
    state_dir.mkdir(parents=True, exist_ok=True)

    state_file = get_context_state_file_path(workspace, state.session_id)
    temp_file = state_file.with_suffix(".json.tmp")
    temp_file.write_text(
        json.dumps(state.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_file.replace(state_file)
    return state_file


def load_context_state(workspace: str, session_id: str) -> ContextStateData | None:
    """从本地恢复某个会话的活动上下文状态。"""
    state_file = get_context_state_file_path(workspace, session_id)
    if not state_file.exists():
        return None

    try:
        raw_text = state_file.read_text(encoding="utf-8")
        raw_data = json.loads(raw_text)
        if not isinstance(raw_data, dict):
            return None
        return ContextStateData.from_dict(raw_data)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def build_history_fingerprint(messages: list[ChatMessage]) -> str:
    """用稳定字段生成历史指纹，判断活动上下文是否还能复用。"""
    digest = hashlib.md5(usedforsecurity=False)
    for message in messages:
        digest.update(str(message.get("role", "")).encode("utf-8", errors="ignore"))
        digest.update(str(message.get("tool_name", "")).encode("utf-8", errors="ignore"))
        digest.update(str(message.get("tool_use_id", "")).encode("utf-8", errors="ignore"))
        digest.update(str(message.get("content", "")).encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def build_token_stats_snapshot(*, preview_stats: ContextStats, final_stats: ContextStats) -> dict[str, Any]:
    """抽取一份轻量 token 快照，方便恢复和排查。"""
    return {
        "preview_total": preview_stats.total_tokens,
        "preview_usage": preview_stats.usage_ratio,
        "preview_budget": preview_stats.usable_budget,
        "final_total": final_stats.total_tokens,
        "final_usage": final_stats.usage_ratio,
        "final_budget": final_stats.usable_budget,
    }
