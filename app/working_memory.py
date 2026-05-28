from __future__ import annotations

import time
from dataclasses import dataclass, field


def _normalize_text(text: str) -> str:
    """规范化文本空白，便于作为运行时工作记忆存储。"""
    return " ".join(str(text).strip().split())


@dataclass(slots=True)
class WorkingMemoryEntry:
    """仅存在于运行时的受保护上下文条目。"""

    content: str
    entry_type: str
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    importance: float = 1.0

    def is_expired(self, now: float | None = None) -> bool:
        """判断当前条目是否已经过期。"""
        if self.expires_at is None:
            return False
        current = time.time() if now is None else now
        return current > self.expires_at


@dataclass(slots=True)
class WorkingMemory:
    """类似 minicode 的运行时工作记忆跟踪器。"""

    max_entries: int = 15
    entries: list[WorkingMemoryEntry] = field(default_factory=list)

    def protect(
        self,
        content: str,
        *,
        entry_type: str = "active_task",
        ttl_seconds: float | None = None,
        importance: float = 1.0,
        replace_latest_of_type: bool = False,
    ) -> WorkingMemoryEntry | None:
        """
        添加一条运行时受保护上下文。

        默认 `entry_type` 是 `active_task`：
        - 调用方不传 `entry_type` 时，按 `active_task` 处理
        - 传入空字符串时，也会回退到 `active_task`
        """
        text = _normalize_text(content)
        normalized_type = _normalize_text(entry_type) or "active_task"
        if not text:
            return None

        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds

        if replace_latest_of_type:
            self.entries = [
                entry for entry in self.get_entries()
                if entry.entry_type != normalized_type
            ]

        entry = WorkingMemoryEntry(
            content=text,
            entry_type=normalized_type,
            expires_at=expires_at,
            importance=float(importance),
        )
        self.entries.append(entry)
        self._enforce_entry_limits()
        return entry

    def clear_failures(self) -> None:
        """清理当前保存的错误上下文条目。"""
        self.entries = [
            entry for entry in self.get_entries()
            if entry.entry_type != "error_context"
        ]

    def clear_expired(self) -> int:
        """删除已过期的运行时条目，并返回删除数量。"""
        before = len(self.entries)
        now = time.time()
        self.entries = [
            entry for entry in self.entries
            if not entry.is_expired(now)
        ]
        return before - len(self.entries)

    def get_entries(self) -> list[WorkingMemoryEntry]:
        """按插入顺序返回所有未过期条目。"""
        self.clear_expired()
        return list(self.entries)

    def get_primary_user_intent(self) -> str:
        """返回最新一条用户意图，用于记忆检索和 prompt 聚焦。"""
        for entry in reversed(self.get_entries()):
            if entry.entry_type == "user_intent":
                return entry.content
        return ""

    def get_entries_by_type(self, entry_type: str) -> list[WorkingMemoryEntry]:
        """按 entry_type 过滤并返回对应条目。"""
        normalized_type = _normalize_text(entry_type)
        return [
            entry for entry in self.get_entries()
            if entry.entry_type == normalized_type
        ]

    def format_for_prompt(self) -> str:
        """把运行时保护上下文格式化成可注入 prompt 的文本。"""
        sections: list[str] = []
        grouped = {
            "user_intent": self.get_entries_by_type("user_intent"),
            "active_task": self.get_entries_by_type("active_task"),
            "key_decision": self.get_entries_by_type("key_decision"),
            "error_context": self.get_entries_by_type("error_context"),
        }

        if grouped["user_intent"]:
            sections.append("用户意图：")
            for entry in grouped["user_intent"][-3:]:
                sections.append(f"- {entry.content}")

        if grouped["active_task"]:
            sections.append("活跃任务：")
            for entry in grouped["active_task"][-5:]:
                sections.append(f"- {entry.content}")

        if grouped["key_decision"]:
            sections.append("关键决策：")
            for entry in grouped["key_decision"][-5:]:
                sections.append(f"- {entry.content}")

        if grouped["error_context"]:
            sections.append("错误上下文：")
            for entry in grouped["error_context"][-5:]:
                sections.append(f"- {entry.content}")

        supplemental_entries = [
            entry for entry in self.get_entries()
            if entry.entry_type not in grouped
        ]
        if supplemental_entries:
            sections.append("运行时保护上下文：")
            for entry in supplemental_entries[-5:]:
                sections.append(f"- [{entry.entry_type}] {entry.content}")

        return "\n".join(sections).strip()

    def _enforce_entry_limits(self) -> None:
        """超过最大条目数时，优先删除重要度最低、最旧的条目。"""
        self.clear_expired()
        while len(self.entries) > self.max_entries:
            lowest = min(
                range(len(self.entries)),
                key=lambda index: (
                    self.entries[index].importance,
                    self.entries[index].created_at,
                ),
            )
            self.entries.pop(lowest)
