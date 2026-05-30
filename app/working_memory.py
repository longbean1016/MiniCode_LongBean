from __future__ import annotations

import time
from dataclasses import dataclass, field


def _estimate_tokens(text: str) -> int:
    """轻量估算 token，用于约束受保护记忆自身的体积。"""
    return max(1, len(str(text)) // 4)


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

    def token_count(self) -> int:
        return _estimate_tokens(self.content)

    def is_expired(self, now: float | None = None) -> bool:
        """判断当前条目是否已经过期。"""
        if self.expires_at is None:
            return False

        current = time.time() if now is None else now
        return current > self.expires_at


@dataclass(frozen=True, slots=True)
class ProtectedSlotSpec:
    """定义受保护工作记忆槽位和上限。"""

    entry_types: tuple[str, ...]
    max_entries: int


_PROTECTED_SLOT_SPECS: dict[str, ProtectedSlotSpec] = {
    "preferences": ProtectedSlotSpec(("user_preference",), 2),
    "stable_constraints": ProtectedSlotSpec(("project_constraint",), 3),
    "active_tasks": ProtectedSlotSpec(("active_task", "user_intent"), 3),
    "decisions": ProtectedSlotSpec(("key_decision",), 4),
    "open_issues": ProtectedSlotSpec(("recent_risk", "error_context"), 4),
    "tool_findings": ProtectedSlotSpec(("reflection_file", "tool_finding"), 2),
}
_ENTRY_TYPE_LIMITS: dict[str, int] = {
    "user_preference": 2,
    "project_constraint": 3,
    "active_task": 2,
    "user_intent": 1,
    "key_decision": 4,
    "recent_risk": 3,
    "error_context": 2,
    "reflection_file": 2,
    "tool_finding": 2,
}
_PROMPT_SECTION_TITLES = {
    "preferences": "用户偏好",
    "stable_constraints": "项目约束",
    "active_tasks": "当前任务",
    "decisions": "关键决策",
    "open_issues": "未解决问题",
    "tool_findings": "关键工具发现",
}


@dataclass(slots=True)
class WorkingMemory:
    """类似 minicode 的运行时工作记忆跟踪器。"""

    max_entries: int = 15
    max_tokens: int = 420
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

    def clear_entries_by_type(self, *entry_types: str) -> None:
        """
        按类型批量清理运行时条目。

        这个方法主要给反思链路使用：
        - 每轮开始前清空上一轮的 `reflection_*` 条目
        - 避免上一轮采集到的决策、失败、文件触点污染本轮反思输入
        """
        normalized_types = {
            _normalize_text(entry_type)
            for entry_type in entry_types
            if _normalize_text(entry_type)
        }
        if not normalized_types:
            return

        self.entries = [
            entry for entry in self.get_entries()
            if entry.entry_type not in normalized_types
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
        """按 `entry_type` 过滤并返回对应条目。"""
        normalized_type = _normalize_text(entry_type)
        return [
            entry for entry in self.get_entries()
            if entry.entry_type == normalized_type
        ]

    def get_protected_tokens(self) -> int:
        """返回当前受保护工作记忆的大致 token 总量。"""
        return sum(entry.token_count() for entry in self.get_entries())

    def build_protected_snapshot(self) -> dict[str, list[str]]:
        """把受保护工作记忆整理成结构化槽位，供 compact/render 统一使用。"""
        snapshot: dict[str, list[str]] = {}
        for slot_name, spec in _PROTECTED_SLOT_SPECS.items():
            slot_entries: list[WorkingMemoryEntry] = []
            for entry_type in spec.entry_types:
                slot_entries.extend(self.get_entries_by_type(entry_type))
            if not slot_entries:
                continue

            slot_entries.sort(
                key=lambda entry: (entry.importance, entry.created_at),
                reverse=True,
            )
            lines: list[str] = []
            seen: set[str] = set()
            for entry in slot_entries:
                normalized = " ".join(entry.content.strip().split())
                dedupe_key = normalized.lower()
                if not dedupe_key or dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                lines.append(normalized)
                if len(lines) >= spec.max_entries:
                    break
            if lines:
                snapshot[slot_name] = lines
        return snapshot

    def format_for_prompt(self) -> str:
        """把运行时保护上下文化成可注入 prompt 的文本。"""
        sections: list[str] = []
        snapshot = self.build_protected_snapshot()
        for slot_name in (
            "preferences",
            "stable_constraints",
            "active_tasks",
            "decisions",
            "open_issues",
            "tool_findings",
        ):
            lines = snapshot.get(slot_name, [])
            if not lines:
                continue
            sections.append(f"{_PROMPT_SECTION_TITLES[slot_name]}：")
            sections.extend(f"- {line}" for line in lines)

        supplemental_entries = [
            entry for entry in self.get_entries()
            if entry.entry_type not in _ENTRY_TYPE_LIMITS
            and not entry.entry_type.startswith("reflection_")
        ]
        if supplemental_entries:
            sections.append("运行时补充上下文：")
            for entry in supplemental_entries[-4:]:
                sections.append(f"- [{entry.entry_type}] {entry.content}")
        return "\n".join(sections).strip()

    def _enforce_entry_limits(self) -> None:
        """超过类型/总量上限时，优先删除重要度最低、最旧的条目。"""
        self.clear_expired()
        self._enforce_type_limits()
        self._enforce_token_budget()
        while len(self.entries) > self.max_entries:
            self.entries.pop(self._pick_lowest_priority_index())

    def _enforce_type_limits(self) -> None:
        for entry_type, max_allowed in _ENTRY_TYPE_LIMITS.items():
            matching_indexes = [
                index
                for index, entry in enumerate(self.entries)
                if entry.entry_type == entry_type
            ]
            while len(matching_indexes) > max_allowed:
                lowest_index = min(
                    matching_indexes,
                    key=lambda index: (
                        self.entries[index].importance,
                        self.entries[index].created_at,
                    ),
                )
                self.entries.pop(lowest_index)
                matching_indexes = [
                    index
                    for index, entry in enumerate(self.entries)
                    if entry.entry_type == entry_type
                ]

    def _enforce_token_budget(self) -> None:
        while self.get_protected_tokens() > self.max_tokens and self.entries:
            self.entries.pop(self._pick_lowest_priority_index())

    def _pick_lowest_priority_index(self) -> int:
        return min(
            range(len(self.entries)),
            key=lambda index: (
                self.entries[index].importance,
                self.entries[index].created_at,
            ),
        )
