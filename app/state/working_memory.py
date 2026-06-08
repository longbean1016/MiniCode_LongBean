from __future__ import annotations

import time
from dataclasses import dataclass, field

"""工作记忆模块，维护当前任务的短期上下文和受保护条目。"""


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
    # user_intent 已经会通过当前用户消息和检索 query 参与主链路，
    # 这里不再把它提升进 active_tasks，避免它和长期记忆/压缩基线重复抢位。
    "active_tasks": ProtectedSlotSpec(("active_task",), 2),
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
_TOPIC_SHIFT_KEEP_ENTRY_TYPES = {"user_preference", "project_constraint", "user_intent"}
_TOPIC_SHIFT_TRANSIENT_ENTRY_TYPES = {
    "active_task",
    "key_decision",
    "recent_risk",
    "error_context",
    "tool_finding",
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
        # 工作记忆只服务“当前会话中的持续约束和近期事实”，不是长期记忆库。
        # 所以这里入库前先做轻量归一化，并在写入后立刻执行限额，防止它本身反过来挤爆 prompt。
        text = _normalize_text(content)
        normalized_type = _normalize_text(entry_type) or "active_task"
        if not text:
            return None

        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds

        # 某些类型天然只应该保留最新版本，例如当前任务、最新用户意图。
        # 这种场景用 replace_latest_of_type，避免旧版本继续污染后续选择。
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

    def clear_transient_entries_on_topic_shift(self, next_user_intent: str) -> int:
        """
        新用户意图和上一轮几乎无交集时，清理旧话题的短期工作记忆。

        用户偏好和项目约束是跨话题稳定信息，不能因为当前任务切换就删除；
        active_task / key_decision / risk / tool_finding 则默认只服务旧话题。
        """
        previous_intent = self.get_primary_user_intent()
        if not previous_intent:
            return 0
        if not _is_topic_shift(previous_intent, next_user_intent):
            return 0

        before = len(self.get_entries())
        self.entries = [
            entry for entry in self.get_entries()
            if (
                entry.entry_type in _TOPIC_SHIFT_KEEP_ENTRY_TYPES
                or entry.entry_type not in _TOPIC_SHIFT_TRANSIENT_ENTRY_TYPES
            )
        ]
        return before - len(self.entries)

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

            # 同一个槽位内部按“重要度优先，其次时间更近优先”排序。
            # 这样 prompt 头部更容易先看到当前最该遵守的约束，而不是最早写入的历史条目。
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
            # snapshot 只保存每个槽位最值得带进 prompt 的少量事实，
            # 不把 WorkingMemory 原样平铺，避免结构化记忆再次膨胀成原始日志。
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

        # 未列入固定槽位的条目放到补充区，避免因为没建专门槽位就完全失声。
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
        # 顺序不能反：
        # 1. 先按类型限额，防止单一类别霸占全部预算
        # 2. 再按总 token / 总条数做全局淘汰
        # 这样“用户偏好被大量 tool_finding 挤掉”的概率会低很多。
        self._enforce_type_limits()
        self._enforce_token_budget()
        while len(self.entries) > self.max_entries:
            self.entries.pop(self._pick_lowest_priority_index())

    def _enforce_type_limits(self) -> None:
        # 每种 entry_type 都有独立上限，防止某一种运行时噪声无限堆积。
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
        # token 预算是真正影响 prompt 注入成本的硬约束，
        # 即使总条数不多，只要单条太长，也必须继续淘汰。
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


def _is_topic_shift(previous_intent: str, next_intent: str) -> bool:
    """用中文友好的 2-4 字 ngram 粗判话题切换，避免旧任务 WM 残留。"""
    previous_tokens = _topic_ngrams(previous_intent)
    next_tokens = _topic_ngrams(next_intent)
    if not previous_tokens or not next_tokens:
        return False
    return len(previous_tokens & next_tokens) <= 1


def _topic_ngrams(text: str) -> set[str]:
    """提取 2-4 字连续片段；对中英文都先移除空白和常见标点。"""
    normalized = "".join(
        char.lower()
        for char in str(text)
        if not char.isspace() and char not in "，。！？；：,.!?;:()（）[]【】{}<>《》\"'"
    )
    if len(normalized) < 2:
        return set()
    tokens: set[str] = set()
    for size in (2, 3, 4):
        if len(normalized) < size:
            continue
        for index in range(0, len(normalized) - size + 1):
            tokens.add(normalized[index:index + size])
    return tokens
