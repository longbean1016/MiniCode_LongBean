from __future__ import annotations

from dataclasses import dataclass

from app.context_manager import estimate_tokens
from app.working_memory import WorkingMemory, WorkingMemoryEntry

COMPACT_MEMORY_MAX_TOKENS = 220
COMPACT_MEMORY_SECTION_ENTRY_LIMIT = 3


@dataclass(frozen=True, slots=True)
class _SectionSpec:
    """定义 compact memory 各个层的优先级和预算。"""

    title: str
    entry_types: tuple[str, ...]
    max_entries: int
    max_item_chars: int
    max_section_tokens: int


_SECTION_SPECS = (
    _SectionSpec("用户偏好", ("user_preference",), 2, 90, 48),
    _SectionSpec("项目约束", ("project_constraint",), 3, 100, 54),
    _SectionSpec("最近风险", ("recent_risk", "error_context"), 3, 100, 54),
    _SectionSpec("当前任务", ("active_task",), 2, 80, 34),
    _SectionSpec("关键决策", ("key_decision",), 2, 80, 34),
    _SectionSpec("用户意图", ("user_intent",), 1, 90, 22),
)


def build_compact_memory_context(
    *,
    older_history_summary: str,
    working_memory: WorkingMemory,
    previous_compact_memory_context: str = "",
    resolved_user_preferences: list[str] | None = None,
    resolved_project_constraints: list[str] | None = None,
    recent_risks: list[str] | None = None,
) -> str:
    """
    构造一段专门给压缩阶段使用的短摘要基线。

    参考 minicode 的思路：
    - 不直接把 working memory 整段原样塞回去
    - 先抽稳定信号，再做分层预算组装
    - 高价值层优先：偏好 / 约束 / 风险 > 当前任务 / 决策 > 摘要兜底
    """
    lines: list[str] = ["压缩记忆基线"]

    explicit_sections = {
        "用户偏好": list(resolved_user_preferences or []),
        "项目约束": list(resolved_project_constraints or []),
        "最近风险": list(recent_risks or []),
    }

    for spec in _SECTION_SPECS:
        section_lines = _build_structured_section(
            working_memory=working_memory,
            title=spec.title,
            entry_types=spec.entry_types,
            max_entries=spec.max_entries,
            max_item_chars=spec.max_item_chars,
            max_section_tokens=spec.max_section_tokens,
            explicit_lines=explicit_sections.get(spec.title, []),
        )
        if not section_lines:
            continue
        if not _append_with_global_budget(lines, section_lines):
            break

    summary_section = _build_summary_fallback_section(
        older_history_summary=older_history_summary,
        previous_compact_memory_context=previous_compact_memory_context,
    )
    if summary_section:
        _append_with_global_budget(lines, summary_section)

    return "\n".join(lines).strip()


def _build_structured_section(
    *,
    working_memory: WorkingMemory,
    title: str,
    entry_types: tuple[str, ...],
    max_entries: int,
    max_item_chars: int,
    max_section_tokens: int,
    explicit_lines: list[str],
) -> list[str]:
    """按类型收集高价值条目，并在 section 内再做一次预算裁剪。"""
    entries = _collect_ranked_entries(
        working_memory=working_memory,
        entry_types=entry_types,
    ) if not explicit_lines else []
    if not explicit_lines and not entries:
        return []

    lines = [f"## {title}"]
    kept_count = 0
    candidate_sources = (
        [str(item).strip() for item in explicit_lines if str(item).strip()]
        if explicit_lines
        else [entry.content for entry in entries]
    )
    for content in candidate_sources:
        if kept_count >= max_entries:
            break

        candidate_line = f"- {_shorten(content, max_item_chars)}"
        candidate_lines = [*lines, candidate_line]
        if estimate_tokens("\n".join(candidate_lines)) > max_section_tokens:
            if kept_count == 0:
                lines.append(candidate_line)
            break

        lines.append(candidate_line)
        kept_count += 1

    return lines if len(lines) > 1 else []


def _collect_ranked_entries(
    *,
    working_memory: WorkingMemory,
    entry_types: tuple[str, ...],
) -> list[WorkingMemoryEntry]:
    """对不同类型的 working memory 条目做去重和优先级排序。"""
    raw_entries: list[WorkingMemoryEntry] = []
    for entry_type in entry_types:
        raw_entries.extend(working_memory.get_entries_by_type(entry_type))

    deduped = _dedupe_entries(raw_entries)
    deduped.sort(
        key=lambda entry: (
            _entry_type_priority(entry.entry_type),
            entry.importance,
            entry.created_at,
        ),
        reverse=True,
    )
    return deduped[:COMPACT_MEMORY_SECTION_ENTRY_LIMIT]


def _build_summary_fallback_section(
    *,
    older_history_summary: str,
    previous_compact_memory_context: str,
) -> list[str]:
    """
    构造低优先级兜底层。

    只有在高价值结构化信号放完之后，才用旧摘要和上一版 compact baseline 补足。
    """
    lines: list[str] = []

    normalized_summary = older_history_summary.strip()
    if normalized_summary:
        lines.append("## 旧对话摘要")
        lines.append(_shorten(normalized_summary, 140))

    carry_lines = _extract_previous_baseline_lines(previous_compact_memory_context)
    if carry_lines:
        lines.append("## 上次压缩延续")
        lines.extend(carry_lines[:3])

    return lines


def _extract_previous_baseline_lines(previous_compact_memory_context: str) -> list[str]:
    """从上一版 compact memory 中提取可延续的有效内容，避免递归复制标题。"""
    result: list[str] = []
    for raw_line in previous_compact_memory_context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "压缩记忆基线":
            continue
        if line.startswith("## "):
            continue
        result.append(f"- {_shorten(line.lstrip('- ').strip(), 90)}")
    return _dedupe_lines(result)


def _append_with_global_budget(lines: list[str], section_lines: list[str]) -> bool:
    """把 section 追加到最终上下文中，同时保证总 token 不超预算。"""
    candidate = "\n".join([*lines, *section_lines]).strip()
    if estimate_tokens(candidate) <= COMPACT_MEMORY_MAX_TOKENS:
        lines.extend(section_lines)
        return True

    # 如果整段放不下，尝试逐行压进去，尽量保留标题和前几个高价值 bullet。
    partial = list(lines)
    for line in section_lines:
        next_candidate = "\n".join([*partial, line]).strip()
        if estimate_tokens(next_candidate) > COMPACT_MEMORY_MAX_TOKENS:
            return False
        partial.append(line)

    lines[:] = partial
    return True


def _dedupe_entries(entries: list[WorkingMemoryEntry]) -> list[WorkingMemoryEntry]:
    """按归一化内容去重，避免同一句在多个来源里重复出现。"""
    result: list[WorkingMemoryEntry] = []
    seen: set[str] = set()
    for entry in entries:
        normalized = " ".join(entry.content.strip().lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(entry)
    return result


def _dedupe_lines(lines: list[str]) -> list[str]:
    """去掉重复的 carry-over 行。"""
    result: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = " ".join(line.strip().lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(line)
    return result


def _entry_type_priority(entry_type: str) -> int:
    """定义 compact memory 内部的类型优先级。"""
    priority_map = {
        "user_preference": 100,
        "project_constraint": 95,
        "recent_risk": 90,
        "error_context": 88,
        "key_decision": 82,
        "active_task": 78,
        "user_intent": 72,
    }
    return priority_map.get(entry_type, 50)


def _shorten(text: str, max_chars: int) -> str:
    """统一裁剪摘要段落，防止 compact memory 本身反过来撑爆上下文。"""
    normalized = " ".join(str(text).strip().split())
    if len(normalized) <= max_chars:
        return normalized
    suffix = " ...[已截断]"
    head_limit = max(0, max_chars - len(suffix))
    return f"{normalized[:head_limit]}{suffix}"
