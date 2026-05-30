from __future__ import annotations

from app.memory_pipeline import MemoryPipeline
from app.memory_store import MemoryEntry
from app.user_profile import load_user_profile
from app.working_memory import WorkingMemory

_CONSTRAINT_CATEGORIES = {"constraint", "convention"}
_CONSTRAINT_TAGS = {
    "constraint",
    "context_management",
    "token_budget",
    "architecture",
    "convention",
}


def resolve_user_preferences(*, workspace: str) -> list[str]:
    """从 USER.md 解析当前会话应继承的稳定用户偏好。"""
    profile = load_user_profile(workspace)
    if profile is None:
        return []
    return profile.to_preference_lines()[:6]


def resolve_project_constraints(
    *,
    memory_pipeline: MemoryPipeline | None,
    working_memory: WorkingMemory,
) -> list[str]:
    """
    从 project scope 长时记忆筛选稳定约束，并合并本轮运行时约束。

    这里把 project memory 当成长期真相源，
    `working_memory.project_constraint` 只作为本轮补充层。
    """
    lines: list[str] = []
    seen: set[str] = set()

    for entry in _load_project_constraint_entries(memory_pipeline):
        _append_unique_line(lines, seen, entry.content, limit=120)
        if len(lines) >= 8:
            break

    for entry in working_memory.get_entries_by_type("project_constraint"):
        _append_unique_line(lines, seen, entry.content, limit=120)
        if len(lines) >= 8:
            break

    return lines


def resolve_recent_risks(
    *,
    working_memory: WorkingMemory,
    cached_risks: list[str] | None = None,
) -> list[str]:
    """提取当前 session 最近需要记住的风险信号。"""
    lines: list[str] = []
    seen: set[str] = set()

    for entry in working_memory.get_entries_by_type("recent_risk"):
        _append_unique_line(lines, seen, entry.content, limit=120)
        if len(lines) >= 5:
            break

    for entry in working_memory.get_entries_by_type("error_context"):
        _append_unique_line(lines, seen, entry.content, limit=120)
        if len(lines) >= 5:
            break

    for item in cached_risks or []:
        _append_unique_line(lines, seen, item, limit=120)
        if len(lines) >= 5:
            break

    return lines


def _load_project_constraint_entries(memory_pipeline: MemoryPipeline | None) -> list[MemoryEntry]:
    """从 memory pipeline 挂着的 memory store 里筛选高置信项目约束。"""
    if memory_pipeline is None:
        return []

    read_pipeline = getattr(memory_pipeline, "read_pipeline", None)
    memory_store = getattr(read_pipeline, "memory_store", None)
    if memory_store is None:
        return []

    try:
        entries = memory_store.filter_memories(scope="project")
    except Exception:
        return []

    candidates: list[MemoryEntry] = []
    for entry in entries:
        if entry.archived:
            continue
        if float(entry.confidence) < 0.80:
            continue

        normalized_category = entry.category.strip().lower()
        normalized_tags = {
            tag.strip().lower()
            for tag in entry.tags
            if tag.strip()
        }
        if normalized_category not in _CONSTRAINT_CATEGORIES and not (
            normalized_tags & _CONSTRAINT_TAGS
        ):
            continue

        candidates.append(entry)

    candidates.sort(
        key=lambda item: (
            float(item.confidence),
            int(item.usage_count),
            float(item.updated_at),
        ),
        reverse=True,
    )
    return candidates


def _append_unique_line(lines: list[str], seen: set[str], raw_text: str, *, limit: int) -> None:
    """归一化去重后追加一条短文本。"""
    line = _shorten(raw_text, limit)
    normalized = " ".join(line.strip().lower().split())
    if not normalized or normalized in seen:
        return
    seen.add(normalized)
    lines.append(line)


def _shorten(text: str, max_chars: int) -> str:
    """裁短快照条目，避免 resolved 字段本身过长。"""
    normalized = " ".join(str(text).strip().split())
    if len(normalized) <= max_chars:
        return normalized
    suffix = " ...[已截断]"
    head_limit = max(0, max_chars - len(suffix))
    return f"{normalized[:head_limit]}{suffix}"
